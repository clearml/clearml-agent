//! Kiro, AWS's Kiro CLI backend.
//!
//! kiro-cli chats with `runtime.<region>.kiro.dev`, one request per model call
//! (a tool turn is two requests), over one of two wires. Neither carries token
//! counts, so a Kiro generation is always byte-estimated by the caller.
//!
//! kiro-cli 3.x: `POST /v1/responses`, the OpenAI Responses API. The JSON
//! request names the `model`; the `text/event-stream` answer carries
//! `response.output_text.delta` and `response.function_call_arguments.delta`
//! (the generated text and tool arguments), a `response.output_item.added`
//! per `function_call`, and a closing `response.completed` whose `response`
//! holds the served `model` and a `usage` of only `kiro:credits` and
//! `kiro:context_usage_percentage`.
//!
//! kiro-cli 2.x: `POST /` with
//! `x-amz-target: AmazonCodeWhispererStreamingService.GenerateAssistantResponse`,
//! a JSON request answered with an `application/vnd.amazon.eventstream` body.
//! The stream is binary frames, each a 12-byte prelude (total length, headers
//! length, CRC), typed headers and a JSON payload. Its events:
//! `assistantResponseEvent` (`content`, a text delta), `toolUseEvent` (one call
//! streamed as several frames sharing a `toolUseId`, each carrying the `name`
//! and an `input` fragment), `meteringEvent` (the request's charge in Kiro
//! credits), plus `initial-response`, `reasoningContentEvent` (encrypted, not
//! countable), `metadataEvent` and `contextUsageEvent`, which carry nothing
//! measured, and no served model.

use serde_json::Value;

/// Event-stream frame prelude: total length, headers length, prelude CRC.
const PRELUDE_LEN: usize = 12;

/// Cap on one frame. Kiro's frames are small deltas; a larger length means the
/// bytes are not an event stream.
const MAX_FRAME: usize = 1024 * 1024;

/// Kiro's regional chat endpoint, `runtime.<region>.kiro.dev`. The sibling
/// `management.*` (profile, model list, credit limits) and `telemetry` hosts
/// carry no generations and stay unmetered.
pub fn is_runtime_host(host: &str) -> bool {
    host.strip_prefix("runtime.")
        .and_then(|rest| rest.strip_suffix(".kiro.dev"))
        .is_some_and(|region| !region.is_empty() && !region.contains('.'))
}

/// The model the request asked for: 3.x's `model`, or 2.x's `modelId` (`auto`
/// unless the user picked one).
pub fn model_from_request(body: &[u8]) -> Option<String> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let m = v
        .get("model")
        .or_else(|| v.pointer("/conversationState/currentMessage/userInputMessage/modelId"))
        .and_then(Value::as_str)?
        .trim();
    (!m.is_empty()).then(|| m.to_string())
}

/// What one Kiro response carried, decoded as the body arrives: event-stream
/// frames (2.x) through `feed`, SSE `data:` payloads (3.x) through `feed_sse`.
#[derive(Debug, Default)]
pub struct Response {
    /// Bytes of a not yet complete frame.
    buf: Vec<u8>,
    /// The `toolUseId` of the tool call being streamed; a new id is a new call.
    tool_id: Option<String>,
    /// Generated text plus tool-argument bytes, for the output estimate.
    pub output_text_bytes: u64,
    pub tool_call_names: Vec<String>,
    /// The request's charge in Kiro credits.
    pub credits: Option<f64>,
    /// The served model, which only the 3.x wire reports.
    pub model: Option<String>,
}

impl Response {
    /// Decode every complete frame in `buf` plus what came before, keeping the
    /// partial tail. Returns false once the bytes turn out not to be an event
    /// stream; the caller then stops feeding and reports nothing. Frame CRCs are
    /// not checked: the stream already passed TLS, and a corrupt frame only skews
    /// an estimate.
    pub fn feed(&mut self, body: &[u8]) -> bool {
        self.buf.extend_from_slice(body);
        let mut at = 0;
        while self.buf.len() - at >= PRELUDE_LEN {
            let b = &self.buf[at..];
            let total = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
            let headers_len = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
            if total < PRELUDE_LEN + 4 + headers_len || total > MAX_FRAME {
                self.buf = Vec::new();
                return false;
            }
            if b.len() < total {
                break;
            }
            let event = event_type(&b[PRELUDE_LEN..PRELUDE_LEN + headers_len]).map(str::to_string);
            let payload = b[PRELUDE_LEN + headers_len..total - 4].to_vec();
            if let Some(event) = event {
                self.on_event(&event, &payload);
            }
            at += total;
        }
        self.buf.drain(..at);
        true
    }

    /// Take one SSE `data:` payload of the 3.x wire. Only the events that carry
    /// something counted are parsed; the rest are skipped by a byte scan.
    pub fn feed_sse(&mut self, payload: &[u8]) {
        let wanted = [&b".delta\""[..], b"\"function_call\"", b"response.completed"];
        if !wanted.iter().any(|w| payload.windows(w.len()).any(|x| x == *w)) {
            return;
        }
        let v = match serde_json::from_slice::<Value>(payload) {
            Ok(v) => v,
            Err(_) => return,
        };
        match v.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") | Some("response.function_call_arguments.delta") => {
                self.output_text_bytes += v.get("delta").and_then(Value::as_str).map_or(0, |d| d.len() as u64);
            }
            Some("response.output_item.added") => {
                let item = v.get("item");
                if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("function_call") {
                    let name = item.and_then(|i| i.get("name")).and_then(Value::as_str).unwrap_or("unknown");
                    self.tool_call_names.push(name.to_string());
                }
            }
            Some("response.completed") => {
                let r = v.get("response");
                self.model = r
                    .and_then(|r| r.get("model"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                    .map(str::to_string);
                self.credits = r
                    .and_then(|r| r.pointer("/usage/kiro:credits/value"))
                    .and_then(Value::as_f64);
            }
            _ => {}
        }
    }

    fn on_event(&mut self, event: &str, payload: &[u8]) {
        let v = match serde_json::from_slice::<Value>(payload) {
            Ok(v) => v,
            Err(_) => return,
        };
        let text_len = |key: &str| v.get(key).and_then(Value::as_str).map_or(0, |t| t.len() as u64);
        match event {
            "assistantResponseEvent" => self.output_text_bytes += text_len("content"),
            "toolUseEvent" => {
                self.output_text_bytes += text_len("input");
                let id = v.get("toolUseId").and_then(Value::as_str);
                if id.is_some() && id != self.tool_id.as_deref() {
                    self.tool_id = id.map(str::to_string);
                    let name = v.get("name").and_then(Value::as_str).unwrap_or("unknown");
                    self.tool_call_names.push(name.to_string());
                }
            }
            "meteringEvent" => {
                if let Some(u) = v.get("usage").and_then(Value::as_f64) {
                    self.credits = Some(self.credits.unwrap_or(0.0) + u);
                }
            }
            _ => {}
        }
    }
}

/// The `:event-type` header of a frame. Headers are a name (1-byte length), a
/// value type and a value; only string values (type 7, 2-byte length) are read,
/// and an unknown type ends the scan.
fn event_type(mut h: &[u8]) -> Option<&str> {
    while let Some((&name_len, rest)) = h.split_first() {
        let name = rest.get(..name_len as usize)?;
        let (&kind, rest) = rest.get(name_len as usize..)?.split_first()?;
        if kind != 7 {
            return None;
        }
        let len = u16::from_be_bytes([*rest.first()?, *rest.get(1)?]) as usize;
        let value = rest.get(2..2 + len)?;
        if name == b":event-type" {
            return std::str::from_utf8(value).ok();
        }
        h = &rest[2 + len..];
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_scan::{model_from_request as model_for, provider_for_host, Provider, RespParse};

    /// One event-stream frame: prelude, `:event-type` / `:message-type` string
    /// headers, JSON payload. CRCs are zero; the decoder doesn't check them.
    fn frame(event: &str, payload: &str) -> Vec<u8> {
        let mut headers = Vec::new();
        for (name, value) in [(":event-type", event), (":message-type", "event")] {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            headers.push(7);
            headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
            headers.extend_from_slice(value.as_bytes());
        }
        let total = 12 + headers.len() + payload.len() + 4;
        let mut f = Vec::new();
        f.extend_from_slice(&(total as u32).to_be_bytes());
        f.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        f.extend_from_slice(&[0; 4]);
        f.extend_from_slice(&headers);
        f.extend_from_slice(payload.as_bytes());
        f.extend_from_slice(&[0; 4]);
        f
    }

    /// A chunked `application/vnd.amazon.eventstream` response carrying `body`
    /// split into chunks of `chunk` bytes, as Kiro's runtime endpoint sends it.
    fn response(body: &[u8], chunk: usize) -> Vec<u8> {
        let mut r = b"HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\n\
Transfer-Encoding: chunked\r\nx-amzn-RequestId: a02db2ad\r\n\r\n"
            .to_vec();
        for c in body.chunks(chunk) {
            r.extend_from_slice(format!("{:x}\r\n", c.len()).as_bytes());
            r.extend_from_slice(c);
            r.extend_from_slice(b"\r\n");
        }
        r.extend_from_slice(b"0\r\n\r\n");
        r
    }

    /// The frames of a captured kiro-cli 2.25 tool-use turn.
    fn tool_turn() -> Vec<u8> {
        let id = "toolu_bdrk_01J9o8bY8biEdPftfi58oPpT";
        let mut body = frame("initial-response", r#"{"conversationId":""}"#);
        for t in ["I'll run", " that command for", " you."] {
            body.extend(frame("assistantResponseEvent", &format!(r#"{{"content":"{}"}}"#, t)));
        }
        body.extend(frame("toolUseEvent", &format!(r#"{{"name":"shell","toolUseId":"{}"}}"#, id)));
        for part in [r#"{\"command\""#, r#": \"ls /etc\"}"#] {
            body.extend(frame(
                "toolUseEvent",
                &format!(r#"{{"input":"{}","name":"shell","toolUseId":"{}"}}"#, part, id),
            ));
        }
        body.extend(frame("toolUseEvent", &format!(r#"{{"name":"shell","stop":true,"toolUseId":"{}"}}"#, id)));
        body.extend(frame("reasoningContentEvent", r#"{"redactedContent":"LktUUn5+ZXlK"}"#));
        body.extend(frame("metadataEvent", r#"{"stopReason":"TOOL_USE"}"#));
        body.extend(frame("contextUsageEvent", r#"{"contextUsagePercentage":1.6355000734329224}"#));
        body.extend(frame("meteringEvent", r#"{"unit":"credit","unitPlural":"credits","usage":0.04089855572139304}"#));
        body
    }

    fn feed_all(resp: &[u8], provider: Provider) -> RespParse {
        let mut p = RespParse::new(provider);
        p.feed(resp);
        p
    }

    #[test]
    fn runtime_host_is_a_provider() {
        assert_eq!(provider_for_host("runtime.us-east-1.kiro.dev"), Some(Provider::Kiro));
        assert_eq!(provider_for_host("runtime.eu-central-1.kiro.dev"), Some(Provider::Kiro));
        assert_eq!(provider_for_host("management.us-east-1.kiro.dev"), None);
        assert_eq!(provider_for_host("prod.us-east-1.telemetry-v2.kiro.dev"), None);
        assert_eq!(provider_for_host("runtime.kiro.dev"), None);
        assert_eq!(provider_for_host("runtime.a.b.kiro.dev"), None);
    }

    #[test]
    fn event_stream_yields_text_tools_and_credits() {
        // Split mid-frame and mid-chunk to exercise the partial-frame carry.
        for chunk in [1, 7, 64, 4096] {
            let resp = response(&tool_turn(), chunk);
            let mut p = RespParse::new(Provider::Kiro);
            for piece in resp.chunks(13) {
                p.feed(piece);
            }
            assert!(p.is_complete(), "chunk={}", chunk);
            let f = p.finalize();
            let text = "I'll run that command for you.".len() + r#"{"command": "ls /etc"}"#.len();
            assert_eq!(f.output_text_bytes, Some(text as u64), "chunk={}", chunk);
            assert_eq!(f.tool_calls, 1);
            assert_eq!(f.tool_call_names, vec!["shell".to_string()]);
            assert_eq!(f.credits, Some(0.04089855572139304));
            assert_eq!((f.tokens_in, f.tokens_out, f.model), (None, None, None));
        }
    }

    #[test]
    fn two_tool_calls_count_twice() {
        let mut body = Vec::new();
        for id in ["toolu_a", "toolu_b"] {
            body.extend(frame("toolUseEvent", &format!(r#"{{"name":"read","toolUseId":"{}"}}"#, id)));
            body.extend(frame("toolUseEvent", &format!(r#"{{"input":"{{}}","name":"read","toolUseId":"{}"}}"#, id)));
        }
        let mut s = Response::default();
        assert!(s.feed(&body));
        assert_eq!(s.tool_call_names, vec!["read".to_string(), "read".to_string()]);
    }

    #[test]
    fn json_error_is_not_a_generation() {
        let resp = b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/x-amz-json-1.0\r\n\
Content-Length: 2\r\n\r\n{}";
        let f = feed_all(resp, Provider::Kiro).finalize();
        assert_eq!(f.output_text_bytes, None);
    }

    #[test]
    fn event_stream_from_another_provider_is_not_parsed() {
        let f = feed_all(&response(&tool_turn(), 4096), Provider::Anthropic).finalize();
        assert_eq!((f.output_text_bytes, f.credits, f.tool_calls), (None, None, 0));
    }

    #[test]
    fn garbage_is_not_a_generation() {
        assert!(!Response::default().feed(&[0xff; 64]));
        let f = feed_all(&response(&[0xff; 64], 4096), Provider::Kiro).finalize();
        assert_eq!((f.output_text_bytes, f.tool_calls), (None, 0));
    }

    #[test]
    fn model_comes_from_the_request() {
        let body = br#"{"conversationState":{"conversationId":"c","currentMessage":{"userInputMessage":{"content":"hi","origin":"KIRO_CLI","modelId":"claude-haiku-4.5"}},"chatTriggerType":"MANUAL"}}"#;
        assert_eq!(model_for(Provider::Kiro, body, Some("/")).as_deref(), Some("claude-haiku-4.5"));
        assert!(model_for(Provider::Kiro, b"{}", Some("/")).is_none());
    }

    /// A chunked `text/event-stream` response, as Kiro 3.x's `/v1/responses`
    /// sends it, split into chunks of `chunk` bytes.
    fn sse_response(events: &[&str], chunk: usize) -> Vec<u8> {
        let mut body = Vec::new();
        for e in events {
            body.extend_from_slice(format!("data: {}\n\n", e).as_bytes());
        }
        body.extend_from_slice(b"data: [DONE]\n\n");
        let mut r = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for c in body.chunks(chunk) {
            r.extend_from_slice(format!("{:x}\r\n", c.len()).as_bytes());
            r.extend_from_slice(c);
            r.extend_from_slice(b"\r\n");
        }
        r.extend_from_slice(b"0\r\n\r\n");
        r
    }

    /// The events of a captured kiro-cli 3.0 tool-call turn, trimmed.
    const RESPONSES_TOOL_TURN: &[&str] = &[
        r#"{"type":"response.created","response":{"id":"resp_ee52","model":"gpt-5.6-sol","status":"in_progress"}}"#,
        r#"{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"function_call","id":"fc_5839","call_id":"call_c39c","name":"execute_bash"}}"#,
        r#"{"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_5839","output_index":0,"delta":"{\""}"#,
        r#"{"type":"response.function_call_arguments.delta","sequence_number":4,"item_id":"fc_5839","output_index":0,"delta":"command\":\"ls /etc\"}"}"#,
        r#"{"type":"response.function_call_arguments.done","sequence_number":5,"item_id":"fc_5839","output_index":0,"arguments":"{\"command\":\"ls /etc\"}"}"#,
        r#"{"type":"response.output_text.delta","sequence_number":6,"item_id":"msg_89e3","output_index":1,"content_index":0,"delta":"Running it."}"#,
        r#"{"type":"response.completed","response":{"id":"resp_ee52","model":"gpt-5.6-sol","status":"completed","usage":{"kiro:credits":{"value":1.0524348157213932,"unit":"credit","unit_plural":"credits"},"kiro:context_usage_percentage":1.4392495}}}"#,
    ];

    #[test]
    fn responses_wire_yields_text_tools_model_and_credits() {
        for chunk in [1, 9, 4096] {
            let resp = sse_response(RESPONSES_TOOL_TURN, chunk);
            let mut p = RespParse::new(Provider::Kiro);
            for piece in resp.chunks(11) {
                p.feed(piece);
            }
            assert!(p.is_complete(), "chunk={}", chunk);
            let f = p.finalize();
            let text = r#"{"command":"ls /etc"}"#.len() + "Running it.".len();
            assert_eq!(f.output_text_bytes, Some(text as u64), "chunk={}", chunk);
            assert_eq!(f.tool_call_names, vec!["execute_bash".to_string()]);
            assert_eq!(f.model.as_deref(), Some("gpt-5.6-sol"));
            assert_eq!(f.credits, Some(1.0524348157213932));
            assert_eq!((f.tokens_in, f.tokens_out), (None, None));
        }
    }

    #[test]
    fn responses_wire_on_another_provider_is_untouched() {
        // The same SSE on api.openai.com goes through the OpenAI parser: no
        // Kiro credits, and no Kiro-style text count without the proxy's opt-in.
        let f = feed_all(&sse_response(RESPONSES_TOOL_TURN, 4096), Provider::OpenAi).finalize();
        assert_eq!((f.credits, f.output_text_bytes), (None, None));
    }

    #[test]
    fn model_from_a_responses_request() {
        let body = br#"{"model":"gpt-5.6-sol","input":[],"stream":true,"tools":[]}"#;
        assert_eq!(model_for(Provider::Kiro, body, Some("/v1/responses")).as_deref(), Some("gpt-5.6-sol"));
    }
}
