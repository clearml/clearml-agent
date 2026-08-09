//! Process-global ring buffer of the last N full request/response pairs, plus
//! the redaction + chunked-emit + dump logic for the call-history feature.
//!
//! Mirrors `session.rs`: a process-global `OnceLock<Mutex<...>>` the hot path
//! appends to (from `state::capture_call_history`) and the reporter's poll
//! thread drains (on a `dump` trigger). Independent of the metering Event
//! channel — in `collect` mode nothing crosses the channel until a dump.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use base64::Engine as _;
use clearml_snug_messages::{CallHistoryMode, Event};

use crate::control;
use crate::meter;

/// Header names (ASCII case-insensitive) whose values are masked before a
/// captured request is buffered or printed.
const SENSITIVE_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "proxy-authorization",
    "cookie",
    "x-goog-api-key",
    "openai-organization",
];

/// Query-string parameter names (ASCII case-insensitive) whose values are
/// masked in the request line (Gemini passes the key as `?key=...`).
const SENSITIVE_QUERY_PARAMS: &[&str] = &["key", "access_token"];

const REDACTED: &str = "<redacted>";

/// One captured request/response pair. `request` is already redacted; both
/// directions are capped at `control::call_history_cap_bytes()`.
pub struct CapturedPair {
    pub conn_id: u64,
    pub ts_ms: u64,
    pub seq: u64,
    pub host: String,
    pub path: String,
    pub method: String,
    pub status: Option<u16>,
    pub request: Vec<u8>,
    pub response: Vec<u8>,
    pub request_truncated: bool,
    pub response_truncated: bool,
    pub response_compressed: bool,
    /// The task's running chat ordinal for this request ("1","2",…), shared
    /// from the `RequestCompleted` event so it's computed once. Rendered in the
    /// console header; matches the SCALARS series.
    pub chat_id: Option<String>,
}

struct Ring {
    pairs: VecDeque<CapturedPair>,
    seq: u64,
}

static RING: OnceLock<Mutex<Ring>> = OnceLock::new();

fn ring() -> &'static Mutex<Ring> {
    RING.get_or_init(|| {
        Mutex::new(Ring {
            pairs: VecDeque::new(),
            seq: 0,
        })
    })
}

/// Assign the next monotonic capture sequence number. Used by `Continuous` mode,
/// which emits directly without buffering, so its entries share the ring's
/// ordering.
pub fn next_seq() -> u64 {
    match ring().lock() {
        Ok(mut r) => {
            r.seq += 1;
            r.seq
        }
        Err(_) => 0,
    }
}

/// Hot-path append (`Collect` / `Dump` modes). Stamps the pair with the next
/// seq and evicts the oldest when the ring is full (sliding window).
pub fn push(mut p: CapturedPair) {
    if let Ok(mut r) = ring().lock() {
        r.seq += 1;
        p.seq = r.seq;
        let cap = control::call_history_buffer_size();
        while r.pairs.len() >= cap {
            r.pairs.pop_front();
        }
        r.pairs.push_back(p);
    }
}

/// Drain a snapshot of the buffered pairs, clearing the ring (post-dump the
/// sliding window resumes from empty).
pub fn drain() -> Vec<CapturedPair> {
    match ring().lock() {
        Ok(mut r) => r.pairs.drain(..).collect(),
        Err(_) => Vec::new(),
    }
}

/// Poll-callback target for a `_snug_call_history` runtime-property change. Sets
/// the mode and emits a one-line `[SNUG-CALL]` NOTICE so each transition is
/// visible on the console (the flips are otherwise silent). On `Dump` it prints
/// the backlog once then leaves the steady mode as `Collect` (keep sliding); the
/// poll thread only calls this on an actual change, so a property parked at
/// `dump` does not re-dump every tick. Runs on the poll thread (calls
/// `meter::emit`, the same channel the hot path uses).
pub fn set_call_history_mode_and_maybe_dump(mode: CallHistoryMode) {
    let prev = control::call_history_mode();
    match mode {
        CallHistoryMode::Dump => {
            // Dump settles to Collect (the reporter also reverts the property).
            control::set_call_history_mode(CallHistoryMode::Collect);
            let pairs = drain();
            emit_notice(format!(
                "dump: {} call(s) (mode -> collect)",
                pairs.len()
            ));
            for pair in pairs {
                meter::emit(pair_to_event(&pair));
            }
        }
        other => {
            control::set_call_history_mode(other);
            emit_notice(format!(
                "mode -> {} (was {})",
                mode_name(other),
                mode_name(prev)
            ));
        }
    }
}

fn mode_name(m: CallHistoryMode) -> &'static str {
    match m {
        CallHistoryMode::Off => "off",
        CallHistoryMode::Collect => "collect",
        CallHistoryMode::Dump => "dump",
        CallHistoryMode::Continuous => "continuous",
    }
}

/// Emit a one-line call-history notice event (rendered as a `[SNUG-CALL]` row).
fn emit_notice(text: String) {
    meter::emit(Event::CallHistoryNotice {
        ts_ms: Event::now_ts_ms(),
        text,
    });
}

/// Build a single `CallHistoryEntry` event carrying the WHOLE pair (full request
/// + full response, base64). One event per pair so the reporter renders the
/// request and response as ONE contiguous console block — concurrent metering
/// events on the shared channel can't split a pair across two events. Payloads
/// are capped upstream (`control::call_history_cap_bytes`).
pub fn pair_to_event(p: &CapturedPair) -> Event {
    Event::CallHistoryEntry {
        conn_id: p.conn_id,
        ts_ms: p.ts_ms,
        seq: p.seq,
        host: p.host.clone(),
        path: p.path.clone(),
        method: p.method.clone(),
        status: p.status,
        request_b64: b64(&p.request),
        response_b64: b64(&p.response),
        request_truncated: p.request_truncated,
        response_truncated: p.response_truncated,
        response_compressed: p.response_compressed,
        chat_id: p.chat_id.clone(),
    }
}

/// Parse the HTTP status code from a captured response head (`HTTP/1.1 200 …`).
pub fn status_from_head(resp: &[u8]) -> Option<u16> {
    let line_end = find_subsequence(resp, b"\r\n").unwrap_or(resp.len());
    let line = std::str::from_utf8(&resp[..line_end]).ok()?;
    // "HTTP/1.1 200 OK" -> the second whitespace-separated token.
    line.split_whitespace().nth(1)?.parse().ok()
}

/// True when the captured response declares a `Content-Encoding` other than
/// `identity` (its body bytes are compressed/opaque in the dump).
pub fn response_is_compressed(resp: &[u8]) -> bool {
    let head_end = find_subsequence(resp, b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(resp.len());
    let head = match std::str::from_utf8(&resp[..head_end]) {
        Ok(h) => h,
        Err(_) => return false,
    };
    for line in head.split("\r\n") {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-encoding") {
                let v = value.trim().to_ascii_lowercase();
                return !v.is_empty() && v != "identity";
            }
        }
    }
    false
}

/// Mask credentials in a captured request's header section + request-line query
/// string, replacing their values with `<redacted>`. Only the head (up to the
/// first CRLFCRLF) is scanned; the body is left untouched. Pure + testable.
pub fn redact_request(bytes: &[u8]) -> Vec<u8> {
    let head_end = find_subsequence(bytes, b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(bytes.len());

    let mut out = Vec::with_capacity(bytes.len());
    let mut pos = 0;
    let mut is_first_line = true;
    while pos < head_end {
        let (line_end, next) = match find_subsequence(&bytes[pos..head_end], b"\r\n") {
            Some(r) => (pos + r, pos + r + 2),
            None => (head_end, head_end),
        };
        let line = &bytes[pos..line_end];
        if is_first_line {
            is_first_line = false;
            out.extend_from_slice(&redact_request_line(line));
        } else if let Some(red) = redact_header_line(line) {
            out.extend_from_slice(&red);
        } else {
            out.extend_from_slice(line);
        }
        // Re-emit the exact line terminator bytes the split consumed.
        out.extend_from_slice(&bytes[line_end..next]);
        pos = next;
    }
    out.extend_from_slice(&bytes[head_end..]);
    out
}

fn redact_header_line(line: &[u8]) -> Option<Vec<u8>> {
    let colon = line.iter().position(|&b| b == b':')?;
    let name = &line[..colon];
    let name_str = std::str::from_utf8(name).ok()?.trim().to_ascii_lowercase();
    if SENSITIVE_HEADERS.contains(&name_str.as_str()) {
        let mut out = Vec::with_capacity(name.len() + 2 + REDACTED.len());
        out.extend_from_slice(name);
        out.extend_from_slice(b": ");
        out.extend_from_slice(REDACTED.as_bytes());
        Some(out)
    } else {
        None
    }
}

fn redact_request_line(line: &[u8]) -> Vec<u8> {
    let s = match std::str::from_utf8(line) {
        Ok(s) => s,
        Err(_) => return line.to_vec(),
    };
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'?' || c == b'&' {
            out.push(c as char);
            let after = &s[i + 1..];
            if let Some(p) = SENSITIVE_QUERY_PARAMS.iter().find(|p| {
                let pb = p.as_bytes();
                after.len() > pb.len()
                    && after.as_bytes()[..pb.len()].eq_ignore_ascii_case(pb)
                    && after.as_bytes()[pb.len()] == b'='
            }) {
                out.push_str(&after[..p.len() + 1]); // "name="
                out.push_str(REDACTED);
                let mut j = i + 1 + p.len() + 1;
                while j < bytes.len() && bytes[j] != b'&' && !(bytes[j] as char).is_whitespace() {
                    j += 1;
                }
                i = j;
                continue;
            }
            i += 1;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out.into_bytes()
}

fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(seq: u64, req: &[u8], resp: &[u8]) -> CapturedPair {
        CapturedPair {
            conn_id: 1,
            ts_ms: 1,
            seq,
            host: "api.anthropic.com".into(),
            path: "/v1/messages".into(),
            method: "POST".into(),
            status: Some(200),
            request: req.to_vec(),
            response: resp.to_vec(),
            request_truncated: false,
            response_truncated: false,
            response_compressed: false,
            chat_id: Some("7".into()),
        }
    }

    #[test]
    fn ring_evicts_oldest_past_capacity() {
        // The default buffer size is 50 (no env override in tests).
        for n in 0..60u64 {
            push(pair(0, format!("req{n}").as_bytes(), b"resp"));
        }
        let drained = drain();
        assert_eq!(drained.len(), 50);
        // Oldest 10 evicted: surviving seqs are the last 50 assigned. seq is
        // assigned by push() monotonically, so the min surviving seq > 10.
        let min_seq = drained.iter().map(|p| p.seq).min().unwrap();
        assert!(min_seq >= 11, "min surviving seq was {min_seq}");
        // drain cleared the ring.
        assert!(drain().is_empty());
    }

    #[test]
    fn redact_authorization_bearer() {
        let req = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\nAuthorization: Bearer sk-secret-123\r\nContent-Type: application/json\r\n\r\n{\"model\":\"x\"}";
        let out = redact_request(req);
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("Authorization: <redacted>"), "got: {s}");
        assert!(!s.contains("sk-secret-123"));
        // Non-sensitive headers + body untouched.
        assert!(s.contains("Content-Type: application/json"));
        assert!(s.contains("{\"model\":\"x\"}"));
        assert!(s.starts_with("POST /v1/messages HTTP/1.1\r\n"));
    }

    #[test]
    fn redact_x_api_key_case_insensitive() {
        let req = b"POST /v1/messages HTTP/1.1\r\nX-Api-Key: sk-ant-xyz\r\n\r\n";
        let s = String::from_utf8(redact_request(req)).unwrap();
        assert!(s.contains("X-Api-Key: <redacted>"));
        assert!(!s.contains("sk-ant-xyz"));
    }

    #[test]
    fn redact_gemini_key_in_query_string() {
        let req = b"POST /v1beta/models/gemini:generateContent?key=AItopsecret&alt=sse HTTP/1.1\r\nHost: generativelanguage.googleapis.com\r\n\r\n";
        let s = String::from_utf8(redact_request(req)).unwrap();
        assert!(s.contains("?key=<redacted>"), "got: {s}");
        assert!(s.contains("&alt=sse"), "non-sensitive params preserved: {s}");
        assert!(!s.contains("AItopsecret"));
    }

    #[test]
    fn redact_leaves_body_authorization_untouched() {
        // An "Authorization:" string inside the BODY must not be redacted (only
        // the header section is scanned).
        let req = b"POST /v1/x HTTP/1.1\r\nHost: h\r\n\r\nAuthorization: not-a-header";
        let s = String::from_utf8(redact_request(req)).unwrap();
        assert!(s.ends_with("\r\n\r\nAuthorization: not-a-header"));
    }

    #[test]
    fn status_and_compression_parsed_from_head() {
        assert_eq!(status_from_head(b"HTTP/1.1 200 OK\r\n\r\n"), Some(200));
        assert_eq!(status_from_head(b"HTTP/1.1 429 Too Many\r\n\r\n"), Some(429));
        assert_eq!(status_from_head(b"garbage"), None);
        assert!(response_is_compressed(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n\r\n..."
        ));
        assert!(!response_is_compressed(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: identity\r\n\r\n..."
        ));
        assert!(!response_is_compressed(b"HTTP/1.1 200 OK\r\n\r\n..."));
    }

    #[test]
    fn pair_to_event_carries_both_directions_in_one_event() {
        // One event per pair (so the reporter renders it contiguously) carrying
        // BOTH the request and response payloads.
        match pair_to_event(&pair(7, b"req", b"resp")) {
            Event::CallHistoryEntry {
                seq,
                request_b64,
                response_b64,
                chat_id,
                ..
            } => {
                assert_eq!(seq, 7);
                assert!(!request_b64.is_empty());
                assert!(!response_b64.is_empty());
                // The chat ordinal flows through to the event for the header.
                assert_eq!(chat_id.as_deref(), Some("7"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn pair_to_event_handles_large_and_empty_payloads() {
        // Large payloads ride a single event (capped upstream), never split.
        let big = vec![b'x'; 200_000];
        match pair_to_event(&pair(1, &big, b"")) {
            Event::CallHistoryEntry {
                request_b64,
                response_b64,
                ..
            } => {
                assert!(!request_b64.is_empty());
                assert!(response_b64.is_empty()); // empty direction omitted
            }
            _ => panic!("wrong variant"),
        }
        // Both directions empty still yields a (header-only) event.
        match pair_to_event(&pair(1, b"", b"")) {
            Event::CallHistoryEntry {
                request_b64,
                response_b64,
                ..
            } => assert!(request_b64.is_empty() && response_b64.is_empty()),
            _ => panic!("wrong variant"),
        }
    }
}
