//! Response-body scanning to extract provider-reported token usage.
//!
//! The shim sees decrypted (post-TLS) but HTTP-layer-encoded bytes on
//! `SSL_read`. To read the real `usage` an LLM provider reports (instead of
//! the byte-ratio estimate), we feed each read chunk through a small,
//! per-response pipeline:
//!
//!   raw read bytes
//!     -> [1] accumulate + parse the HTTP response head (status, content-type,
//!            transfer-encoding) once
//!     -> [2] de-chunk if `Transfer-Encoding: chunked`
//!     -> [3a] SSE: split on lines, parse each `data:` event
//!        [3b] JSON: accumulate (capped), parse once at completion
//!     -> [4] extract the provider's usage fields
//!
//! Compression is handled upstream: the shim forces `Accept-Encoding:
//! identity` on whitelisted requests (see `inject.rs`), so bodies arrive as
//! plaintext JSON/SSE — no decompressor lives in the user's address space.
//!
//! Hot-path note: this only runs when a reporting sink is enabled
//! (`control::parse_usage_enabled()`) and the host is a known provider. A
//! cheap `"usage"` substring pre-filter gates the (relatively expensive)
//! `serde_json` parse, so the hundreds of `content_block_delta` SSE events in
//! a long completion are skipped with a byte scan — only the ~2 usage-bearing
//! events are actually parsed. All buffers are capped; any parse failure
//! simply leaves the measured value unset and the caller falls back to the
//! byte estimate.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use serde_json::Value;

use crate::parser;

/// Cap on bytes accumulated while hunting for the end of the response head.
/// Real heads are well under this; the cap guards against a response that
/// never presents a `\r\n\r\n`.
const MAX_HEAD_BYTES: usize = 16 * 1024;

/// Cap on a single un-terminated SSE line. Usage events are tiny; this only
/// bounds a pathological never-newline stream. Large-but-normal events
/// (e.g. a big `tool_use` block) still fit.
const MAX_LINE_BYTES: usize = 1024 * 1024;

/// Cap on the accumulated non-streaming JSON body. The `usage` object sits at
/// the end of the document, so truncating past this means we fall back to the
/// estimate — acceptable for the rare multi-megabyte non-streaming response.
const JSON_CAP: usize = 1024 * 1024;

/// LLM providers whose response `usage` schema we know how to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAi,
    Gemini,
}

/// Map a request `Host` to a known provider, or `None` if we don't have a
/// usage schema for it (the caller then skips parsing and keeps the estimate).
pub fn provider_for_host(host: &str) -> Option<Provider> {
    match host {
        "api.anthropic.com" => Some(Provider::Anthropic),
        "api.openai.com" => Some(Provider::OpenAi),
        "generativelanguage.googleapis.com" => Some(Provider::Gemini),
        _ => None,
    }
}

/// Body framing decided from the response `Content-Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RespMode {
    /// `text/event-stream` — parse `data:` lines incrementally.
    Sse,
    /// `application/json` — accumulate then parse once.
    Json,
    /// Anything else (or head we couldn't classify) — don't parse a body.
    Unknown,
}

/// What a completed response yields to the caller: measured token usage (or
/// `None` per field, so the caller falls back to the estimate), the count of
/// tool calls the model requested, and the model the provider served the
/// request with (`None` when the response didn't echo one).
#[derive(Debug, Default, Clone)]
pub struct Finalized {
    pub tokens_in: Option<u64>,
    pub tokens_out: Option<u64>,
    /// Anthropic prompt-cache breakdown of the input `tokens_in` already folds
    /// in: tokens served from the cache (`cache_read_input_tokens`) and tokens
    /// written to it (`cache_creation_input_tokens`). Surfaced separately so the
    /// metrics sink can plot fresh vs cache-read vs cache-write input, while
    /// `tokens_in` stays the summed billable total. `None` for OpenAI/Gemini
    /// (they report a cache-inclusive prompt total with no breakdown) and where
    /// no usage was parsed.
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub tool_calls: u64,
    pub tool_call_names: Vec<String>,
    /// The model name echoed in the response (Anthropic/OpenAI `model`, Gemini
    /// `modelVersion`), for per-model usage attribution. `None` when the
    /// response carried no model (e.g. an error body, or an unparsed stream);
    /// the caller then falls back to the model named in the request.
    pub model: Option<String>,
    /// Byte length of the GENERATED text from an SSE response — the sum of the
    /// Anthropic content deltas (`text`/`thinking`/tool-arg), NOT the whole SSE
    /// envelope. Refines the output-token ESTIMATE when a response streams SSE:
    /// estimating from the full decompressed body would overcount, since the
    /// `event:`/`data:`/JSON framing dwarfs the actual text. `Some` only when
    /// output-text counting was enabled (the proxy sets it); `None` for the shim,
    /// and `None` for any non-SSE response — including the common consumer-wire
    /// case, where a `POST .../completion` returns a buffered JSON ack rather than
    /// an SSE stream, so the estimate falls back to the whole body. Unused when
    /// usage is measured.
    pub output_text_bytes: Option<u64>,
}

/// Per-response usage scanner. One is created lazily on the first response
/// read of a known-provider connection and lives until the request is reset.
pub struct RespParse {
    provider: Provider,

    // --- head phase ---
    head_done: bool,
    head_buf: Vec<u8>,
    /// Response status code once the head is parsed. Read by the caller to
    /// populate `RequestCompleted.status`.
    pub status: Option<u16>,
    mode: RespMode,
    chunked: bool,

    // --- body phase ---
    dechunk: Dechunker,
    line_buf: Vec<u8>,
    json_buf: Vec<u8>,

    // --- results ---
    measured_in: Option<u64>,
    measured_out: Option<u64>,
    /// Anthropic-only prompt-cache breakdown of the input, captured alongside
    /// `measured_in` (they ride the same `usage` object on `message_start`).
    /// Merged with the same latest-wins-if-Some semantics, so `message_delta`
    /// (output only) never clobbers them.
    measured_cache_read: Option<u64>,
    measured_cache_write: Option<u64>,
    /// Model name parsed from the response (`model` for Anthropic/OpenAI,
    /// `modelVersion` for Gemini). Captured during SSE from the first event that
    /// carries it (Anthropic `message_start`, the OpenAI/Gemini usage chunk);
    /// for non-streaming JSON it's read in `finalize`. First non-empty wins.
    measured_model: Option<String>,
    /// Tool-call names from the response. Anthropic/Gemini accumulate these
    /// during SSE; for non-streaming JSON they're computed in `finalize`. The
    /// count is the length; the names drive the per-tool-type calls graph.
    tool_call_names: Vec<String>,

    /// When set, SSE parsing also sums the generated-text bytes into
    /// `output_text_bytes` (see `Finalized::output_text_bytes`). Off by default
    /// so the shim's hot path never parses the hundreds of content-delta events;
    /// the proxy turns it on for the output-token estimate.
    count_output_text: bool,
    /// Running sum of generated-text bytes when `count_output_text` is on.
    output_text_bytes: u64,

    /// Set once we give up (unparseable/unknown head). Makes further `feed`
    /// calls cheap no-ops. `status` is preserved if it was already parsed.
    finished: bool,
}

impl RespParse {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            head_done: false,
            head_buf: Vec::new(),
            status: None,
            mode: RespMode::Unknown,
            chunked: false,
            dechunk: Dechunker::new(),
            line_buf: Vec::new(),
            json_buf: Vec::new(),
            measured_in: None,
            measured_out: None,
            measured_cache_read: None,
            measured_cache_write: None,
            measured_model: None,
            tool_call_names: Vec::new(),
            count_output_text: false,
            output_text_bytes: 0,
            finished: false,
        }
    }

    /// Enable summing the generated-text bytes during SSE parsing (for the
    /// output-token estimate on responses that carry no measured usage). Off by
    /// default; the proxy enables it, the shim leaves it off to keep its hot path
    /// from parsing every content-delta event.
    pub fn enable_output_text_count(&mut self) {
        self.count_output_text = true;
    }

    /// Scanner for an HTTP/2 response body. Unlike `new`, there is no HTTP/1
    /// head to parse (h2 status/content-type live in an HPACK HEADERS frame),
    /// so we start straight in the body phase: the caller sniffs JSON-vs-SSE
    /// from the first DATA bytes (see `looks_like_sse`) and passes `sse`. h2
    /// DATA payloads are already de-framed, so `chunked` stays false and
    /// completion is driven by the stream's END_STREAM flag (in state.rs), not
    /// `is_complete`. Status defaults to 200 until the HPACK `:status` is read.
    pub fn new_h2_body(provider: Provider, sse: bool) -> Self {
        let mut s = Self::new(provider);
        s.head_done = true;
        s.status = Some(200);
        s.mode = if sse { RespMode::Sse } else { RespMode::Json };
        s
    }

    /// Feed one decrypted response read chunk.
    pub fn feed(&mut self, buf: &[u8]) {
        if self.finished {
            return;
        }
        if self.head_done {
            self.feed_body(buf);
            return;
        }

        // Head phase: accumulate until the `\r\n\r\n` terminator, then parse.
        self.head_buf.extend_from_slice(buf);
        loop {
            let pos = match find_sub(&self.head_buf, b"\r\n\r\n") {
                Some(p) => p,
                None => {
                    if self.head_buf.len() > MAX_HEAD_BYTES {
                        self.give_up();
                    }
                    return;
                }
            };
            match parser::parse_response_head(&self.head_buf[..pos + 4]) {
                // 1xx interim responses (e.g. `100 Continue`) precede the real
                // head; drop and keep scanning the remainder.
                Some(h) if h.status < 200 => {
                    self.head_buf.drain(..pos + 4);
                    continue;
                }
                Some(h) => {
                    self.status = Some(h.status);
                    self.chunked = h.chunked;
                    self.mode = if h.is_event_stream {
                        RespMode::Sse
                    } else if h.is_json {
                        RespMode::Json
                    } else {
                        RespMode::Unknown
                    };
                    let remainder = self.head_buf.split_off(pos + 4);
                    self.head_buf = Vec::new();
                    self.head_done = true;
                    if self.mode == RespMode::Unknown {
                        // Keep the status we just learned, stop body parsing.
                        self.give_up();
                        return;
                    }
                    self.feed_body(&remainder);
                    return;
                }
                None => {
                    self.give_up();
                    return;
                }
            }
        }
    }

    /// Final per-response results consumed at request completion: measured
    /// token usage (or `None` per field) and the tool-call count. For SSE these
    /// were captured during `feed`; for non-streaming JSON the accumulated body
    /// is parsed here once.
    pub fn finalize(&self) -> Finalized {
        match self.mode {
            RespMode::Sse => Finalized {
                tokens_in: self.measured_in,
                tokens_out: self.measured_out,
                cache_read_tokens: self.measured_cache_read,
                cache_write_tokens: self.measured_cache_write,
                tool_calls: self.tool_call_names.len() as u64,
                tool_call_names: self.tool_call_names.clone(),
                model: self.measured_model.clone(),
                output_text_bytes: self.count_output_text.then_some(self.output_text_bytes),
            },
            RespMode::Json => match serde_json::from_slice::<Value>(&self.json_buf) {
                Ok(v) => {
                    let u = extract_usage(self.provider, &v);
                    let names = tool_call_names_json(self.provider, &v);
                    Finalized {
                        tokens_in: u.input,
                        tokens_out: u.output,
                        cache_read_tokens: u.cache_read,
                        cache_write_tokens: u.cache_write,
                        tool_calls: names.len() as u64,
                        tool_call_names: names,
                        model: extract_model(self.provider, &v),
                        // Non-streaming bodies are the measured api.anthropic.com
                        // path (never estimated); leave text-byte counting to SSE.
                        output_text_bytes: None,
                    }
                }
                Err(_) => Finalized::default(),
            },
            RespMode::Unknown => Finalized::default(),
        }
    }

    /// Token-usage-only accessor used by tests; production code reads
    /// `finalize` (which also yields the tool-call count).
    #[cfg(test)]
    pub fn measured(&self) -> (Option<u64>, Option<u64>) {
        let f = self.finalize();
        (f.tokens_in, f.tokens_out)
    }

    /// True once the response body is fully received, detected via the
    /// chunked-transfer terminal 0-size chunk. Lets the caller emit
    /// `RequestCompleted` the moment the response ends instead of deferring to
    /// the next request boundary / `SSL_free` / exit drain — which loses the
    /// LAST request of a run (its response arrives but the process exits before
    /// the boundary fires). Only chunked responses (the common LLM case: both
    /// non-streaming chunked JSON and streaming SSE end at the 0-chunk) report
    /// completion; Content-Length / unknown framing return false and stay on the
    /// deferred path, so there's no behavior change there.
    pub fn is_complete(&self) -> bool {
        self.head_done && self.chunked && self.dechunk.done
    }

    fn feed_body(&mut self, buf: &[u8]) {
        if self.finished {
            return;
        }
        if self.chunked {
            let decoded = self.dechunk.push(buf);
            self.consume(&decoded);
        } else {
            self.consume(buf);
        }
    }

    fn consume(&mut self, body: &[u8]) {
        match self.mode {
            RespMode::Sse => self.consume_sse(body),
            RespMode::Json => self.consume_json(body),
            RespMode::Unknown => {}
        }
    }

    fn consume_sse(&mut self, body: &[u8]) {
        self.line_buf.extend_from_slice(body);
        while let Some(pos) = self.line_buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.line_buf.drain(..=pos).collect();
            // Strip the trailing `\n` and an optional `\r`.
            let mut end = line.len() - 1;
            if end > 0 && line[end - 1] == b'\r' {
                end -= 1;
            }
            self.scan_sse_line(&line[..end]);
        }
        if self.line_buf.len() > MAX_LINE_BYTES {
            // Pathological un-terminated line; drop it. Usage events are
            // small and arrive on their own lines, so this never loses usage.
            self.line_buf.clear();
        }
    }

    fn scan_sse_line(&mut self, line: &[u8]) {
        let payload = match line.strip_prefix(b"data:") {
            Some(p) => trim_ascii_start(p),
            None => return,
        };
        // Cheap pre-filter: only `serde_json`-parse lines that could carry
        // usage, a tool call, or (until we've captured it) the served model.
        // "usage" is a substring of "usageMetadata", so it covers all three
        // providers' usage; the tool markers are provider-specific. The model
        // term lets us read the model from an event that carries it but no usage
        // — notably Anthropic's `message_start`, whose `usage` is absent on the
        // consumer web API — so response-side model attribution works even when
        // the request body wasn't captured (e.g. read-only h2 connections). It's
        // gated on `measured_model.is_none()` so it stops firing once the model
        // is known: OpenAI streams a `model` on every chunk, and without the gate
        // that would defeat the pre-filter for the whole stream. (OpenAI
        // streaming tool-call deltas aren't counted yet - non-streaming is.)
        let has_usage = contains(payload, b"usage");
        let has_tool = contains(payload, b"tool_use") || contains(payload, b"functionCall");
        let want_model = self.measured_model.is_none() && contains(payload, b"\"model\"");
        // Only when output-text counting is on: `content_block_delta` carries the
        // streamed output text (Anthropic). This is what lets the estimate see the
        // generated text rather than the full SSE envelope.
        let want_text = self.count_output_text && contains(payload, b"content_block_delta");
        if !has_usage && !has_tool && !want_model && !want_text {
            return;
        }
        let v = match serde_json::from_slice::<Value>(payload) {
            Ok(v) => v,
            Err(_) => return,
        };
        if want_text {
            self.output_text_bytes += sse_output_delta_len(self.provider, &v);
        }
        // Capture the model from whichever parsed event carries it first
        // (Anthropic `message_start`, the OpenAI/Gemini usage chunk). It's
        // constant across a response, so first non-empty wins.
        if self.measured_model.is_none() {
            self.measured_model = extract_model(self.provider, &v);
        }
        if has_usage {
            let u = extract_usage(self.provider, &v);
            if u.input.is_some() {
                self.measured_in = u.input;
            }
            if u.output.is_some() {
                self.measured_out = u.output; // latest value wins (cumulative)
            }
            // Cache buckets ride `message_start` only; `message_delta` (output
            // alone) yields None here and leaves the start values intact.
            if u.cache_read.is_some() {
                self.measured_cache_read = u.cache_read;
            }
            if u.cache_write.is_some() {
                self.measured_cache_write = u.cache_write;
            }
        }
        if has_tool {
            self.tool_call_names
                .extend(tool_use_names_sse_event(self.provider, &v));
        }
    }

    fn consume_json(&mut self, body: &[u8]) {
        if self.json_buf.len() >= JSON_CAP {
            return;
        }
        let room = JSON_CAP - self.json_buf.len();
        if body.len() <= room {
            self.json_buf.extend_from_slice(body);
        } else {
            self.json_buf.extend_from_slice(&body[..room]);
        }
    }

    fn give_up(&mut self) {
        self.finished = true;
        self.head_buf = Vec::new();
        self.line_buf = Vec::new();
        self.json_buf = Vec::new();
    }
}

/// Token usage extracted from one parsed provider JSON value. `None` per field
/// so callers can merge with latest-wins semantics across streamed events.
/// `input` is the billable input total; `cache_read`/`cache_write` are the two
/// Anthropic prompt-cache buckets, already folded into `input`, kept separate
/// so the metrics sink can plot the fresh-vs-cached breakdown.
#[derive(Debug, Default, Clone, Copy)]
struct Usage {
    input: Option<u64>,
    output: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
}

/// Extract token usage from a parsed provider JSON value. Returns `None` per
/// field when the value doesn't carry it (so callers can merge with latest-wins
/// semantics across streamed events).
fn extract_usage(provider: Provider, v: &Value) -> Usage {
    match provider {
        Provider::Anthropic => {
            // Non-streaming + `message_delta`: top-level `.usage`.
            // `message_start`: nested `.message.usage`.
            let u = v
                .get("usage")
                .or_else(|| v.get("message").and_then(|m| m.get("usage")));
            match u {
                Some(u) => {
                    // Anthropic reports input in three buckets — fresh
                    // `input_tokens`, `cache_read_input_tokens`, and
                    // `cache_creation_input_tokens` — and bills all three as
                    // input. Prompt-cache-heavy clients (Claude Code resends a
                    // large cached system prompt + tools every turn) land almost
                    // the entire prompt in `cache_read_input_tokens`, so reading
                    // only `input_tokens` undercounts a turn by orders of
                    // magnitude (a real turn shows `input_tokens:2` while tens of
                    // thousands were cache reads). Sum the buckets into `input` so
                    // the reported input reflects the true prompt size, and return
                    // the two cache buckets on their own so the metrics sink can
                    // split fresh vs cache-read vs cache-write. The cache fields
                    // ride `message_start` only; `message_delta` carries output
                    // alone, so they yield `None` there and the merge leaves the
                    // start values intact. OpenAI/Gemini already report a
                    // cache-inclusive prompt total, so only Anthropic needs this.
                    let base = u.get("input_tokens").and_then(Value::as_u64);
                    let cache_read = u.get("cache_read_input_tokens").and_then(Value::as_u64);
                    let cache_creation =
                        u.get("cache_creation_input_tokens").and_then(Value::as_u64);
                    let input = match (base, cache_read, cache_creation) {
                        (None, None, None) => None,
                        _ => Some(
                            base.unwrap_or(0)
                                + cache_read.unwrap_or(0)
                                + cache_creation.unwrap_or(0),
                        ),
                    };
                    Usage {
                        input,
                        output: u.get("output_tokens").and_then(Value::as_u64),
                        cache_read,
                        cache_write: cache_creation,
                    }
                }
                None => Usage::default(),
            }
        }
        Provider::OpenAi => match v.get("usage") {
            // Streaming chunks carry `"usage": null` until the final one;
            // require an object so nulls are skipped.
            Some(u) if u.is_object() => Usage {
                input: u.get("prompt_tokens").and_then(Value::as_u64),
                output: u.get("completion_tokens").and_then(Value::as_u64),
                ..Usage::default()
            },
            _ => Usage::default(),
        },
        Provider::Gemini => match v.get("usageMetadata") {
            Some(u) => Usage {
                input: u.get("promptTokenCount").and_then(Value::as_u64),
                output: u.get("candidatesTokenCount").and_then(Value::as_u64),
                ..Usage::default()
            },
            None => Usage::default(),
        },
    }
}

/// Extract the served model name from a parsed provider response value. Used
/// for per-model usage attribution (the "coset" the usage aggregator groups
/// on, alongside the provider). Handles all three providers from the SAME place
/// the usage is read:
///   * Anthropic — top-level `model` (non-streaming / `message_delta`) or
///     `message.model` (the SSE `message_start` event).
///   * OpenAI — top-level `model` (present on every streaming chunk and the
///     non-streaming body).
///   * Gemini — top-level `modelVersion` (the resolved model id; Gemini doesn't
///     use a `model` field in the response).
/// `None` when the value carries no model (so the caller keeps looking / falls
/// back to the request).
fn extract_model(provider: Provider, v: &Value) -> Option<String> {
    let raw = match provider {
        Provider::Anthropic => v
            .get("model")
            .or_else(|| v.get("message").and_then(|m| m.get("model"))),
        Provider::OpenAi => v.get("model"),
        Provider::Gemini => v.get("modelVersion"),
    };
    let m = raw.and_then(Value::as_str)?.trim();
    (!m.is_empty()).then(|| m.to_string())
}

/// The model the request *asked for*, used as a fallback when the response
/// didn't echo one (an error response, or an OpenAI stream without usage).
/// Anthropic and OpenAI carry it as the top-level `"model"` string in the
/// request body; Gemini puts it in the URL path
/// (`/v1beta/models/{model}:generateContent`), never the body, so we read it
/// from `path` there. `None` when it can't be determined.
pub fn model_from_request(provider: Provider, body: &[u8], path: Option<&str>) -> Option<String> {
    match provider {
        Provider::Anthropic | Provider::OpenAi => {
            let v: Value = serde_json::from_slice(body).ok()?;
            let m = v.get("model").and_then(Value::as_str)?.trim();
            (!m.is_empty()).then(|| m.to_string())
        }
        Provider::Gemini => model_from_gemini_path(path?),
    }
}

/// Extract the Gemini model id from a request path, e.g.
/// `/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse` ->
/// `gemini-1.5-pro`. The model is the last path segment up to its `:method`
/// suffix (`tunedModels/{id}:…` resolves to `{id}` the same way). `None` when
/// there's no `:` segment — i.e. it isn't a generate call — or the model is
/// empty.
fn model_from_gemini_path(path: &str) -> Option<String> {
    let path = path.split('?').next().unwrap_or(path);
    let last = path.rsplit('/').next()?;
    let (model, _method) = last.split_once(':')?;
    let model = model.trim();
    (!model.is_empty()).then(|| model.to_string())
}

/// Byte length of the GENERATED output text in one SSE `content_block_delta`
/// event, for the output-token estimate. Anthropic streams three billable output
/// kinds: assistant `text_delta` (`text`), extended-thinking `thinking_delta`
/// (`thinking`), and tool-argument `input_json_delta` (`partial_json`) — all
/// counted. `signature_delta` (a thinking-block crypto signature, not model
/// output) and everything else count 0. Only the consumer wire (Anthropic) is
/// estimated, so other providers return 0.
fn sse_output_delta_len(provider: Provider, v: &Value) -> u64 {
    match provider {
        Provider::Anthropic => {
            if v.get("type").and_then(Value::as_str) != Some("content_block_delta") {
                return 0;
            }
            let Some(delta) = v.get("delta") else { return 0 };
            let field = match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => "text",
                Some("thinking_delta") => "thinking",
                Some("input_json_delta") => "partial_json",
                _ => return 0,
            };
            delta
                .get(field)
                .and_then(Value::as_str)
                .map(|t| t.len() as u64)
                .unwrap_or(0)
        }
        Provider::OpenAi | Provider::Gemini => 0,
    }
}

/// Tool-call names in a single SSE event (one `data:` JSON object). Anthropic:
/// a `content_block_start` whose block is a `tool_use`. Gemini: `functionCall`
/// parts in the chunk. OpenAI streaming deltas aren't handled here yet
/// (non-streaming OpenAI is handled by `tool_call_names_json`).
fn tool_use_names_sse_event(provider: Provider, v: &Value) -> Vec<String> {
    match provider {
        Provider::Anthropic => {
            let is_start =
                v.get("type").and_then(Value::as_str) == Some("content_block_start");
            let block = v.get("content_block");
            let is_tool = block
                .and_then(|b| b.get("type"))
                .and_then(Value::as_str)
                == Some("tool_use");
            if is_start && is_tool {
                vec![tool_name_or_unknown(block.and_then(|b| b.get("name")))]
            } else {
                Vec::new()
            }
        }
        Provider::Gemini => gemini_function_names(v),
        Provider::OpenAi => Vec::new(),
    }
}

/// Tool-call names in a complete (non-streaming) response body.
fn tool_call_names_json(provider: Provider, v: &Value) -> Vec<String> {
    match provider {
        Provider::Anthropic => v
            .get("content")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                    .map(|b| tool_name_or_unknown(b.get("name")))
                    .collect()
            })
            .unwrap_or_default(),
        Provider::OpenAi => v
            .get("choices")
            .and_then(Value::as_array)
            .map(|choices| {
                choices
                    .iter()
                    .flat_map(|c| {
                        c.get("message")
                            .and_then(|m| m.get("tool_calls"))
                            .and_then(Value::as_array)
                            .map(|tcs| {
                                tcs.iter()
                                    .map(|tc| {
                                        tool_name_or_unknown(
                                            tc.get("function").and_then(|f| f.get("name")),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Provider::Gemini => gemini_function_names(v),
    }
}

/// `functionCall` names across all candidates of a Gemini response/stream chunk.
fn gemini_function_names(v: &Value) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(cands) = v.get("candidates").and_then(Value::as_array) {
        for c in cands {
            if let Some(parts) = c
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array)
            {
                for p in parts {
                    if let Some(fc) = p.get("functionCall") {
                        names.push(tool_name_or_unknown(fc.get("name")));
                    }
                }
            }
        }
    }
    names
}

/// A tool name as a String, or "unknown" when the field is missing/non-string.
fn tool_name_or_unknown(name: Option<&Value>) -> String {
    name.and_then(Value::as_str).unwrap_or("unknown").to_string()
}

/// Max request body buffered to look for freshest-turn tool errors. JSON must
/// be parsed from the start, so a body exceeding this can't be parsed and
/// yields 0 (an under-count) rather than growing memory without bound.
pub const REQ_BODY_CAP: usize = 2 * 1024 * 1024;

/// Names of the tools whose results failed in the request's freshest turn (its
/// last message), for per-tool error attribution. The resent history makes
/// this a single-body parse, not cross-request correlation: an
/// Anthropic `tool_result` carries only `tool_use_id`, but the `tool_use` block
/// that named that id is an earlier assistant turn in the SAME request body, so
/// we build the id→name map from all turns and resolve each freshest-turn error
/// to its tool name (`"unknown"` if the id is absent). Counting only the last
/// message counts each error once (the resent history is ignored). Only
/// Anthropic marks tool errors structurally (`tool_result.is_error`);
/// OpenAI/Gemini don't, so they yield an empty vec.
///
/// The aggregate error count is just this vec's length (see the caller).
pub fn tool_error_names_in_request(provider: Provider, body: &[u8]) -> Vec<String> {
    if provider != Provider::Anthropic {
        return Vec::new();
    }
    let v: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let messages = match v.get("messages").and_then(Value::as_array) {
        Some(m) => m,
        None => return Vec::new(),
    };
    // tool_use_id -> tool name, gathered from every assistant turn's tool_use
    // blocks (the freshest turn's tool_results reference ids defined earlier).
    let mut id_to_name: HashMap<&str, &str> = HashMap::new();
    for m in messages {
        if let Some(content) = m.get("content").and_then(Value::as_array) {
            for b in content {
                if b.get("type").and_then(Value::as_str) == Some("tool_use") {
                    if let Some(id) = b.get("id").and_then(Value::as_str) {
                        let name = b.get("name").and_then(Value::as_str).unwrap_or("unknown");
                        id_to_name.insert(id, name);
                    }
                }
            }
        }
    }
    let last = match messages.last() {
        Some(m) => m,
        None => return Vec::new(),
    };
    let content = match last.get("content").and_then(Value::as_array) {
        Some(c) => c,
        None => return Vec::new(),
    };
    content
        .iter()
        .filter(|b| {
            b.get("type").and_then(Value::as_str) == Some("tool_result")
                && b.get("is_error").and_then(Value::as_bool) == Some(true)
        })
        .map(|b| {
            b.get("tool_use_id")
                .and_then(Value::as_str)
                .and_then(|id| id_to_name.get(id).copied())
                .unwrap_or("unknown")
                .to_string()
        })
        .collect()
}

/// A request's conversation identity, decomposed so the session registry
/// (`crate::session`) can match a request to the chat it continues even as the
/// transcript evolves. `system_hash` is matched for equality (a gate);
/// `turn_hashes` is matched by longest-common-prefix / tail-overlap so append,
/// retry, and sliding-window trimming all stay one chat. See `crate::session`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// Hash of the system prompt (top-level for Anthropic/Gemini; the leading
    /// `role:"system"` messages for OpenAI). 0 when absent.
    pub system_hash: u64,
    /// One hash per conversation turn (message / content entry), in order,
    /// EXCLUDING the system prompt so a shared system doesn't merge distinct
    /// chats and front-trimming aligns cleanly.
    pub turn_hashes: Vec<u64>,
}

/// Stable hash of one JSON value. `serde_json`'s default `Map` is a `BTreeMap`,
/// so object keys serialize in sorted order — canonical regardless of the
/// client's field ordering. `DefaultHasher` (fixed-key SipHash) needs no extra
/// dependency and is deterministic within a task process; it is not crypto.
fn hash_value(v: &Value) -> u64 {
    let mut h = DefaultHasher::new();
    match serde_json::to_vec(v) {
        Ok(bytes) => bytes.hash(&mut h),
        Err(_) => 0u8.hash(&mut h),
    }
    h.finish()
}

/// Fingerprint the request body for session matching. `None` when the body
/// isn't parseable JSON (e.g. truncated past `REQ_BODY_CAP`) or
/// carries no conversation turns — the caller then leaves the series keyed by
/// provider alone.
pub fn conversation_fingerprint(provider: Provider, body: &[u8]) -> Option<Fingerprint> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let (system_hash, turn_hashes): (u64, Vec<u64>) = match provider {
        Provider::Anthropic => {
            let msgs = v.get("messages").and_then(Value::as_array)?;
            let sh = v.get("system").map(hash_value).unwrap_or(0);
            (sh, msgs.iter().map(hash_value).collect())
        }
        Provider::OpenAi => {
            // OpenAI has no separate system field: fold the leading
            // `role:"system"` message(s) into system_hash, leave the rest as
            // turns — so a shared system prompt gates rather than merges.
            let msgs = v.get("messages").and_then(Value::as_array)?;
            let split = msgs
                .iter()
                .take_while(|m| m.get("role").and_then(Value::as_str) == Some("system"))
                .count();
            let sh = if split > 0 {
                hash_value(&Value::Array(msgs[..split].to_vec()))
            } else {
                0
            };
            (sh, msgs[split..].iter().map(hash_value).collect())
        }
        Provider::Gemini => {
            let contents = v.get("contents").and_then(Value::as_array)?;
            let sh = v.get("systemInstruction").map(hash_value).unwrap_or(0);
            (sh, contents.iter().map(hash_value).collect())
        }
    };
    if turn_hashes.is_empty() {
        return None;
    }
    Some(Fingerprint {
        system_hash,
        turn_hashes,
    })
}

/// Incremental HTTP/1.1 chunked-transfer decoder. Feed raw bytes via `push`;
/// it returns the de-chunked body bytes seen so far, carrying partial state
/// (size line, remaining chunk bytes, trailing CRLF) across calls.
struct Dechunker {
    in_size_line: bool,
    size_line: Vec<u8>,
    remaining: usize,
    skip_crlf: u8,
    done: bool,
}

impl Dechunker {
    fn new() -> Self {
        Self {
            in_size_line: true,
            size_line: Vec::new(),
            remaining: 0,
            skip_crlf: 0,
            done: false,
        }
    }

    fn push(&mut self, buf: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < buf.len() && !self.done {
            if self.skip_crlf > 0 {
                self.skip_crlf -= 1;
                i += 1;
                continue;
            }
            if self.in_size_line {
                let b = buf[i];
                i += 1;
                if b == b'\n' {
                    let size = parse_hex_prefix(&self.size_line);
                    self.size_line.clear();
                    self.in_size_line = false;
                    self.remaining = size;
                    if size == 0 {
                        self.done = true;
                    }
                } else if b != b'\r' && self.size_line.len() < 32 {
                    self.size_line.push(b);
                }
                continue;
            }
            if self.remaining > 0 {
                let take = (buf.len() - i).min(self.remaining);
                out.extend_from_slice(&buf[i..i + take]);
                i += take;
                self.remaining -= take;
                if self.remaining == 0 {
                    self.skip_crlf = 2; // trailing \r\n after the chunk body
                    self.in_size_line = true;
                }
            } else {
                self.in_size_line = true;
            }
        }
        out
    }
}

/// Parse leading hex digits as a chunk size, stopping at the first non-hex
/// byte (e.g. a `;` chunk extension).
fn parse_hex_prefix(s: &[u8]) -> usize {
    let mut n: usize = 0;
    for &b in s {
        let d = match b {
            b'0'..=b'9' => (b - b'0') as usize,
            b'a'..=b'f' => (b - b'a' + 10) as usize,
            b'A'..=b'F' => (b - b'A' + 10) as usize,
            _ => break,
        };
        n = n.saturating_mul(16).saturating_add(d);
    }
    n
}

fn find_sub(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    find_sub(haystack, needle).is_some()
}

fn trim_ascii_start(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    s
}

/// Sniff an HTTP/2 response body (raw DATA, no HTTP head) as SSE vs JSON from
/// its leading bytes: SSE streaming opens with an `event:`/`data:` line or a `:`
/// comment; JSON with `{`/`[`. Defaults to JSON (false).
pub fn looks_like_sse(bytes: &[u8]) -> bool {
    let s = trim_ascii_start(bytes);
    s.starts_with(b"event:") || s.starts_with(b"data:") || s.starts_with(b":")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2_body_sse_anthropic_usage() {
        let mut r = RespParse::new_h2_body(Provider::Anthropic, true);
        r.feed(b"event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":736,\"output_tokens\":1}}}\r\n\r\n");
        r.feed(b"event: message_delta\r\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":21}}\r\n\r\n");
        let f = r.finalize();
        assert_eq!(f.tokens_in, Some(736));
        assert_eq!(f.tokens_out, Some(21));
        assert_eq!(f.model.as_deref(), Some("claude-haiku-4-5"));
    }

    #[test]
    fn h2_body_json_anthropic_usage() {
        let mut r = RespParse::new_h2_body(Provider::Anthropic, false);
        r.feed(b"{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}");
        let f = r.finalize();
        assert_eq!(f.tokens_in, Some(10));
        assert_eq!(f.tokens_out, Some(5));
    }

    #[test]
    fn sse_output_text_bytes_sums_generated_deltas_when_enabled() {
        // With counting on, output_text_bytes sums only the GENERATED text across
        // text_delta + thinking_delta + input_json_delta — ignoring signature_delta
        // and all the SSE `event:`/`data:` framing. This is what keeps the output
        // estimate off the (much larger) whole-envelope byte count.
        let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello there\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"IGNOREME\"}}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":21}}\n\n";
        let mut on = RespParse::new_h2_body(Provider::Anthropic, true);
        on.enable_output_text_count();
        on.feed(sse);
        // "hmm"(3) + "Hello there"(11) + "{\"a\":1}"(7) = 21; signature ignored.
        assert_eq!(on.finalize().output_text_bytes, Some(21));

        // Default (shim): counting off -> None, and content deltas aren't parsed.
        let mut off = RespParse::new_h2_body(Provider::Anthropic, true);
        off.feed(sse);
        assert_eq!(off.finalize().output_text_bytes, None);
    }

    #[test]
    fn looks_like_sse_detects_shape() {
        assert!(looks_like_sse(b"event: message_start\n"));
        assert!(looks_like_sse(b"  data: {}\n"));
        assert!(!looks_like_sse(b"{\"model\":\"x\"}"));
        assert!(!looks_like_sse(b"  [1,2]"));
    }

    fn feed_all(resp: &[u8], provider: Provider) -> RespParse {
        let mut p = RespParse::new(provider);
        p.feed(resp);
        p
    }

    fn feed_chunks(chunks: &[&[u8]], provider: Provider) -> RespParse {
        let mut p = RespParse::new(provider);
        for c in chunks {
            p.feed(c);
        }
        p
    }

    /// Encode `body` with HTTP/1.1 chunked transfer framing, deliberately
    /// splitting into small chunks to exercise the de-chunker's boundaries.
    fn chunk_encode(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for piece in body.chunks(37) {
            out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
            out.extend_from_slice(piece);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }

    const ANTHROPIC_JSON_BODY: &[u8] = br#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":14,"output_tokens":10}}"#;

    const ANTHROPIC_SSE_BODY: &[u8] = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":14,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello there\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":495}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

    #[test]
    fn provider_mapping() {
        assert_eq!(
            provider_for_host("api.anthropic.com"),
            Some(Provider::Anthropic)
        );
        assert_eq!(provider_for_host("api.openai.com"), Some(Provider::OpenAi));
        assert_eq!(
            provider_for_host("generativelanguage.googleapis.com"),
            Some(Provider::Gemini)
        );
        assert_eq!(provider_for_host("example.com"), None);
    }

    #[test]
    fn anthropic_non_streaming_json() {
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_JSON_BODY);
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.status, Some(200));
        assert_eq!(p.measured(), (Some(14), Some(10)));
    }

    #[test]
    fn is_complete_tracks_chunked_terminal_chunk() {
        // Chunked JSON: incomplete until the terminal 0-size chunk arrives; once
        // it does, the body is fully de-chunked + usage still parses. This drives
        // the early RequestCompleted emit (so the last request of a run isn't lost
        // to the exit drain).
        let mut head =
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
                .to_vec();
        let body = chunk_encode(ANTHROPIC_JSON_BODY); // ends with "0\r\n\r\n"
        head.extend_from_slice(&body[..body.len() - 5]); // all but the terminal chunk
        let mut p = RespParse::new(Provider::Anthropic);
        p.feed(&head);
        assert!(!p.is_complete(), "missing terminal chunk -> not complete");
        p.feed(b"0\r\n\r\n");
        assert!(p.is_complete(), "terminal 0-chunk -> complete");
        assert_eq!(p.measured(), (Some(14), Some(10)));

        // Non-chunked (Content-Length framing) never reports completion, so it
        // stays on the deferred emit path — no behavior change there.
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_JSON_BODY);
        let np = feed_all(&resp, Provider::Anthropic);
        assert!(!np.is_complete(), "non-chunked response reports incomplete");
    }

    #[test]
    fn anthropic_sse_input_from_start_output_from_delta() {
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_SSE_BODY);
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.status, Some(200));
        // input from message_start, final output from message_delta (not the
        // initial output_tokens:1).
        assert_eq!(p.measured(), (Some(14), Some(495)));
    }

    #[test]
    fn anthropic_json_sums_cache_tokens_into_input() {
        // Non-streaming body carrying the three input buckets: the measured
        // input is their sum (fresh 5 + cache_read 40000 + cache_creation 120).
        let body = br#"{"model":"claude-sonnet-4-5","usage":{"input_tokens":5,"cache_creation_input_tokens":120,"cache_read_input_tokens":40000,"output_tokens":37}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.measured(), (Some(40125), Some(37)));
    }

    #[test]
    fn anthropic_sse_cache_read_dominant_input() {
        // The real Claude Code streaming shape: message_start reports a tiny
        // fresh `input_tokens` with the prompt bulk in `cache_read_input_tokens`;
        // message_delta carries only the cumulative output. The measured input
        // must be the full prompt (2 + 45000), not the fresh 2 — the prod bug
        // where a real turn showed `tokens_in=2`.
        let body = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":2,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":45000,\"output_tokens\":1}}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":13}}\n\n";
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.measured(), (Some(45002), Some(13)));
        assert_eq!(p.finalize().model.as_deref(), Some("claude-sonnet-5"));
    }

    #[test]
    fn anthropic_sse_separates_cache_read_and_write_keeping_summed_input() {
        // message_start carries fresh input + both cache buckets; message_delta
        // carries only cumulative output. The summed input (fresh 2 + read 45000 +
        // write 300) must be unchanged, AND the two cache buckets must be surfaced
        // separately (read=45000, write=300) — while message_delta's output-only
        // usage does NOT clobber the cache values captured at start.
        let body = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":2,\"cache_creation_input_tokens\":300,\"cache_read_input_tokens\":45000,\"output_tokens\":1}}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":13}}\n\n";
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let f = feed_all(&resp, Provider::Anthropic).finalize();
        assert_eq!(f.tokens_in, Some(45302), "summed input unchanged");
        assert_eq!(f.tokens_out, Some(13));
        assert_eq!(f.cache_read_tokens, Some(45000), "cache-read surfaced");
        assert_eq!(f.cache_write_tokens, Some(300), "cache-write surfaced");
    }

    #[test]
    fn anthropic_json_separates_cache_read_and_write() {
        // Non-streaming body: the summed input holds, and the two cache buckets
        // are surfaced individually.
        let body = br#"{"model":"claude-sonnet-4-5","usage":{"input_tokens":5,"cache_creation_input_tokens":120,"cache_read_input_tokens":40000,"output_tokens":37}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let f = feed_all(&resp, Provider::Anthropic).finalize();
        assert_eq!(f.tokens_in, Some(40125), "summed input unchanged");
        assert_eq!(f.cache_read_tokens, Some(40000));
        assert_eq!(f.cache_write_tokens, Some(120));
    }

    #[test]
    fn non_anthropic_has_no_cache_breakdown() {
        // OpenAI + Gemini report a cache-inclusive prompt total with no breakdown;
        // their cache fields stay None (never fabricated).
        let openai = br#"{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":7,"total_tokens":27}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(openai);
        let f = feed_all(&resp, Provider::OpenAi).finalize();
        assert_eq!(f.tokens_in, Some(20));
        assert!(f.cache_read_tokens.is_none() && f.cache_write_tokens.is_none());

        let gemini = br#"{"candidates":[],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4,"totalTokenCount":15},"modelVersion":"gemini-1.5-pro"}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(gemini);
        let f = feed_all(&resp, Provider::Gemini).finalize();
        assert_eq!(f.tokens_in, Some(11));
        assert!(f.cache_read_tokens.is_none() && f.cache_write_tokens.is_none());
    }

    #[test]
    fn anthropic_count_tokens_top_level_stays_unmeasured() {
        // The `/v1/messages/count_tokens` pre-flight returns `input_tokens` at the
        // top level with NO `usage` object. It is advisory (the following
        // /v1/messages bills the same tokens), so it must NOT be metered —
        // otherwise the prompt would be double-counted.
        let body = br#"{"input_tokens":45231}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.measured(), (None, None));
    }

    #[test]
    fn anthropic_sse_chunked_transfer() {
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        resp.extend_from_slice(&chunk_encode(ANTHROPIC_SSE_BODY));
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.measured(), (Some(14), Some(495)));
    }

    #[test]
    fn anthropic_sse_split_across_many_reads() {
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_SSE_BODY);
        // Feed one byte at a time — the meanest possible read boundaries.
        let mut p = RespParse::new(Provider::Anthropic);
        for b in &resp {
            p.feed(std::slice::from_ref(b));
        }
        assert_eq!(p.measured(), (Some(14), Some(495)));
    }

    #[test]
    fn openai_non_streaming_json() {
        let body = br#"{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":7,"total_tokens":27}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::OpenAi);
        assert_eq!(p.measured(), (Some(20), Some(7)));
    }

    #[test]
    fn openai_streaming_include_usage_final_chunk() {
        let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n\
data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":7,\"total_tokens\":27}}\n\n\
data: [DONE]\n\n";
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::OpenAi);
        // The `usage: null` chunk must not clobber the real final usage.
        assert_eq!(p.measured(), (Some(20), Some(7)));
    }

    #[test]
    fn openai_streaming_without_usage_falls_back() {
        let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n\
data: [DONE]\n\n";
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::OpenAi);
        assert_eq!(p.measured(), (None, None));
    }

    #[test]
    fn gemini_non_streaming_usage_metadata() {
        let body = br#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4,"totalTokenCount":15}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let p = feed_all(&resp, Provider::Gemini);
        assert_eq!(p.measured(), (Some(11), Some(4)));
    }

    // --- tool-call counting ---------------------------------------------

    #[test]
    fn anthropic_sse_counts_tool_use_blocks() {
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":50,\"output_tokens\":1}}}\n\n");
        resp.extend_from_slice(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
        resp.extend_from_slice(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n");
        resp.extend_from_slice(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_2\",\"name\":\"get_time\",\"input\":{}}}\n\n");
        resp.extend_from_slice(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":40}}\n\n");
        let f = feed_all(&resp, Provider::Anthropic).finalize();
        assert_eq!(f.tool_calls, 2, "two tool_use blocks; the text block excluded");
        assert_eq!(f.tool_call_names, vec!["get_weather", "get_time"]);
        assert_eq!((f.tokens_in, f.tokens_out), (Some(50), Some(40)));
    }

    #[test]
    fn anthropic_json_counts_tool_use_blocks() {
        let body = br#"{"type":"message","content":[{"type":"text","text":"let me check"},{"type":"tool_use","id":"t1","name":"get_weather","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":50,"output_tokens":30}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let f = feed_all(&resp, Provider::Anthropic).finalize();
        assert_eq!(f.tool_calls, 1);
        assert_eq!((f.tokens_in, f.tokens_out), (Some(50), Some(30)));
    }

    #[test]
    fn openai_json_counts_tool_calls() {
        let body = br#"{"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"a","arguments":"{}"}},{"id":"c2","type":"function","function":{"name":"b","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":40,"completion_tokens":20,"total_tokens":60}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let f = feed_all(&resp, Provider::OpenAi).finalize();
        assert_eq!(f.tool_calls, 2);
        assert_eq!(f.tool_call_names, vec!["a", "b"]);
        assert_eq!((f.tokens_in, f.tokens_out), (Some(40), Some(20)));
    }

    #[test]
    fn gemini_json_counts_function_calls() {
        let body = br#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{}}}],"role":"model"}}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":8,"totalTokenCount":23}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        let f = feed_all(&resp, Provider::Gemini).finalize();
        assert_eq!(f.tool_calls, 1);
        assert_eq!(f.tool_call_names, vec!["get_weather"]);
    }

    #[test]
    fn no_tool_calls_when_absent() {
        // The plain non-streaming Anthropic fixture has no tool_use blocks.
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_JSON_BODY);
        assert_eq!(feed_all(&resp, Provider::Anthropic).finalize().tool_calls, 0);
    }

    #[test]
    fn tool_use_substring_in_text_is_not_counted() {
        // A text delta that merely mentions "tool_use" must not be miscounted -
        // the structural check (content_block_start + block type) guards it.
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        resp.extend_from_slice(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"the tool_use block is great\"}}\n\n");
        resp.extend_from_slice(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":5}}\n\n");
        assert_eq!(feed_all(&resp, Provider::Anthropic).finalize().tool_calls, 0);
    }

    // --- freshest-turn tool errors -> tool names (request side) ---------

    #[test]
    fn anthropic_request_maps_freshest_turn_errors_to_tool_names() {
        // An earlier turn's tool_result is clean; the LAST message has two
        // tool_results, one is_error referencing toolu_b (named get_time in a
        // prior assistant turn of the SAME body) -> ["get_time"]. The clean
        // result (toolu_c) is excluded; the resent history isn't double-counted.
        let body = br#"{"model":"claude","messages":[
            {"role":"user","content":"go"},
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_a","name":"get_weather","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","content":"ok"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_b","name":"get_time","input":{}},{"type":"tool_use","id":"toolu_c","name":"get_weather","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_b","content":"boom","is_error":true},{"type":"tool_result","tool_use_id":"toolu_c","content":"ok"}]}
        ]}"#;
        assert_eq!(
            tool_error_names_in_request(Provider::Anthropic, body),
            vec!["get_time"]
        );
    }

    #[test]
    fn tool_error_names_unknown_when_id_unmapped() {
        // An is_error result referencing an id with no prior tool_use -> "unknown".
        let body = br#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"ghost","content":"x","is_error":true}]}]}"#;
        assert_eq!(
            tool_error_names_in_request(Provider::Anthropic, body),
            vec!["unknown"]
        );
    }

    #[test]
    fn tool_error_names_empty_for_non_anthropic_garbage_and_clean() {
        let err = br#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","is_error":true}]}]}"#;
        assert!(tool_error_names_in_request(Provider::OpenAi, err).is_empty());
        assert!(tool_error_names_in_request(Provider::Gemini, err).is_empty());
        assert!(tool_error_names_in_request(Provider::Anthropic, b"not json").is_empty());
        let clean = br#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}"#;
        assert!(tool_error_names_in_request(Provider::Anthropic, clean).is_empty());
    }

    // --- conversation fingerprint --------------------------------------
    // (The matching that turns fingerprints into stable chat ids is tested in
    // `crate::session`; here we only verify the fingerprint extraction.)

    #[test]
    fn fingerprint_turns_exclude_system_and_count_messages() {
        let body = br#"{"system":"be terse","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#;
        let fp = conversation_fingerprint(Provider::Anthropic, body).unwrap();
        assert_eq!(fp.turn_hashes.len(), 2, "one hash per message, system excluded");
        assert_ne!(fp.system_hash, 0, "system present -> nonzero gate");
    }

    #[test]
    fn fingerprint_shared_opening_is_a_prefix() {
        // Turn 2 = turn 1 + more, same system: turn 1's hashes prefix turn 2's,
        // and the system gate is identical -> the registry matches them.
        let t1 = br#"{"system":"s","messages":[{"role":"user","content":"hi"}]}"#;
        let t2 = br#"{"system":"s","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},{"role":"user","content":"more"}]}"#;
        let f1 = conversation_fingerprint(Provider::Anthropic, t1).unwrap();
        let f2 = conversation_fingerprint(Provider::Anthropic, t2).unwrap();
        assert_eq!(f1.system_hash, f2.system_hash);
        assert_eq!(f2.turn_hashes.len(), 3);
        assert_eq!(f1.turn_hashes[0], f2.turn_hashes[0], "shared opening turn");
    }

    #[test]
    fn fingerprint_system_change_moves_gate_not_turns() {
        let a = br#"{"system":"s1","messages":[{"role":"user","content":"hi"}]}"#;
        let b = br#"{"system":"s2","messages":[{"role":"user","content":"hi"}]}"#;
        let fa = conversation_fingerprint(Provider::Anthropic, a).unwrap();
        let fb = conversation_fingerprint(Provider::Anthropic, b).unwrap();
        assert_ne!(fa.system_hash, fb.system_hash);
        assert_eq!(fa.turn_hashes, fb.turn_hashes, "same turns, different gate");
    }

    #[test]
    fn fingerprint_openai_folds_leading_system_into_gate() {
        let body = br#"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}"#;
        let fp = conversation_fingerprint(Provider::OpenAi, body).unwrap();
        assert_ne!(fp.system_hash, 0, "leading system message -> gate");
        assert_eq!(fp.turn_hashes.len(), 2, "system message excluded from turns");
    }

    #[test]
    fn fingerprint_none_for_unparseable_or_empty() {
        assert!(conversation_fingerprint(Provider::Anthropic, b"not json").is_none());
        assert!(conversation_fingerprint(Provider::Anthropic, br#"{"messages":[]}"#).is_none());
        assert!(conversation_fingerprint(Provider::Anthropic, br#"{"model":"x"}"#).is_none());
    }

    // --- response model extraction (per-model usage / coset) ----------
    // The model is parsed from the response for ALL THREE providers, from the
    // same body the usage comes from.

    #[test]
    fn response_model_anthropic_json_and_sse() {
        // Non-streaming: top-level `model`.
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_JSON_BODY);
        assert_eq!(
            feed_all(&resp, Provider::Anthropic).finalize().model.as_deref(),
            Some("claude-haiku-4-5")
        );
        // SSE: `message_start.message.model`.
        let mut sse = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        sse.extend_from_slice(ANTHROPIC_SSE_BODY);
        assert_eq!(
            feed_all(&sse, Provider::Anthropic).finalize().model.as_deref(),
            Some("claude-haiku-4-5")
        );
    }

    #[test]
    fn response_model_openai_json_and_sse() {
        // Non-streaming: top-level `model`.
        let body = br#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":7,"total_tokens":27}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        assert_eq!(
            feed_all(&resp, Provider::OpenAi).finalize().model.as_deref(),
            Some("gpt-4o-2024-08-06")
        );
        // Streaming: the final usage chunk carries `model`.
        let stream = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":7,\"total_tokens\":27}}\n\ndata: [DONE]\n\n";
        let mut sse = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        sse.extend_from_slice(stream);
        assert_eq!(
            feed_all(&sse, Provider::OpenAi).finalize().model.as_deref(),
            Some("gpt-4o-2024-08-06")
        );
    }

    #[test]
    fn response_model_gemini_uses_model_version() {
        // Gemini reports the resolved model as `modelVersion`, not `model`.
        let body = br#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4,"totalTokenCount":15},"modelVersion":"gemini-1.5-pro-002"}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        assert_eq!(
            feed_all(&resp, Provider::Gemini).finalize().model.as_deref(),
            Some("gemini-1.5-pro-002")
        );
    }

    #[test]
    fn response_model_anthropic_sse_without_usage() {
        // Anthropic's consumer web API (claude.ai / Claude Desktop) streams a
        // `message_start` that carries the served `model` but NO `usage`, and its
        // `message_delta` omits usage too. The model must still resolve from the
        // response even though the "usage" pre-filter never trips (regression for
        // the read-only h2 path, where the request body isn't captured, so the
        // response is the only model source). Tokens stay unmeasured.
        let mut sse = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_vec();
        sse.extend_from_slice(b"event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"chatcompl_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5-20251001\",\"content\":[]}}\r\n\r\n");
        sse.extend_from_slice(b"event: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\r\n\r\n");
        sse.extend_from_slice(b"event: message_delta\r\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\r\n\r\n");
        let f = feed_all(&sse, Provider::Anthropic).finalize();
        assert_eq!(f.model.as_deref(), Some("claude-haiku-4-5-20251001"));
        assert_eq!((f.tokens_in, f.tokens_out), (None, None), "no usage -> unmeasured");
    }

    #[test]
    fn response_model_none_when_absent() {
        // A response body with no model field -> None (caller falls back to the
        // request). The plain fixture above carries no usage either; here we use
        // a minimal usage-only body to isolate the missing-model case.
        let body = br#"{"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(body);
        assert!(feed_all(&resp, Provider::Anthropic).finalize().model.is_none());
    }

    // --- request model fallback (per-model usage / coset) -------------

    #[test]
    fn model_from_request_anthropic_and_openai_read_body() {
        let a = br#"{"model":"claude-opus-4-20250514","messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(
            model_from_request(Provider::Anthropic, a, Some("/v1/messages")).as_deref(),
            Some("claude-opus-4-20250514")
        );
        let o = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(
            model_from_request(Provider::OpenAi, o, Some("/v1/chat/completions")).as_deref(),
            Some("gpt-4o")
        );
    }

    #[test]
    fn model_from_request_gemini_reads_path_not_body() {
        let body = br#"{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}"#;
        assert_eq!(
            model_from_request(
                Provider::Gemini,
                body,
                Some("/v1beta/models/gemini-1.5-pro:generateContent")
            )
            .as_deref(),
            Some("gemini-1.5-pro")
        );
        // Streaming variant + query string.
        assert_eq!(
            model_from_request(
                Provider::Gemini,
                body,
                Some("/v1beta/models/gemini-1.5-pro-latest:streamGenerateContent?alt=sse")
            )
            .as_deref(),
            Some("gemini-1.5-pro-latest")
        );
        // Tuned model path resolves to the tuned id.
        assert_eq!(
            model_from_request(
                Provider::Gemini,
                body,
                Some("/v1beta/tunedModels/my-tuned-123:generateContent")
            )
            .as_deref(),
            Some("my-tuned-123")
        );
    }

    #[test]
    fn model_from_request_none_when_absent_or_unparseable() {
        assert!(model_from_request(Provider::Anthropic, br#"{"messages":[]}"#, None).is_none());
        assert!(model_from_request(Provider::OpenAi, br#"{"model":123}"#, None).is_none());
        assert!(model_from_request(Provider::Anthropic, br#"{"model":""}"#, None).is_none());
        assert!(model_from_request(Provider::OpenAi, b"not json", None).is_none());
        // Gemini with no path, or a path with no `:method` segment.
        assert!(model_from_request(Provider::Gemini, b"{}", None).is_none());
        assert!(model_from_request(Provider::Gemini, b"{}", Some("/v1beta/models")).is_none());
    }

    #[test]
    fn head_split_from_body() {
        // Head arrives in one read, body in the next.
        let p = feed_chunks(
            &[
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
                ANTHROPIC_JSON_BODY,
            ],
            Provider::Anthropic,
        );
        assert_eq!(p.measured(), (Some(14), Some(10)));
    }

    #[test]
    fn interim_100_continue_then_real_head() {
        let mut resp = b"HTTP/1.1 100 Continue\r\n\r\n".to_vec();
        resp.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n");
        resp.extend_from_slice(ANTHROPIC_JSON_BODY);
        let p = feed_all(&resp, Provider::Anthropic);
        assert_eq!(p.status, Some(200));
        assert_eq!(p.measured(), (Some(14), Some(10)));
    }

    #[test]
    fn unknown_content_type_keeps_status_but_no_usage() {
        let resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nsome body";
        let p = feed_all(resp, Provider::Anthropic);
        assert_eq!(p.status, Some(200));
        assert_eq!(p.measured(), (None, None));
    }

    #[test]
    fn error_status_still_captured() {
        let resp = b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n\r\n{\"error\":\"rate_limited\"}";
        let p = feed_all(resp, Provider::Anthropic);
        assert_eq!(p.status, Some(429));
        assert_eq!(p.measured(), (None, None));
    }

    #[test]
    fn dechunker_reassembles_body() {
        let original = b"hello world, this is a body long enough to span several small chunks!";
        let enc = chunk_encode(original);
        let mut d = Dechunker::new();
        let dec = d.push(&enc);
        assert_eq!(dec, original);
        assert!(d.done);
    }

    #[test]
    fn dechunker_handles_split_pushes() {
        let original = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOP";
        let enc = chunk_encode(original);
        let mid = enc.len() / 2;
        let mut d = Dechunker::new();
        let mut dec = d.push(&enc[..mid]);
        dec.extend_from_slice(&d.push(&enc[mid..]));
        assert_eq!(dec, original);
    }

    #[test]
    fn dechunker_parses_chunk_extension() {
        // "5;foo=bar\r\nhello\r\n0\r\n\r\n"
        let enc = b"5;foo=bar\r\nhello\r\n0\r\n\r\n";
        let mut d = Dechunker::new();
        let dec = d.push(enc);
        assert_eq!(dec, b"hello");
    }

    #[test]
    fn parse_hex_prefix_stops_at_non_hex() {
        assert_eq!(parse_hex_prefix(b"1a3"), 0x1a3);
        assert_eq!(parse_hex_prefix(b"ff;ext"), 0xff);
        assert_eq!(parse_hex_prefix(b"0"), 0);
    }

    // Perf micro-bench (not a correctness test; #[ignore]'d so it never runs in
    // normal `cargo test` or CI). Quantifies the inline parse cost the shim
    // adds per LLM request on the hot path, in isolation from the network.
    // Run with:
    //   cargo test -p clearml_snug_shim --release parse_cpu_microbench \
    //     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn parse_cpu_microbench() {
        // A large Anthropic SSE response: message_start (input usage), ~2000
        // content_block_delta events (the bulk - these must be cheaply skipped
        // by the "usage" pre-filter, NOT serde-parsed), then message_delta
        // (final output usage) and message_stop.
        let mut body = Vec::new();
        body.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
        body.extend_from_slice(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3000,\"output_tokens\":1}}}\n\n");
        for i in 0..2000 {
            let line = format!(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"token{} \"}}}}\n\n",
                i
            );
            body.extend_from_slice(line.as_bytes());
        }
        body.extend_from_slice(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2000}}\n\n");
        body.extend_from_slice(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");

        let kib = body.len() as f64 / 1024.0;
        let iters: u32 = 2000;
        // Feed in 16 KiB chunks to mimic SSL_read record sizing.
        let t0 = std::time::Instant::now();
        let (mut acc_in, mut acc_out) = (0u64, 0u64);
        for _ in 0..iters {
            let mut p = RespParse::new(Provider::Anthropic);
            for chunk in body.chunks(16 * 1024) {
                p.feed(chunk);
            }
            let (i, o) = p.measured();
            // Consume the result so the parse can't be optimized away.
            acc_in = acc_in.wrapping_add(i.unwrap_or(0));
            acc_out = acc_out.wrapping_add(o.unwrap_or(0));
        }
        let elapsed = t0.elapsed();
        let per_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
        let mibs = (kib * iters as f64 / 1024.0) / elapsed.as_secs_f64();
        eprintln!(
            "[microbench] response={:.0} KiB | {} iters in {:?} | {:.1} us/response | {:.0} MiB/s | checksum in={} out={}",
            kib, iters, elapsed, per_us, mibs, acc_in, acc_out
        );
        // Correctness sanity alongside the timing.
        assert_eq!(acc_in, 3000 * iters as u64);
        assert_eq!(acc_out, 2000 * iters as u64);
    }
}
