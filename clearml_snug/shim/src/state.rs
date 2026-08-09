//! Per-SSL connection state. Keyed by `SSL*` pointer treated as opaque
//! `usize`. A single `parking_lot::Mutex<HashMap<...>>` serializes all
//! observation work; lock contention is fine for the LLM-traffic
//! workloads the shim targets (a handful of long-lived connections).
//!
//! Also accumulates the full request/response bytes per request when the
//! call-history capture mode is on (`control::call_history_mode() != Off`) and
//! the host is whitelisted, handing each completed pair to `crate::call_history`
//! (a ring buffer for Collect/Dump, or a direct console emit for Continuous).

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use clearml_snug_messages::{CallHistoryMode, Direction, Event};
use parking_lot::Mutex;

use crate::body_scan;
use crate::call_history::{self, CapturedPair};
use crate::control;
use crate::decompress;
use crate::inject;
use crate::meter;
use crate::parser::{self, ParseOutcome};
use crate::tokens;
use crate::whitelist::Whitelist;

/// Per-(SSL*) state. Mixes two scopes:
///   * Per-CONNECTION (lives across keep-alive requests, only reset on
///     ``SSL_free``): ``http2_detected``, ``h2`` (the demux + per-stream
///     state), ``suppress``. HTTP/2 is a connection-wide decision; TLS+SNI pins
///     the destination host so suppression is also connection-wide.
///   * Per-REQUEST (reset every time a new HTTP/1 request line is
///     detected mid-connection): everything else (host/path/method,
///     bytes_tx/rx, tokens_in/out, started_at, request_started_emitted,
///     cap_* capture buffers, tokenizer). HTTP/1.1 keep-alive sends N requests
///     on one SSL*; each gets its own RequestStarted/RequestCompleted
///     pair via `reset_per_request` below.
pub(crate) struct ConnectionState {
    // ---- Per-request (reset by reset_per_request) ----------------------
    started_at: Instant,
    pub(crate) host: Option<String>,
    pub(crate) path: Option<String>,
    pub(crate) method: Option<String>,
    pub(crate) bytes_tx: u64,
    pub(crate) bytes_rx: u64,
    pub(crate) request_started_emitted: bool,
    /// Whether to capture this request's full request/response bytes for the
    /// call-history feature. Sampled at request start: the capture mode is on
    /// (`call_history_mode() != Off`) AND the host is whitelisted.
    cap_active: bool,
    /// Full request bytes (request line + headers + body) and full response
    /// bytes (status line + headers + body), each capped at
    /// `control::call_history_cap_bytes()`. Populated only when `cap_active`;
    /// reset per request alongside the metering fields. The request is redacted
    /// (credentials masked) when it's turned into a `CapturedPair`.
    cap_req: Vec<u8>,
    cap_resp: Vec<u8>,
    cap_req_truncated: bool,
    cap_resp_truncated: bool,
    /// Cached tokenizer for the in-flight request. Set from the
    /// matched rule's `tokenizer` field; falls back to the
    /// `CLEARML_SNUG_DEFAULT_TOKENIZER` env var when no rule matched.
    /// Always Some(..) for non-suppressed requests after their first
    /// request-line parse.
    pub(crate) tokenizer: Option<String>,
    /// Running tokens counts for the in-flight request; emitted on
    /// RequestCompleted then reset.
    pub(crate) tokens_in: u64,
    pub(crate) tokens_out: u64,
    /// Per-response usage scanner. Lazily created on the first response read
    /// of a known-provider connection when usage parsing is enabled. Holds the
    /// parsed HTTP status and provider-reported token counts; consumed in
    /// `build_request_completed`. Reset with the other per-request fields.
    resp: Option<body_scan::RespParse>,
    /// h2 response transfer-encoding, decided from the first DATA frame's magic
    /// bytes. For gzip/zstd the raw compressed body is buffered into
    /// `resp_comp_buf` and inflated once at END_STREAM before the usage scan
    /// (the shim has no streaming inflater); for identity the body is scanned
    /// incrementally via `resp`. Only meaningful on h2 streams.
    resp_enc: decompress::Encoding,
    /// Raw compressed h2 response body accumulated when `resp_enc` is gzip/zstd,
    /// capped at `H2_COMP_CAP`; inflated + scanned at END_STREAM. Empty otherwise.
    resp_comp_buf: Vec<u8>,
    /// Accumulated request body (capped) for known-provider whitelisted hosts
    /// when usage parsing is on; parsed at completion for the freshest turn's
    /// tool errors. Reset per request.
    req_body: Vec<u8>,
    /// Cached gate: whether to capture this request's body for tool-error
    /// parsing (set on the request-line write, used by body-continuation writes).
    parse_req_body: bool,
    /// Set once this request's `RequestCompleted` has been emitted early (the
    /// moment its chunked response terminated, in `observe_read_inner`). Guards
    /// the deferred emit sites (next-request boundary / `SSL_free` / exit drain,
    /// all via `build_request_completed`) so the event is emitted exactly once.
    /// Without the early emit, the LAST request of a run is lost to the bounded
    /// exit drain. Reset per request.
    completed_emitted: bool,

    // ---- Per-connection (NOT reset by reset_per_request) ----------------
    pub(crate) http2_detected: bool,
    /// Per-connection HTTP/2 demux + per-`stream_id` request state, `None` until
    /// the h2 preface is seen; then every write/read is fed to it. Boxed so the
    /// nested per-stream `ConnectionState` (each carrying its own `h2: None`)
    /// keeps this struct sized.
    h2: Option<Box<H2State>>,
    /// Connection-level opt-out: no events for this SSL* while set. Two causes:
    /// the self-host exclusion (permanent) and the `default_action == "ignore"`
    /// + unmatched-host decision. TLS+SNI pins the host per SSL*, so within one
    /// whitelist this needs no per-request re-evaluation; a whitelist hot-reload
    /// re-arms the `ignore` case (see `suppress_rearmable`) so a newly-added host
    /// meters the connection's next request.
    pub(crate) suppress: bool,
    /// True when `suppress` was set by the `default_action == "ignore"` opt-out
    /// (NOT the permanent self-host exclusion). Only these suppressions are
    /// cleared by `rearm_whitelist_suppressions` on a whitelist hot-reload, so a
    /// host added at runtime can start metering an already-open keep-alive
    /// connection at its next request.
    pub(crate) suppress_rearmable: bool,
    /// True once we've processed the first observe_write call on this
    /// connection. The "should we suppress?" decision is made once on
    /// that first write (based on whether the first parse outcome
    /// matched any whitelist rule); this flag gates the decision so
    /// per-write parsing on subsequent keep-alive requests doesn't
    /// accidentally re-trigger it. Reset to false only when a whitelist
    /// hot-reload re-arms the connection (`rearm_whitelist_suppressions`),
    /// so the decision re-runs against the new whitelist.
    pub(crate) first_write_seen: bool,
}

impl ConnectionState {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            host: None,
            path: None,
            method: None,
            bytes_tx: 0,
            bytes_rx: 0,
            request_started_emitted: false,
            cap_active: false,
            cap_req: Vec::new(),
            cap_resp: Vec::new(),
            cap_req_truncated: false,
            cap_resp_truncated: false,
            tokenizer: None,
            tokens_in: 0,
            tokens_out: 0,
            resp: None,
            resp_enc: decompress::Encoding::Undecided,
            resp_comp_buf: Vec::new(),
            req_body: Vec::new(),
            parse_req_body: false,
            completed_emitted: false,
            http2_detected: false,
            h2: None,
            suppress: false,
            suppress_rearmable: false,
            first_write_seen: false,
        }
    }

    /// Reset the per-request fields. Called when a new HTTP/1 request
    /// line is detected mid-connection (keep-alive). Per-connection
    /// fields (http2_detected, h2, suppress) are
    /// intentionally NOT reset because they describe the SSL session,
    /// not the request.
    fn reset_per_request(&mut self) {
        self.started_at = Instant::now();
        self.host = None;
        self.path = None;
        self.method = None;
        self.bytes_tx = 0;
        self.bytes_rx = 0;
        self.request_started_emitted = false;
        self.cap_active = false;
        self.cap_req = Vec::new();
        self.cap_resp = Vec::new();
        self.cap_req_truncated = false;
        self.cap_resp_truncated = false;
        self.tokenizer = None;
        self.tokens_in = 0;
        self.tokens_out = 0;
        self.resp = None;
        self.resp_enc = decompress::Encoding::Undecided;
        self.resp_comp_buf = Vec::new();
        self.req_body = Vec::new();
        self.parse_req_body = false;
        self.completed_emitted = false;
    }
}

/// Per-connection HTTP/2 state: a frame demux per direction plus one
/// `ConnectionState` per active `stream_id`. h2 multiplexes many concurrent
/// request/response exchanges over one SSL*, so — unlike HTTP/1 keep-alive,
/// which is one-request-at-a-time in the parent `ConnectionState` — each stream
/// gets its own request state and its own RequestStarted/RequestCompleted. A
/// stream reuses the full `ConnectionState` machinery (with its own `h2: None`),
/// so `complete_request`/`build_request_completed` apply per-stream unchanged.
struct H2State {
    /// client->server frames (request bodies).
    tx: crate::h2::FrameParser,
    /// server->client frames (response bodies).
    rx: crate::h2::FrameParser,
    streams: HashMap<u32, ConnectionState>,
}

impl H2State {
    fn new() -> Self {
        Self {
            tx: crate::h2::FrameParser::new_client(),
            rx: crate::h2::FrameParser::new_server(),
            streams: HashMap::new(),
        }
    }
}

static STATE: OnceLock<Mutex<HashMap<usize, ConnectionState>>> = OnceLock::new();

fn map() -> &'static Mutex<HashMap<usize, ConnectionState>> {
    STATE.get_or_init(|| Mutex::new(HashMap::with_capacity(64)))
}

// --- Public entry points -------------------------------------------------

pub fn observe_write(ssl: usize, buf: &[u8]) -> Option<Vec<u8>> {
    let mut guard = map().lock();
    let st = guard.entry(ssl).or_insert_with(ConnectionState::new);
    let wl = crate::whitelist::current();
    let self_hosts = crate::self_host::current();
    observe_write_inner(
        st,
        ssl as u64,
        buf,
        &wl,
        self_hosts,
        crate::init::project_id(),
        crate::init::task_id(),
        &mut |e| meter::emit(e),
    )
}

pub fn observe_read(ssl: usize, buf: &[u8]) {
    let mut guard = map().lock();
    let st = guard.entry(ssl).or_insert_with(ConnectionState::new);
    observe_read_inner(st, ssl as u64, buf, &mut |e| meter::emit(e));
}

pub fn observe_free(ssl: usize) {
    let mut guard = map().lock();
    if let Some(st) = guard.remove(&ssl) {
        complete_request(&st, ssl as u64, &mut |e| meter::emit(e));
    }
}

/// Drain every still-open connection: emit `RequestCompleted` for any
/// entry that has a `RequestStarted` but no matching `RequestCompleted`,
/// then remove it from the map. Called from `hooks/exit.rs` before
/// chaining to the real `exit(3)`, while the Rust runtime + IPC are
/// still alive.
///
/// Without this, a keep-alive connection whose SSL_free hasn't fired
/// by process exit would drop its trailing `RequestCompleted` silently
/// (the final outbound LLM call per task producing no usage event).
///
/// Safe to call when the map is empty / uninitialised and idempotent
/// on repeated calls.
pub fn flush_all_pending_requests() {
    let map_ref = match STATE.get() {
        // STATE never initialised - no SSL traffic ever happened.
        Some(m) => m,
        None => return,
    };
    let mut guard = map_ref.lock();
    flush_all_pending_inner(&mut guard, &mut |e| meter::emit(e));
}

/// Pure helper extracted from `flush_all_pending_requests` so tests can
/// drive it with an in-memory map and a collecting sink, without
/// touching the global `STATE` or the real IPC emitter.
fn flush_all_pending_inner<F: FnMut(Event)>(
    map: &mut HashMap<usize, ConnectionState>,
    emit: &mut F,
) {
    for (ssl, st) in map.drain() {
        complete_request(&st, ssl as u64, emit);
    }
}

/// Re-arm connections suppressed by the `default_action == "ignore"` opt-out so
/// their suppression is re-evaluated against the just-hot-reloaded whitelist.
/// Called from `whitelist::apply_whitelist_additions` after a swap. A re-armed
/// connection re-runs its first-write decision on its NEXT write: a now-
/// whitelisted host starts metering at the next request; a still-unmatched host
/// re-suppresses. Self-host suppression is permanent and never re-armed. Safe
/// and a no-op when no SSL traffic has happened yet.
pub fn rearm_whitelist_suppressions() {
    if let Some(m) = STATE.get() {
        rearm_inner(&mut m.lock());
    }
}

/// Pure core of `rearm_whitelist_suppressions`, testable with a local map.
fn rearm_inner(map: &mut HashMap<usize, ConnectionState>) {
    for st in map.values_mut() {
        if st.suppress && st.suppress_rearmable {
            st.suppress = false;
            st.suppress_rearmable = false;
            // Re-run the first-write suppression decision on the next write.
            st.first_write_seen = false;
        }
    }
}

// --- Pure helpers ------------------------------------------------------

fn observe_write_inner(
    st: &mut ConnectionState,
    conn_id: u64,
    buf: &[u8],
    whitelist: &Whitelist,
    self_hosts: &[String],
    project_id: &str,
    task_id: &str,
    sink: &mut dyn FnMut(Event),
) -> Option<Vec<u8>> {
    // Connections we already opted out of stay silent for their entire
    // lifetime. TLS+SNI pins the destination host per SSL*, so the
    // suppression decision (set when the host first didn't match any
    // whitelist rule and `default_action == "ignore"`) applies to every
    // subsequent write too.
    if st.suppress {
        return None;
    }

    let mut spliced: Option<Vec<u8>> = None;
    let mut is_new_request = false;

    // We try to parse on EVERY write (not just the first one on the
    // connection). The reason: HTTP/1.1 keep-alive lets urllib3 send N
    // requests through one SSL*, with no SSL_free in between. Each new
    // request starts with a fresh "METHOD path HTTP/1.1\r\n..." block.
    // If we only parsed the first write we'd miss every request after
    // the first - no RequestStarted, no per-request token attribution,
    // no fresh project:/session: injection. So: parse every write,
    // treat a `Complete` parse as a new request boundary, and treat
    // `Incomplete`/`NotHttp` as body data for the in-flight request.
    //
    // We skip parsing on HTTP/2 connections because HPACK frames can
    // incidentally contain ASCII method-shaped bytes; once a connection
    // has been tagged HTTP/2 it stays that way until SSL_free.
    if !st.http2_detected {
        let parse = parser::parse_first_write(buf);

        // Default-action enforcement, done ONCE per connection on the
        // very first write. If the first parse couldn't identify
        // a whitelisted host AND the operator chose `default_action ==
        // "ignore"`, opt this connection out for its entire lifetime.
        // This covers HTTP/2, partial-headers, and non-HTTP first
        // writes too: under "ignore" we only track explicitly
        // whitelisted hosts; anything we can't identify on first
        // contact is treated as not opted-in.
        if !st.first_write_seen {
            st.first_write_seen = true;
            let matched_anything = match &parse {
                ParseOutcome::Http1 { host, path, .. } => {
                    whitelist.matches(host, path).is_some()
                }
                _ => false,
            };
            if !matched_anything && whitelist.default_action == "ignore" {
                st.suppress = true;
                // Re-armable: a whitelist hot-reload re-evaluates this decision
                // (unlike the permanent self-host suppression below), so a host
                // added at runtime meters this connection's next request.
                st.suppress_rearmable = true;
                return spliced;
            }
        }

        match parse {
            ParseOutcome::Http1 { method, host, path } => {
                // Self-host exclusion (overrides whitelist + default_action):
                // the task's own ClearML SDK traffic to the backend this very
                // task reports to must never be metered/billed/injected. TLS+SNI
                // pins the host per SSL*, so pinning suppress here silences every
                // subsequent write on the connection too (the early return at the
                // top of this fn). No event was emitted yet, so nothing downstream
                // (RequestCompleted, usage, scalars) ever fires for it.
                if crate::self_host::matches(self_hosts, &host) {
                    st.suppress = true;
                    return None;
                }
                // New HTTP/1 request boundary. Close out whatever was
                // in flight (if anything) before opening the new one.
                if st.request_started_emitted {
                    complete_request(st, conn_id, sink);
                    st.reset_per_request();
                }

                let matched = whitelist.matches(&host, &path);
                let whitelisted = matched.is_some();

                st.method = Some(method.clone());
                st.host = Some(host.clone());
                st.path = Some(path.clone());
                st.request_started_emitted = true;
                st.started_at = std::time::Instant::now();
                // Call-history capture: active when the mode is on AND the host
                // is whitelisted (we capture the external-provider calls). The
                // full first write (request line + headers + any body) is `buf`.
                st.cap_active =
                    whitelisted && control::call_history_mode() != CallHistoryMode::Off;
                // Cache the tokenizer name. Falls back to the
                // process-global default (CLEARML_SNUG_DEFAULT_TOKENIZER)
                // so every metered request has a tokenizer, not just
                // whitelisted ones.
                st.tokenizer = Some(
                    matched
                        .map(|r| r.tokenizer.clone())
                        .unwrap_or_else(|| control::default_tokenizer().to_string()),
                );

                let should_inject = matched.map(|r| r.inject_headers).unwrap_or(false);
                // Force identity content-encoding on whitelisted requests when
                // usage parsing is on, so the response body arrives
                // uncompressed for the scanner. Independent of the rule's
                // inject_headers flag; both edits go through one rewrite pass.
                let force_identity = whitelisted && control::parse_usage_enabled();
                if should_inject || force_identity {
                    spliced = inject::rewrite_headers(
                        buf,
                        project_id,
                        task_id,
                        should_inject,
                        force_identity,
                    );
                }
                let inject_headers = should_inject && spliced.is_some();

                // M2: capture this request's body (for the freshest turn's tool
                // errors) on known-provider whitelisted hosts when parsing is
                // on. Cached in parse_req_body for the body-continuation writes.
                is_new_request = true;
                st.parse_req_body = whitelisted
                    && control::parse_usage_enabled()
                    && body_scan::provider_for_host(&host).is_some();
                if st.parse_req_body {
                    if let Some(bstart) = body_start_offset(buf) {
                        append_req_body(st, &buf[bstart..]);
                    }
                }
                // Call-history: capture the FULL request (request line + headers
                // + body) starting from this first write. Continuation writes
                // append below.
                if st.cap_active {
                    append_cap_req(st, buf);
                }

                sink(Event::RequestStarted {
                    conn_id,
                    ts_ms: Event::now_ts_ms(),
                    host,
                    path,
                    method,
                    whitelisted,
                    inject_headers,
                });
            }
            ParseOutcome::Http2 => {
                // Tag the connection h2 for its whole lifetime and stand up the
                // demux. This first write carried the preface; it (and every
                // subsequent write) is fed to `process_h2_tx` below — the tx
                // parser skips the client preface.
                st.http2_detected = true;
                if st.h2.is_none() {
                    st.h2 = Some(Box::new(H2State::new()));
                }
            }
            // Incomplete or NotHttp: this write is body data for the
            // in-flight request (or for a non-HTTP-1 connection that
            // never parses). Just count bytes below; no event lifecycle
            // change.
            ParseOutcome::Incomplete | ParseOutcome::NotHttp => {}
        }
    }

    // HTTP/2: once detected, every write is a raw frame stream (no HTTP/1
    // request line). Feed it to the per-connection demux, which reassembles
    // per-stream request DATA and drives each stream's lifecycle. Gated on usage
    // parsing being on (matches the HTTP/1 response-scan gate).
    if st.http2_detected && control::parse_usage_enabled() {
        process_h2_tx(st, conn_id, buf, sink);
    }

    // M2: body-continuation writes (anything that wasn't a fresh request line)
    // for a request whose body we're capturing for tool-error parsing.
    if st.parse_req_body && !is_new_request {
        append_req_body(st, buf);
    }
    // Call-history: continuation writes append to the full-request capture.
    if st.cap_active && !is_new_request {
        append_cap_req(st, buf);
    }

    // Always count bytes + emit BytesObserved (unless we just flipped
    // suppress=true above, in which case the early return inside the
    // Http1 arm already short-circuited us).
    st.bytes_tx = st.bytes_tx.saturating_add(buf.len() as u64);
    let est_tx = estimate_tokens_for(st, buf.len() as u64);
    st.tokens_in = st.tokens_in.saturating_add(est_tx);
    sink(Event::BytesObserved {
        conn_id,
        ts_ms: Event::now_ts_ms(),
        direction: Direction::Tx,
        bytes: buf.len() as u64,
        tokens_est: est_tx,
    });

    spliced
}

fn observe_read_inner(
    st: &mut ConnectionState,
    conn_id: u64,
    buf: &[u8],
    sink: &mut dyn FnMut(Event),
) {
    st.bytes_rx = st.bytes_rx.saturating_add(buf.len() as u64);
    if st.suppress {
        return;
    }
    let est_rx = estimate_tokens_for(st, buf.len() as u64);
    st.tokens_out = st.tokens_out.saturating_add(est_rx);
    sink(Event::BytesObserved {
        conn_id,
        ts_ms: Event::now_ts_ms(),
        direction: Direction::Rx,
        bytes: buf.len() as u64,
        tokens_est: est_rx,
    });

    // Bootstrap HTTP/2 from the SSL_read side for connections whose SSL_write we
    // never observe (only SSL_read is hookable). The tx path sets
    // `http2_detected` on the client preface; without it, detect the server's
    // opening SETTINGS frame here, tag the connection h2 and stand up the demux,
    // then fall through to feed these (and every later) read bytes to
    // `process_h2_rx`. Gated on usage parsing (mirrors the tx gate) and only
    // before any HTTP/1 request was seen, so the tx+rx path and ordinary HTTP/1
    // responses are untouched.
    if !st.http2_detected
        && !st.request_started_emitted
        && control::parse_usage_enabled()
        && looks_like_h2_server_start(buf)
    {
        st.http2_detected = true;
        if st.h2.is_none() {
            st.h2 = Some(Box::new(H2State::new()));
        }
    }

    // HTTP/2: feed the raw frame stream to the demux; each stream's
    // RequestCompleted (with body-parsed usage) is emitted on its END_STREAM.
    // The HTTP/1 call-history + response-scan path below does not apply to h2
    // framing, so return once handled.
    if st.http2_detected {
        if control::parse_usage_enabled() {
            process_h2_rx(st, conn_id, buf, sink);
        }
        return;
    }

    // Call-history: capture the FULL response (status line + headers + body) as
    // the scanner sees it, BEFORE the early-emit below so a completing chunked
    // response captures all of its bytes into the pair.
    if st.cap_active {
        append_cap_resp(st, buf);
    }

    // Parse provider-reported usage out of the response body when a reporting
    // sink is enabled (the parent sets CLEARML_SNUG_PARSE_USAGE only then).
    // Lazily create the per-response scanner on the first read of a
    // known-provider connection, then feed every chunk. Cheap and best-effort
    // (see body_scan); the measured counts are consumed at request completion.
    if control::parse_usage_enabled() {
        if st.resp.is_none() {
            if let Some(p) = st.host.as_deref().and_then(body_scan::provider_for_host) {
                st.resp = Some(body_scan::RespParse::new(p));
            }
        }
        if let Some(r) = st.resp.as_mut() {
            r.feed(buf);
        }
        // Emit RequestCompleted the moment the (chunked) response terminates,
        // rather than deferring to the next request boundary / SSL_free / exit
        // drain. The LAST request of a run never sees a next boundary, and the
        // exit drain is bounded — so without this its metering is lost. The
        // `completed_emitted` guard (checked in build_request_completed) keeps
        // every deferred site from re-emitting. Non-chunked / unparsed responses
        // report `is_complete() == false` and stay on the deferred path.
        let complete = st.resp.as_ref().map_or(false, |r| r.is_complete());
        if complete && st.request_started_emitted && !st.completed_emitted {
            complete_request(st, conn_id, sink);
            st.completed_emitted = true;
        }
    }
}

/// The h2 host assumed while HPACK `:authority` is still undecoded. This is a
/// per-app default: an app whose traffic is single-provider can rely on one
/// assumed host per connection. Defaults to `api.anthropic.com` (a reasonable
/// default for such apps), overridable via `CLEARML_SNUG_H2_ASSUMED_HOST` for
/// other hosts. Full HPACK `:authority` decode is the eventual correct fix. Read
/// once and cached so it isn't re-parsed per stream.
fn h2_assumed_host() -> &'static str {
    static HOST: OnceLock<String> = OnceLock::new();
    HOST.get_or_init(|| {
        std::env::var("CLEARML_SNUG_H2_ASSUMED_HOST")
            .unwrap_or_else(|_| "api.anthropic.com".to_string())
    })
}

/// Build a fresh per-stream request state for an h2 stream. Defaults the host to
/// the assumed h2 host (see `h2_assumed_host`), which enables provider detection
/// + request-body capture for model/chat attribution. `h2` stays `None` (a
/// stream is not itself an h2 connection), so `complete_request` treats it as a
/// plain request.
fn new_h2_stream() -> ConnectionState {
    let mut s = ConnectionState::new();
    s.host = Some(h2_assumed_host().to_string());
    s.tokenizer = Some(control::default_tokenizer().to_string());
    s.parse_req_body = true;
    s
}

/// Emit the stream's `RequestStarted` exactly once. Path/method are HPACK
/// pseudo-headers (decoded later); empty for now. `whitelisted` reflects that
/// the connection already passed the `default_action` gate.
fn start_h2_stream(stream: &mut ConnectionState, conn_id: u64, sink: &mut dyn FnMut(Event)) {
    if stream.request_started_emitted {
        return;
    }
    stream.request_started_emitted = true;
    stream.started_at = Instant::now();
    sink(Event::RequestStarted {
        conn_id,
        ts_ms: Event::now_ts_ms(),
        host: stream.host.clone().unwrap_or_default(),
        path: String::new(),
        method: String::new(),
        whitelisted: true,
        inject_headers: false,
    });
}

/// Spike diagnostic gate: `CLEARML_SNUG_H2_DEBUG=1` logs the live h2 frame flow
/// (both directions, per stream) so a misfiring usage-completion path is visible
/// without a debugger. Cached; off by default.
fn h2_dbg() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(0);
    match C.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var("CLEARML_SNUG_H2_DEBUG")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            C.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Cap on the raw compressed h2 response body buffered per stream before the
/// inflate at END_STREAM. Chat completions compress to far less; an oversized
/// body (a large sync) is truncated — `read_capped` keeps the usage prelude.
const H2_COMP_CAP: usize = 8 * 1024 * 1024;
/// Cap on inflated output, bounding memory against a compression bomb.
const H2_DECOMP_CAP: usize = 16 * 1024 * 1024;

/// Cap on the inflated REQUEST body (the captured request is itself capped at
/// `body_scan::REQ_BODY_CAP`, so its inflation is bounded too; this guards a
/// decompression bomb). Mirrors `H2_DECOMP_CAP` for responses.
const REQ_DECOMP_CAP: usize = 8 * 1024 * 1024;

/// Feed a client->server h2 chunk: reassemble each stream's request DATA and
/// open its lifecycle. Pure (no global gate); the caller gates on
/// `parse_usage_enabled`.
fn process_h2_tx(st: &mut ConnectionState, conn_id: u64, buf: &[u8], sink: &mut dyn FnMut(Event)) {
    let h2 = match st.h2.as_mut() {
        Some(h) => h,
        None => return,
    };
    for f in h2.tx.feed(buf) {
        if h2_dbg() {
            snug_log!(
                "[snug-h2dbg] tx frame type={:#x} stream={} len={} end={}",
                f.ftype, f.stream_id, f.payload.len(), f.end_stream()
            );
        }
        match f.ftype {
            crate::h2::FRAME_HEADERS => {
                let stream = h2.streams.entry(f.stream_id).or_insert_with(new_h2_stream);
                start_h2_stream(stream, conn_id, sink);
            }
            crate::h2::FRAME_DATA => {
                let stream = h2.streams.entry(f.stream_id).or_insert_with(new_h2_stream);
                start_h2_stream(stream, conn_id, sink);
                append_req_body(stream, &f.payload);
                stream.bytes_tx = stream.bytes_tx.saturating_add(f.payload.len() as u64);
                // Byte-ratio token estimate, mirroring the HTTP/1 write path
                // (observe_write); the fallback when the response carries no
                // parseable usage. build_request_completed prefers measured usage.
                let est = estimate_tokens_for(stream, f.payload.len() as u64);
                stream.tokens_in = stream.tokens_in.saturating_add(est);
            }
            _ => {}
        }
    }
}

/// Conservative test that a server->client read chunk begins an HTTP/2
/// connection: its first frame is a SETTINGS frame (RFC 7540 §6.5) — a 9-byte
/// header carrying type 0x4 on stream 0 with a length that is a multiple of 6
/// (each setting is a 6-byte entry). Lets the read side bootstrap h2 when the
/// SSL_write preface is never seen (connections where only SSL_read is
/// observable), without misclassifying an HTTP/1 response (which begins with the
/// ASCII "HTTP/1" status line).
fn looks_like_h2_server_start(buf: &[u8]) -> bool {
    if buf.len() < 9 || buf.starts_with(b"HTTP/1") {
        return false;
    }
    let len = ((buf[0] as usize) << 16) | ((buf[1] as usize) << 8) | buf[2] as usize;
    let stream_id = (((buf[5] as u32) << 24)
        | ((buf[6] as u32) << 16)
        | ((buf[7] as u32) << 8)
        | buf[8] as u32)
        & 0x7fff_ffff;
    buf[3] == crate::h2::FRAME_SETTINGS && stream_id == 0 && len % 6 == 0
}

/// Feed a server->client h2 chunk: append each stream's response DATA to its
/// usage scanner and, on END_STREAM, finalize + emit its `RequestCompleted`.
/// Pure (no global gate); the caller gates on `parse_usage_enabled`.
fn process_h2_rx(st: &mut ConnectionState, conn_id: u64, buf: &[u8], sink: &mut dyn FnMut(Event)) {
    let h2 = match st.h2.as_mut() {
        Some(h) => h,
        None => return,
    };
    let mut completed: Vec<u32> = Vec::new();
    for f in h2.rx.feed(buf) {
        if h2_dbg() {
            let reg = h2.streams.contains_key(&f.stream_id);
            snug_log!(
                "[snug-h2dbg] rx frame type={:#x} stream={} len={} end={} registered={}",
                f.ftype, f.stream_id, f.payload.len(), f.end_stream(), reg
            );
        }
        match f.ftype {
            crate::h2::FRAME_DATA => {
                // Auto-register the stream on the response side: for rx-only
                // connections (only SSL_read observable, no SSL_write hook) the
                // stream was never opened by `process_h2_tx`. `start_h2_stream`
                // is once-guarded, so the tx+rx path (apps that expose both write
                // and read) finds the existing stream and emits no duplicate
                // RequestStarted.
                h2.streams.entry(f.stream_id).or_insert_with(new_h2_stream);
                if let Some(stream) = h2.streams.get_mut(&f.stream_id) {
                    start_h2_stream(stream, conn_id, sink);
                    if !f.payload.is_empty() {
                        // Decide the response encoding once, from the first DATA
                        // frame's magic bytes (the Content-Encoding header is
                        // HPACK-hidden over h2). Identity bodies are scanned
                        // incrementally; gzip/zstd bodies are buffered raw and
                        // inflated at END_STREAM (no streaming inflater in-process).
                        if stream.resp_enc == decompress::Encoding::Undecided {
                            stream.resp_enc = decompress::detect(&f.payload);
                            if h2_dbg() {
                                let n = f.payload.len().min(240);
                                snug_log!(
                                    "[snug-h2dbg] stream={} first-resp-DATA enc={:?} {}B preview={:?}",
                                    f.stream_id, stream.resp_enc, f.payload.len(),
                                    String::from_utf8_lossy(&f.payload[..n])
                                );
                            }
                            if stream.resp_enc == decompress::Encoding::Identity
                                && stream.resp.is_none()
                            {
                                if let Some(p) =
                                    stream.host.as_deref().and_then(body_scan::provider_for_host)
                                {
                                    let sse = body_scan::looks_like_sse(&f.payload);
                                    stream.resp = Some(body_scan::RespParse::new_h2_body(p, sse));
                                }
                            }
                        }
                        if stream.resp_enc.is_compressed() {
                            let room = H2_COMP_CAP.saturating_sub(stream.resp_comp_buf.len());
                            if room > 0 {
                                let take = room.min(f.payload.len());
                                stream.resp_comp_buf.extend_from_slice(&f.payload[..take]);
                            }
                        } else if let Some(r) = stream.resp.as_mut() {
                            r.feed(&f.payload);
                            if h2_dbg() {
                                let n = f.payload.len().min(200);
                                let s = String::from_utf8_lossy(&f.payload[..n]);
                                if !s.starts_with(':') {
                                    snug_log!(
                                        "[snug-h2dbg] stream={} resp-DATA {}B: {:?}",
                                        f.stream_id, f.payload.len(), s
                                    );
                                }
                            }
                        }
                        stream.bytes_rx = stream.bytes_rx.saturating_add(f.payload.len() as u64);
                        // Byte-ratio estimate for plaintext bodies (mirrors the
                        // HTTP/1 read path). Compressed bodies estimate from the
                        // inflated length at END_STREAM instead (below).
                        if !stream.resp_enc.is_compressed() {
                            let est = estimate_tokens_for(stream, f.payload.len() as u64);
                            stream.tokens_out = stream.tokens_out.saturating_add(est);
                        }
                    }
                    if f.end_stream() {
                        completed.push(f.stream_id);
                    }
                }
            }
            crate::h2::FRAME_HEADERS if f.end_stream() => {
                // Headers-only response (no body) — still a completed request.
                // Auto-register for the rx-only path (once-guarded on tx+rx).
                let stream = h2.streams.entry(f.stream_id).or_insert_with(new_h2_stream);
                start_h2_stream(stream, conn_id, sink);
                completed.push(f.stream_id);
            }
            _ => {}
        }
    }
    for sid in completed {
        if let Some(mut stream) = h2.streams.remove(&sid) {
            // Compressed body: inflate the buffer now and run the usage scan on
            // the plaintext, mirroring the incremental path's setup. A truncated
            // body still often carries the usage prelude (kept by `read_capped`).
            if stream.resp_enc.is_compressed() && !stream.resp_comp_buf.is_empty() {
                if let Some(p) = stream.host.as_deref().and_then(body_scan::provider_for_host) {
                    match decompress::decompress(
                        stream.resp_enc,
                        &stream.resp_comp_buf,
                        H2_DECOMP_CAP,
                    ) {
                        Some(plain) => {
                            let sse = body_scan::looks_like_sse(&plain);
                            let mut rp = body_scan::RespParse::new_h2_body(p, sse);
                            rp.feed(&plain);
                            stream.resp = Some(rp);
                            // Byte-ratio estimate from the INFLATED length (the
                            // buffered bytes_rx were compressed); fallback when the
                            // body carries no parseable usage.
                            let est = estimate_tokens_for(&stream, plain.len() as u64);
                            stream.tokens_out = stream.tokens_out.saturating_add(est);
                            if h2_dbg() {
                                let n = plain.len().min(240);
                                snug_log!(
                                    "[snug-h2dbg] stream={} inflated {}->{}B sse={} preview={:?}",
                                    sid,
                                    stream.resp_comp_buf.len(),
                                    plain.len(),
                                    sse,
                                    String::from_utf8_lossy(&plain[..n])
                                );
                            }
                        }
                        None if h2_dbg() => snug_log!(
                            "[snug-h2dbg] stream={sid} inflate FAILED enc={:?} {}B",
                            stream.resp_enc,
                            stream.resp_comp_buf.len()
                        ),
                        None => {}
                    }
                }
            }
            if h2_dbg() {
                snug_log!("[snug-h2dbg] COMPLETE stream={sid}");
            }
            complete_request(&stream, conn_id, sink);
        }
    }
}

/// Returns the token-estimate for `bytes` using the connection's cached
/// tokenizer name. Returns 0 when no rule matched (no tokenizer hint).
/// This keeps per-write computation a single division. The served model isn't
/// known yet at this incremental per-chunk point, so the model-aware ratio can't
/// be selected here — the estimator keeps its documented default (see
/// `tokens::bytes_per_token`).
fn estimate_tokens_for(st: &ConnectionState, bytes: u64) -> u64 {
    match st.tokenizer.as_deref() {
        Some(t) => tokens::estimate_tokens(bytes, t, None),
        None => 0,
    }
}

/// Append `data` to the per-request body buffer, stopping at `REQ_BODY_CAP`.
/// A body that exceeds the cap can't be JSON-parsed, so it simply yields no
/// tool-error count rather than growing memory without bound.
fn append_req_body(st: &mut ConnectionState, data: &[u8]) {
    let len = st.req_body.len();
    if len >= body_scan::REQ_BODY_CAP {
        return;
    }
    let take = (body_scan::REQ_BODY_CAP - len).min(data.len());
    st.req_body.extend_from_slice(&data[..take]);
}

/// Offset of the request body (just past the `\r\n\r\n` header terminator).
fn body_start_offset(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

/// Append `data` to the call-history request capture, capped at
/// `control::call_history_cap_bytes()` (0 = uncapped), flagging truncation.
fn append_cap_req(st: &mut ConnectionState, data: &[u8]) {
    let cap = control::call_history_cap_bytes();
    append_capped(&mut st.cap_req, &mut st.cap_req_truncated, data, cap);
}

/// Append `data` to the call-history response capture (see `append_cap_req`).
fn append_cap_resp(st: &mut ConnectionState, data: &[u8]) {
    let cap = control::call_history_cap_bytes();
    append_capped(&mut st.cap_resp, &mut st.cap_resp_truncated, data, cap);
}

/// Append `data` to `dst`, stopping at `cap` bytes (0 = uncapped) and setting
/// `*truncated` when bytes are dropped.
fn append_capped(dst: &mut Vec<u8>, truncated: &mut bool, data: &[u8], cap: usize) {
    if cap == 0 {
        dst.extend_from_slice(data);
        return;
    }
    if dst.len() >= cap {
        *truncated = true;
        return;
    }
    let take = (cap - dst.len()).min(data.len());
    dst.extend_from_slice(&data[..take]);
    if take < data.len() {
        *truncated = true;
    }
}

/// Build a call-history `CapturedPair` from the completed request's captured
/// bytes. `None` when capture was inactive, the request never started, or
/// nothing was captured. The request is redacted (credentials masked) unless
/// the operator opted out via `CLEARML_SNUG_CALL_HISTORY_REDACT=0`. `chat_id` is
/// the task's chat ordinal, shared from the `RequestCompleted` event so the
/// chat-counter is assigned exactly once per request.
fn build_captured_pair(
    st: &ConnectionState,
    conn_id: u64,
    chat_id: Option<String>,
) -> Option<CapturedPair> {
    if !st.cap_active || !st.request_started_emitted {
        return None;
    }
    if st.cap_req.is_empty() && st.cap_resp.is_empty() {
        return None;
    }
    let request = if control::call_history_redact() {
        call_history::redact_request(&st.cap_req)
    } else {
        st.cap_req.clone()
    };
    Some(CapturedPair {
        conn_id,
        ts_ms: Event::now_ts_ms(),
        seq: 0,
        host: st.host.clone().unwrap_or_default(),
        path: st.path.clone().unwrap_or_default(),
        method: st.method.clone().unwrap_or_default(),
        status: call_history::status_from_head(&st.cap_resp),
        request,
        response: st.cap_resp.clone(),
        request_truncated: st.cap_req_truncated,
        response_truncated: st.cap_resp_truncated,
        response_compressed: call_history::response_is_compressed(&st.cap_resp),
        chat_id,
    })
}

/// Hand the completed request's full request/response pair to the call-history
/// subsystem: buffer it (Collect/Dump) or emit it live (Continuous). Off => no-op.
fn capture_call_history(
    st: &ConnectionState,
    conn_id: u64,
    chat_id: Option<String>,
    sink: &mut dyn FnMut(Event),
) {
    let mode = control::call_history_mode();
    if mode == CallHistoryMode::Off {
        return;
    }
    let mut pair = match build_captured_pair(st, conn_id, chat_id) {
        Some(p) => p,
        None => return,
    };
    if mode == CallHistoryMode::Continuous {
        pair.seq = call_history::next_seq();
        sink(call_history::pair_to_event(&pair));
    } else {
        // Collect or Dump: buffer. A `Dump`'s one-shot flush is driven by the
        // reporter poll thread (`call_history::dump_now`), not here.
        call_history::push(pair);
    }
}

/// The single completion point for a request: emit its `RequestCompleted` (when
/// not already emitted) and, in the same step, capture the full request/response
/// pair for the call-history feature. Tying the two together guarantees the pair
/// is captured exactly once, aligned with the metering event — whether
/// completion fires early (chunked-response end), at the next keep-alive
/// boundary, on `SSL_free`, or in the exit drain.
fn complete_request(st: &ConnectionState, conn_id: u64, sink: &mut dyn FnMut(Event)) {
    if let Some(e) = build_request_completed(st, conn_id) {
        if h2_dbg() {
            if let Event::RequestCompleted {
                status, bytes_rx, tokens_in, tokens_out, tokens_measured, model, ..
            } = &e
            {
                snug_log!(
                    "[snug-h2dbg] emit RequestCompleted conn={conn_id} status={status:?} \
                     bytes_rx={bytes_rx} tokens_in={tokens_in} tokens_out={tokens_out} \
                     measured={tokens_measured} model={model:?}"
                );
            }
        }
        // Share the chat ordinal the RequestCompleted already computed (so the
        // chat counter is assigned exactly once) with the call-history capture.
        let chat_id = match &e {
            Event::RequestCompleted { chat_id, .. } => chat_id.clone(),
            _ => None,
        };
        sink(e);
        capture_call_history(st, conn_id, chat_id, sink);
    }
    // HTTP/2: on connection teardown (SSL_free / exit drain) emit a
    // RequestCompleted for any stream still open (its response never reached
    // END_STREAM before the socket closed), so a last in-flight h2 request isn't
    // lost. Streams that completed normally were already removed in
    // `process_h2_rx`.
    if let Some(h2) = &st.h2 {
        for stream in h2.streams.values() {
            if let Some(e) = build_request_completed(stream, conn_id) {
                sink(e);
            }
        }
    }
}

fn build_request_completed(st: &ConnectionState, conn_id: u64) -> Option<Event> {
    if !st.request_started_emitted {
        return None;
    }
    // Already emitted early (on chunked-response completion); the deferred sites
    // (next-request boundary / SSL_free / exit drain) must not re-emit it.
    if st.completed_emitted {
        return None;
    }
    // Prefer provider-reported usage parsed from the response body; fall back
    // to the byte-ratio estimate per direction when it's unavailable. The same
    // finalize() pass yields the tool-call count parsed from the response.
    let (
        measured_in,
        measured_out,
        cache_read,
        cache_write,
        status,
        tool_calls,
        tool_call_names,
        measured_model,
    ) = match &st.resp {
        Some(r) => {
            let f = r.finalize();
            (
                f.tokens_in,
                f.tokens_out,
                f.cache_read_tokens,
                f.cache_write_tokens,
                r.status,
                f.tool_calls,
                f.tool_call_names,
                f.model,
            )
        }
        None => (None, None, None, None, None, 0, Vec::new(), None),
    };
    let tokens_measured = measured_in.is_some() || measured_out.is_some();
    // The captured request body (when present) feeds three derived fields: the
    // freshest-turn tool errors, the conversation hash, and the request-named
    // model. Resolve the provider once and share it.
    let req_provider = if st.parse_req_body && !st.req_body.is_empty() {
        st.host.as_deref().and_then(body_scan::provider_for_host)
    } else {
        None
    };
    // Decompress the captured request body when the client sent it gzip/zstd.
    // The `Accept-Encoding: identity` rewrite (inject.rs) only forces UNCOMPRESSED
    // RESPONSES and only on the HTTP/1 path; a client is free to gzip/zstd its
    // REQUEST body, and over h2 the request's Content-Encoding is HPACK-hidden and
    // never rewritten — so a client that gzip/zstd-compresses its request body
    // hands the scanner compressed request bytes, from which the model /
    // conversation hash / tool errors can't be parsed. Detect from the body's own
    // magic bytes and inflate before parsing; an identity body borrows the raw
    // buffer unchanged (no behavior change for the plain-JSON developer path).
    let req_body: std::borrow::Cow<[u8]> = match crate::decompress::detect(&st.req_body) {
        enc if enc.is_compressed() => {
            match crate::decompress::decompress(enc, &st.req_body, REQ_DECOMP_CAP) {
                Some(plain) => std::borrow::Cow::Owned(plain),
                None => std::borrow::Cow::Borrowed(st.req_body.as_slice()),
            }
        }
        _ => std::borrow::Cow::Borrowed(st.req_body.as_slice()),
    };
    // Freshest-turn tool errors attributed to their tool names. The resent
    // history carries the `tool_use` that named each errored result, so this is
    // a single-body parse; the aggregate count is the list length.
    let tool_call_error_names = match req_provider {
        Some(p) => body_scan::tool_error_names_in_request(p, &req_body),
        None => Vec::new(),
    };
    let tool_call_errors = tool_call_error_names.len() as u64;
    // Stable per-conversation id: fingerprint the request and match it to the
    // chat it continues across trimmed/edited histories — keys the per-chat
    // scalar series in the reporter.
    let chat_id = req_provider
        .and_then(|p| body_scan::conversation_fingerprint(p, &req_body))
        .map(|fp| crate::session::assign_chat_id(&fp));
    // The model this request used, for per-model usage attribution — the coset
    // the usage aggregator groups on, not just the provider. Prefer the model the
    // provider echoed in its response (the resolved/served model, parsed from the
    // same body as the measured usage — works for Anthropic, OpenAI and Gemini);
    // fall back to the model the request asked for (body for Anthropic/OpenAI,
    // URL path for Gemini) so error responses and usage-less streams still
    // attribute a model.
    let model = measured_model.or_else(|| {
        req_provider
            .and_then(|p| body_scan::model_from_request(p, &req_body, st.path.as_deref()))
    });
    Some(Event::RequestCompleted {
        conn_id,
        ts_ms: Event::now_ts_ms(),
        status,
        latency_ms: st.started_at.elapsed().as_millis() as u64,
        bytes_tx: st.bytes_tx,
        bytes_rx: st.bytes_rx,
        tokens_in: measured_in.unwrap_or(st.tokens_in),
        tokens_out: measured_out.unwrap_or(st.tokens_out),
        tokens_measured,
        // Anthropic-only cache breakdown; 0 when unmeasured (non-Anthropic, or a
        // byte-estimate fallback) so only real cache usage plots a nonzero point.
        cache_read_tokens: cache_read.unwrap_or(0),
        cache_write_tokens: cache_write.unwrap_or(0),
        tool_calls,
        tool_call_errors,
        tool_call_names,
        tool_call_error_names,
        chat_id,
        model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whitelist::{Whitelist, WhitelistRule};

    // Test helpers for building WhitelistRules.
    fn rule(host: &str, prefix: &str, inject: bool) -> WhitelistRule {
        rule_with_tokenizer(host, prefix, inject, "approx")
    }

    fn rule_with_tokenizer(
        host: &str,
        prefix: &str,
        inject: bool,
        tokenizer: &str,
    ) -> WhitelistRule {
        WhitelistRule {
            host: host.into(),
            path_prefix: prefix.into(),
            inject_headers: inject,
            tokenizer: tokenizer.into(),
            estimate_unmeasured: false,
            completion_path: String::new(),
            provider: String::new(),
        }
    }

    fn empty_wl() -> Whitelist {
        Whitelist::empty()
    }

    fn collect_write(
        st: &mut ConnectionState,
        conn_id: u64,
        buf: &[u8],
        wl: &Whitelist,
        project_id: &str,
        task_id: &str,
    ) -> (Vec<Event>, Option<Vec<u8>>) {
        // No self-hosts configured for the bulk of tests (the common case).
        collect_write_self(st, conn_id, buf, wl, &[], project_id, task_id)
    }

    fn collect_write_self(
        st: &mut ConnectionState,
        conn_id: u64,
        buf: &[u8],
        wl: &Whitelist,
        self_hosts: &[String],
        project_id: &str,
        task_id: &str,
    ) -> (Vec<Event>, Option<Vec<u8>>) {
        let mut events = Vec::new();
        let spliced = observe_write_inner(
            st,
            conn_id,
            buf,
            wl,
            self_hosts,
            project_id,
            task_id,
            &mut |e| events.push(e),
        );
        (events, spliced)
    }

    fn collect_read(st: &mut ConnectionState, conn_id: u64, buf: &[u8]) -> Vec<Event> {
        let mut v = Vec::new();
        observe_read_inner(st, conn_id, buf, &mut |e| v.push(e));
        v
    }

    #[test]
    fn http1_first_write_no_whitelist_emits_request_started_and_bytes() {
        let mut st = ConnectionState::new();
        let buf = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events, spliced) = collect_write(&mut st, 42, buf, &empty_wl(), "", "");
        assert!(spliced.is_none());
        assert_eq!(events.len(), 2);
        match &events[0] {
            Event::RequestStarted {
                host,
                whitelisted,
                inject_headers,
                ..
            } => {
                assert_eq!(host, "example.com");
                assert!(!*whitelisted);
                assert!(!*inject_headers);
            }
            other => panic!("expected RequestStarted, got {:?}", other),
        }
        assert!(matches!(
            events[1],
            Event::BytesObserved {
                direction: Direction::Tx,
                ..
            }
        ));
    }

    #[test]
    fn self_host_request_is_fully_suppressed() {
        // A request to the task's own ClearML backend. default_action defaults
        // to "meter", so WITHOUT the self-host rule this exact request would
        // emit RequestStarted + BytesObserved (see the test above). The
        // self-host exclusion must silence it entirely and pin the connection.
        let mut st = ConnectionState::new();
        let buf = b"POST /api/v2.13/events.add_batch HTTP/1.1\r\nHost: api.clear.ml\r\n\r\n{}";
        let hosts = vec!["api.clear.ml".to_string()];
        let (events, spliced) = collect_write_self(&mut st, 7, buf, &empty_wl(), &hosts, "", "");
        assert!(spliced.is_none(), "self-host writes are never spliced");
        assert!(
            events.is_empty(),
            "self-host write must emit nothing, got {:?}",
            events
        );
        assert!(st.suppress, "connection must be pinned suppressed");
        assert!(
            !st.request_started_emitted,
            "no RequestStarted => build_request_completed yields None on drain/free"
        );

        // The response read on the now-suppressed connection also stays silent.
        let reads = collect_read(&mut st, 7, b"HTTP/1.1 200 OK\r\n\r\n{}");
        assert!(reads.is_empty(), "suppressed read must emit nothing");
    }

    #[test]
    fn self_host_match_is_port_insensitive() {
        // Self-hosted ClearML often lives at host:port (api 8008, files 8081,
        // web 8080); the Host header carries the port but the rule is the bare
        // host.
        let mut st = ConnectionState::new();
        let buf = b"POST /api/v2.13/tasks.ping HTTP/1.1\r\nHost: localhost:8008\r\n\r\n";
        let hosts = vec!["localhost".to_string()];
        let (events, _) = collect_write_self(&mut st, 1, buf, &empty_wl(), &hosts, "", "");
        assert!(events.is_empty());
        assert!(st.suppress);
    }

    #[test]
    fn self_host_exclusion_overrides_a_whitelist_match() {
        // Even if an operator (mis)configured a whitelist rule for the backend
        // host, self-host exclusion wins — we never bill our own backend.
        let mut st = ConnectionState::new();
        let buf = b"POST /api/v2.13/events.add_batch HTTP/1.1\r\nHost: api.clear.ml\r\n\r\n";
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("api.clear.ml", "/", true)],
        };
        let hosts = vec!["api.clear.ml".to_string()];
        let (events, spliced) = collect_write_self(&mut st, 3, buf, &wl, &hosts, "proj", "task");
        assert!(spliced.is_none(), "no header injection on an excluded self-host");
        assert!(events.is_empty());
        assert!(st.suppress);
    }

    #[test]
    fn non_self_host_still_metered_when_self_hosts_present() {
        // A genuine LLM call is unaffected by the presence of a self-host list.
        let mut st = ConnectionState::new();
        let buf = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n";
        let hosts = vec!["api.clear.ml".to_string()];
        let (events, _) = collect_write_self(&mut st, 2, buf, &empty_wl(), &hosts, "", "");
        assert_eq!(events.len(), 2, "RequestStarted + BytesObserved");
        assert!(!st.suppress);
        assert!(matches!(events[0], Event::RequestStarted { .. }));
    }

    #[test]
    fn http1_with_whitelist_match_but_no_inject() {
        let mut st = ConnectionState::new();
        let buf = b"GET /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n";
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("api.anthropic.com", "/v1/", false)],
        };
        let (events, spliced) = collect_write(&mut st, 1, buf, &wl, "P", "T");
        assert!(spliced.is_none());
        match &events[0] {
            Event::RequestStarted {
                whitelisted,
                inject_headers,
                ..
            } => {
                assert!(*whitelisted);
                assert!(!*inject_headers);
            }
            other => panic!("expected RequestStarted, got {:?}", other),
        }
    }

    #[test]
    fn http1_with_inject_rule_returns_spliced_buffer_and_marks_event() {
        let mut st = ConnectionState::new();
        let buf =
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody";
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("api.anthropic.com", "/v1/", true)],
        };
        let (events, spliced) = collect_write(&mut st, 1, buf, &wl, "proj-X", "task-Y");
        let spliced = spliced.expect("should have produced a spliced buffer");
        let s = std::str::from_utf8(&spliced).unwrap();
        assert!(s.contains("project: proj-X\r\n"));
        assert!(s.contains("session: task-Y\r\n"));
        assert!(s.ends_with("\r\n\r\nbody"));

        match &events[0] {
            Event::RequestStarted {
                whitelisted,
                inject_headers,
                host,
                ..
            } => {
                assert!(*whitelisted);
                assert!(*inject_headers);
                assert_eq!(host, "api.anthropic.com");
            }
            other => panic!("expected RequestStarted, got {:?}", other),
        }
    }

    #[test]
    fn second_write_does_not_reparse_or_re_inject() {
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("example.com", "/", true)],
        };
        let header_buf = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (_, first_spliced) = collect_write(&mut st, 1, header_buf, &wl, "P", "T");
        assert!(first_spliced.is_some());

        let body_buf = b"some body bytes";
        let (events, second_spliced) = collect_write(&mut st, 1, body_buf, &wl, "P", "T");
        assert!(second_spliced.is_none());
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::BytesObserved { .. }));
        assert_eq!(st.bytes_tx, (header_buf.len() + body_buf.len()) as u64);
    }

    #[test]
    fn http2_preface_inits_demux_no_diagnostic_no_inject() {
        let mut st = ConnectionState::new();
        let preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x00";
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("doesnt.matter", "/", true)],
        };
        let (events, spliced) = collect_write(&mut st, 7, preface, &wl, "P", "T");
        assert!(spliced.is_none()); // never inject on h2
        // h2 is now metered, not dead-ended: no `http2_unsupported` diagnostic,
        // just the connection-level byte accounting.
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::BytesObserved { .. }));
        assert!(st.http2_detected);
        assert!(st.h2.is_some()); // demux stood up
    }

    #[test]
    fn non_http_first_write_emits_only_bytes_observed() {
        let mut st = ConnectionState::new();
        let (events, spliced) = collect_write(
            &mut st,
            1,
            b"\x00\x01binary garbage",
            &empty_wl(),
            "",
            "",
        );
        assert!(spliced.is_none());
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::BytesObserved { .. }));
        assert!(!st.request_started_emitted);

        // The shim parses on EVERY write, not just the first. Confirm a
        // second non-HTTP write also emits only
        // BytesObserved (no spurious RequestStarted) - i.e. the
        // parser correctly continues to return NotHttp and we don't
        // somehow start treating subsequent writes as request lines.
        let (events2, spliced2) = collect_write(
            &mut st,
            1,
            b"more binary \xff garbage",
            &empty_wl(),
            "",
            "",
        );
        assert!(spliced2.is_none());
        assert_eq!(events2.len(), 1);
        assert!(matches!(events2[0], Event::BytesObserved { .. }));
        assert!(!st.request_started_emitted);
    }

    #[test]
    fn observe_read_increments_rx_and_emits_bytes_observed() {
        let mut st = ConnectionState::new();
        let buf = vec![0u8; 1234];
        let events = collect_read(&mut st, 1, &buf);
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::BytesObserved {
                direction, bytes, ..
            } => {
                assert!(matches!(direction, Direction::Rx));
                assert_eq!(*bytes, 1234);
            }
            other => panic!("expected BytesObserved Rx, got {:?}", other),
        }
        assert_eq!(st.bytes_rx, 1234);
    }

    #[test]
    fn build_request_completed_only_when_request_started_emitted() {
        let mut st = ConnectionState::new();
        assert!(build_request_completed(&st, 1).is_none());

        let _ = collect_write(
            &mut st,
            1,
            b"GET / HTTP/1.1\r\nHost: x\r\n\r\n",
            &empty_wl(),
            "",
            "",
        );
        let _ = collect_read(&mut st, 1, &vec![0u8; 500]);
        let c = build_request_completed(&st, 1).expect("should produce Completed");
        match c {
            Event::RequestCompleted {
                bytes_tx, bytes_rx, ..
            } => {
                assert_eq!(bytes_rx, 500);
                assert!(bytes_tx > 0);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn flush_all_pending_emits_request_completed_for_started_requests_only() {
        // Set up a map with three entries representing three different
        // states a connection can be in at process-exit time:
        //   1: started a request and saw response bytes (the common case
        //      we need to flush - "the last keep-alive request").
        //   2: never wrote anything yet (fresh SSL* with no traffic).
        //   3: started a request but no response bytes yet (request
        //      in-flight at exit time - still has tokens we want).
        let mut map: HashMap<usize, ConnectionState> = HashMap::new();

        let mut st1 = ConnectionState::new();
        let _ = collect_write(
            &mut st1,
            1,
            b"GET / HTTP/1.1\r\nHost: x\r\n\r\n",
            &empty_wl(),
            "",
            "",
        );
        let _ = collect_read(&mut st1, 1, &vec![0u8; 500]);
        map.insert(1usize, st1);

        map.insert(2usize, ConnectionState::new());

        let mut st3 = ConnectionState::new();
        let _ = collect_write(
            &mut st3,
            3,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody",
            &empty_wl(),
            "",
            "",
        );
        map.insert(3usize, st3);

        let mut emitted: Vec<Event> = Vec::new();
        flush_all_pending_inner(&mut map, &mut |e| emitted.push(e));

        // The map must be drained regardless of which entries produced
        // events - subsequent observe_free hooks on these conn_ids
        // should find nothing and become no-ops.
        assert!(map.is_empty(), "flush_all_pending_inner must drain the map");

        // ssl=2 had no RequestStarted_emitted so it should NOT produce
        // an event. ssl=1 and ssl=3 should both produce RequestCompleted.
        assert_eq!(
            emitted.len(),
            2,
            "expected RequestCompleted for ssl=1 and ssl=3 only, got {:?}",
            emitted
        );

        let mut seen_conn_ids: Vec<u64> = emitted
            .iter()
            .map(|e| match e {
                Event::RequestCompleted { conn_id, .. } => *conn_id,
                other => panic!("expected RequestCompleted, got {:?}", other),
            })
            .collect();
        seen_conn_ids.sort();
        assert_eq!(seen_conn_ids, vec![1u64, 3u64]);

        // ssl=1's RequestCompleted should carry the 500 bytes we read.
        let r1 = emitted
            .iter()
            .find(|e| matches!(e, Event::RequestCompleted { conn_id: 1, .. }))
            .expect("RequestCompleted for conn_id=1 missing");
        if let Event::RequestCompleted { bytes_rx, .. } = r1 {
            assert_eq!(*bytes_rx, 500, "byte counts must survive the flush");
        }
    }

    #[test]
    fn flush_all_pending_is_safe_on_empty_map() {
        let mut map: HashMap<usize, ConnectionState> = HashMap::new();
        let mut emitted: Vec<Event> = Vec::new();
        flush_all_pending_inner(&mut map, &mut |e| emitted.push(e));
        assert!(emitted.is_empty(), "empty map should produce no events");
        assert!(map.is_empty());
    }

    #[test]
    fn flush_all_pending_is_idempotent() {
        // Second call after the first drain must be a clean no-op
        // (no panics, no double-emit, no leftover state).
        let mut map: HashMap<usize, ConnectionState> = HashMap::new();
        let mut st = ConnectionState::new();
        let _ = collect_write(
            &mut st,
            7,
            b"GET / HTTP/1.1\r\nHost: x\r\n\r\n",
            &empty_wl(),
            "",
            "",
        );
        map.insert(7usize, st);

        let mut first: Vec<Event> = Vec::new();
        flush_all_pending_inner(&mut map, &mut |e| first.push(e));
        assert_eq!(first.len(), 1);

        let mut second: Vec<Event> = Vec::new();
        flush_all_pending_inner(&mut map, &mut |e| second.push(e));
        assert!(
            second.is_empty(),
            "second flush must be a no-op, got {:?}",
            second
        );
    }

    // --- Call-history capture ------------------------------------------

    fn anthropic_wl() -> Whitelist {
        Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("api.anthropic.com", "/v1/", false)],
        }
    }

    #[test]
    fn call_history_off_captures_nothing() {
        let _g = crate::control::MODE_TEST_LOCK.lock().unwrap();
        control::set_call_history_mode(CallHistoryMode::Off);
        let mut st = ConnectionState::new();
        let req = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n{\"m\":1}";
        let _ = collect_write(&mut st, 1, req, &anthropic_wl(), "", "");
        let _ = collect_read(&mut st, 1, b"HTTP/1.1 200 OK\r\n\r\n{\"ok\":1}");
        assert!(!st.cap_active, "cap_active must stay false in Off mode");
        assert!(st.cap_req.is_empty() && st.cap_resp.is_empty());
        // Completion emits a RequestCompleted but no CallHistoryEntry.
        let mut events = Vec::new();
        complete_request(&st, 1, &mut |e| events.push(e));
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::CallHistoryEntry { .. })));
    }

    #[test]
    fn call_history_captures_full_request_and_response() {
        let _g = crate::control::MODE_TEST_LOCK.lock().unwrap();
        control::set_call_history_mode(CallHistoryMode::Continuous);
        let mut st = ConnectionState::new();
        let req = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n{\"model\":\"x\"}";
        let _ = collect_write(&mut st, 1, req, &anthropic_wl(), "", "");
        assert!(st.cap_active, "whitelisted host + non-Off mode => capture");
        // Full request includes the request LINE, not just the body.
        assert!(st.cap_req.starts_with(b"POST /v1/messages HTTP/1.1\r\n"));
        assert!(st.cap_req.ends_with(b"{\"model\":\"x\"}"));

        let resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}";
        let _ = collect_read(&mut st, 1, resp);
        assert!(st.cap_resp.starts_with(b"HTTP/1.1 200 OK\r\n"));

        // Continuous mode emits CallHistoryEntry events at completion.
        let mut events = Vec::new();
        complete_request(&st, 1, &mut |e| events.push(e));
        control::set_call_history_mode(CallHistoryMode::Off);

        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut req_bytes = Vec::new();
        let mut resp_bytes = Vec::new();
        let mut saw_entry = false;
        for e in &events {
            if let Event::CallHistoryEntry {
                request_b64,
                response_b64,
                ..
            } = e
            {
                saw_entry = true;
                if !request_b64.is_empty() {
                    req_bytes.extend(b64.decode(request_b64).unwrap());
                }
                if !response_b64.is_empty() {
                    resp_bytes.extend(b64.decode(response_b64).unwrap());
                }
            }
        }
        assert!(saw_entry, "Continuous mode must emit a CallHistoryEntry");
        assert!(req_bytes.starts_with(b"POST /v1/messages HTTP/1.1\r\n"));
        assert!(resp_bytes.starts_with(b"HTTP/1.1 200 OK\r\n"));
    }

    #[test]
    fn call_history_redacts_authorization_in_emitted_request() {
        let _g = crate::control::MODE_TEST_LOCK.lock().unwrap();
        control::set_call_history_mode(CallHistoryMode::Continuous);
        let mut st = ConnectionState::new();
        let req = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\nx-api-key: sk-secret\r\n\r\n{}";
        let _ = collect_write(&mut st, 1, req, &anthropic_wl(), "", "");
        let mut events = Vec::new();
        complete_request(&st, 1, &mut |e| events.push(e));
        control::set_call_history_mode(CallHistoryMode::Off);

        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let joined: Vec<u8> = events
            .iter()
            .filter_map(|e| match e {
                Event::CallHistoryEntry { request_b64, .. } if !request_b64.is_empty() => {
                    Some(b64.decode(request_b64).unwrap())
                }
                _ => None,
            })
            .flatten()
            .collect();
        let s = String::from_utf8(joined).unwrap();
        assert!(s.contains("x-api-key: <redacted>"), "got: {s}");
        assert!(!s.contains("sk-secret"));
    }

    #[test]
    fn call_history_resets_capture_per_keepalive_request() {
        let _g = crate::control::MODE_TEST_LOCK.lock().unwrap();
        control::set_call_history_mode(CallHistoryMode::Collect);
        let mut st = ConnectionState::new();
        let req1 = b"POST /v1/a HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nAAA";
        let _ = collect_write(&mut st, 1, req1, &anthropic_wl(), "", "");
        let _ = collect_read(&mut st, 1, b"HTTP/1.1 200 OK\r\n\r\nresp-a");
        // Second request on the same connection resets the capture buffers.
        let req2 = b"POST /v1/b HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nBBB";
        let _ = collect_write(&mut st, 1, req2, &anthropic_wl(), "", "");
        control::set_call_history_mode(CallHistoryMode::Off);
        assert!(st.cap_req.windows(3).any(|w| w == b"BBB"));
        assert!(!st.cap_req.windows(3).any(|w| w == b"AAA"));
        // req2 hasn't read a response yet, so its response capture is empty.
        assert!(st.cap_resp.is_empty());
    }

    #[test]
    fn append_capped_truncates_at_cap() {
        let mut buf = Vec::new();
        let mut truncated = false;
        append_capped(&mut buf, &mut truncated, b"hello", 3);
        assert_eq!(buf, b"hel");
        assert!(truncated);
        // Further appends past the cap are dropped, truncation stays set.
        append_capped(&mut buf, &mut truncated, b"world", 3);
        assert_eq!(buf, b"hel");
        // cap == 0 means uncapped.
        let mut u = Vec::new();
        let mut t = false;
        append_capped(&mut u, &mut t, b"unbounded", 0);
        assert_eq!(u, b"unbounded");
        assert!(!t);
    }

    // --- Token estimation ----------------------------------

    fn bytes_observed_tokens(events: &[Event]) -> Option<u64> {
        events.iter().find_map(|e| {
            if let Event::BytesObserved { tokens_est, .. } = e {
                Some(*tokens_est)
            } else {
                None
            }
        })
    }

    #[test]
    fn tokens_est_populated_when_matched_rule_has_tokenizer() {
        let mut st = ConnectionState::new();
        let buf =
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nthe body";
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule_with_tokenizer(
                "api.anthropic.com",
                "/v1/",
                false,
                "claude",
            )],
        };
        let (events, _) = collect_write(&mut st, 1, buf, &wl, "", "");
        let est = bytes_observed_tokens(&events).expect("BytesObserved emitted");
        assert!(est > 0, "tokens_est should be > 0 when tokenizer is set");
        assert_eq!(st.tokenizer.as_deref(), Some("claude"));
    }

    #[test]
    fn tokens_est_uses_default_tokenizer_when_no_rule_matches() {
        // When default_action="meter" (the default) and no rule
        // matches, the shim falls back to the env-configured
        // default tokenizer so EVERY metered connection gets non-zero
        // tokens_est. Without an env var set, the default is "approx".
        let mut st = ConnectionState::new();
        let buf = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events, _) = collect_write(&mut st, 1, buf, &empty_wl(), "", "");
        let est = bytes_observed_tokens(&events).expect("BytesObserved emitted");
        assert!(est > 0, "tokens_est should fall back to default tokenizer");
        // Whatever the default is, st.tokenizer must now be set so
        // subsequent body chunks get consistent estimates.
        assert!(st.tokenizer.is_some());
    }

    #[test]
    fn default_action_ignore_suppresses_all_events_for_unmatched_hosts() {
        // When the whitelist's default_action is "ignore", unmatched
        // HTTP/1.x connections produce zero events (not even
        // BytesObserved). Operator opt-out for unknown hosts.
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "ignore".into(),
            rules: vec![],
        };
        let buf = b"GET /something HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events, spliced) = collect_write(&mut st, 1, buf, &wl, "", "");
        assert!(spliced.is_none());
        assert_eq!(events.len(), 0, "expected zero events, got: {:?}", events);
        assert!(st.suppress);
        assert!(!st.request_started_emitted);

        // Subsequent observe_read on the same connection also emits
        // nothing.
        let events = collect_read(&mut st, 1, &vec![0u8; 500]);
        assert_eq!(events.len(), 0);
        // And RequestCompleted is suppressed too (it relied on
        // request_started_emitted which never flipped).
        assert!(build_request_completed(&st, 1).is_none());
    }

    #[test]
    fn default_action_ignore_still_emits_for_whitelisted_hosts() {
        // With default_action="ignore", a connection that MATCHES a
        // rule still gets the normal event stream. Only unmatched
        // hosts are suppressed.
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "ignore".into(),
            rules: vec![rule_with_tokenizer(
                "api.anthropic.com",
                "/v1/",
                false,
                "claude",
            )],
        };
        let buf =
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody";
        let (events, _) = collect_write(&mut st, 1, buf, &wl, "", "");
        assert!(events.len() >= 2);
        assert!(matches!(events[0], Event::RequestStarted { .. }));
        assert!(!st.suppress);
    }

    #[test]
    fn default_action_ignore_suppresses_http2_and_non_http_too() {
        // For protocols we can't host-match against, "ignore" still
        // applies - we never explicitly whitelisted them, so they don't
        // count as opted in.
        let wl = Whitelist {
            version: 1,
            default_action: "ignore".into(),
            rules: vec![],
        };

        // HTTP/2 preface.
        let mut st = ConnectionState::new();
        let preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x00";
        let (events, _) = collect_write(&mut st, 1, preface, &wl, "", "");
        assert_eq!(events.len(), 0);
        assert!(st.suppress);

        // Non-HTTP binary garbage.
        let mut st2 = ConnectionState::new();
        let (events2, _) = collect_write(&mut st2, 1, b"\x00\x01garbage", &wl, "", "");
        assert_eq!(events2.len(), 0);
        assert!(st2.suppress);
    }

    #[test]
    fn request_completed_aggregates_tokens_per_direction() {
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule_with_tokenizer(
                "api.anthropic.com",
                "/v1/",
                false,
                "claude",
            )],
        };
        let _ = collect_write(
            &mut st,
            1,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody payload",
            &wl,
            "",
            "",
        );
        // Simulate a few read chunks of the response.
        let _ = collect_read(&mut st, 1, &vec![0u8; 350]);
        let _ = collect_read(&mut st, 1, &vec![0u8; 1050]);
        let completed = build_request_completed(&st, 1).expect("RequestCompleted");
        match completed {
            Event::RequestCompleted {
                tokens_in,
                tokens_out,
                ..
            } => {
                assert!(tokens_in > 0, "tokens_in should accumulate from writes");
                assert!(tokens_out > 0, "tokens_out should accumulate from reads");
                // Sanity: rx had ~1400 bytes, claude ratio 3.5 -> ~400 tokens.
                assert!(tokens_out >= 350 && tokens_out <= 500);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    // --- Keep-alive boundary detection ---------------------------------

    #[test]
    fn two_http1_requests_on_one_connection_emit_per_request_lifecycle() {
        // HTTP/1.1 keep-alive: two requests on one SSL*. Both should
        // emit RequestStarted; the first should also emit
        // RequestCompleted when the second arrives (we close the prior
        // request before opening the new one).
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![],
        };

        let req1 = b"GET /a HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events1, _) = collect_write(&mut st, 7, req1, &wl, "", "");
        assert!(matches!(events1[0], Event::RequestStarted { .. }));
        assert!(matches!(events1[1], Event::BytesObserved { .. }));
        assert_eq!(events1.len(), 2);

        let req2 = b"GET /b HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events2, _) = collect_write(&mut st, 7, req2, &wl, "", "");
        // Expect: RequestCompleted for req1, then RequestStarted for
        // req2, then BytesObserved for req2.
        assert_eq!(events2.len(), 3, "got: {:?}", events2);
        match &events2[0] {
            Event::RequestCompleted { bytes_tx, .. } => {
                // req1 had only its header write counted.
                assert_eq!(*bytes_tx, req1.len() as u64);
            }
            other => panic!("expected RequestCompleted for req1, got {:?}", other),
        }
        match &events2[1] {
            Event::RequestStarted { path, .. } => assert_eq!(path, "/b"),
            other => panic!("expected RequestStarted for req2, got {:?}", other),
        }
        assert!(matches!(events2[2], Event::BytesObserved { .. }));

        // Per-request state should reflect req2 only.
        assert_eq!(st.path.as_deref(), Some("/b"));
        assert_eq!(st.bytes_tx, req2.len() as u64);
    }

    #[test]
    fn body_with_method_like_prefix_doesnt_misfire() {
        // A request body that happens to contain ASCII text resembling
        // an HTTP method must not be parsed as a new request boundary.
        // httparse needs a Complete request (request-line + headers
        // + \r\n\r\n) to call it a request, so loose method-shaped
        // bytes shouldn't trigger anything.
        let mut st = ConnectionState::new();
        let wl = empty_wl();
        let req = b"POST /upload HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let _ = collect_write(&mut st, 1, req, &wl, "", "");

        // Body chunk that LOOKS request-like but isn't a complete
        // valid request - no Host header, no trailing \r\n\r\n.
        let body = b"GET /something HTTP/1.1 fake body content";
        let (events, spliced) = collect_write(&mut st, 1, body, &wl, "", "");
        assert!(spliced.is_none());
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::BytesObserved { .. }));
        // Still on req1's identity.
        assert_eq!(st.path.as_deref(), Some("/upload"));
    }

    #[test]
    fn inject_applies_to_each_keep_alive_request_separately() {
        // With an inject rule, both REQ1 and REQ2 on the same
        // connection should each produce their own spliced buffer
        // carrying project:/session: headers - not just the first.
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule("api.anthropic.com", "/v1/", true)],
        };

        let req1 = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody1";
        let (_, spliced1) = collect_write(&mut st, 9, req1, &wl, "proj", "task-A");
        let s1 = String::from_utf8(spliced1.expect("req1 should be spliced")).unwrap();
        assert!(s1.contains("project: proj\r\n"));
        assert!(s1.contains("session: task-A\r\n"));

        let req2 = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody2";
        let (_, spliced2) = collect_write(&mut st, 9, req2, &wl, "proj", "task-A");
        let s2 = String::from_utf8(spliced2.expect("req2 should also be spliced")).unwrap();
        assert!(s2.contains("project: proj\r\n"));
        assert!(s2.contains("session: task-A\r\n"));
    }

    #[test]
    fn keep_alive_request_completed_bytes_attribute_to_each_request() {
        // RequestCompleted for REQ1 (fired when REQ2 arrives) should
        // include ONLY REQ1's tx bytes and whatever bytes_rx
        // accumulated between REQ1's start and REQ2's start.
        // REQ2's RequestCompleted (fired on SSL_free) should include
        // ONLY its own tx/rx.
        let mut st = ConnectionState::new();
        let wl = empty_wl();

        let req1 = b"GET /a HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let _ = collect_write(&mut st, 1, req1, &wl, "", "");
        // Some response bytes for req1.
        let _ = collect_read(&mut st, 1, &vec![0u8; 500]);

        let req2 = b"GET /b HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events_at_boundary, _) = collect_write(&mut st, 1, req2, &wl, "", "");
        // First event is RequestCompleted for req1.
        match &events_at_boundary[0] {
            Event::RequestCompleted {
                bytes_tx, bytes_rx, ..
            } => {
                assert_eq!(*bytes_tx, req1.len() as u64);
                assert_eq!(*bytes_rx, 500);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }

        // Now simulate req2's response.
        let _ = collect_read(&mut st, 1, &vec![0u8; 750]);
        let req2_completed =
            build_request_completed(&st, 1).expect("req2 should be completable");
        match req2_completed {
            Event::RequestCompleted {
                bytes_tx, bytes_rx, ..
            } => {
                // req2's tx is the headers only; rx is 750 (the 500
                // from req1's response should NOT be lumped here).
                assert_eq!(bytes_tx, req2.len() as u64);
                assert_eq!(bytes_rx, 750);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn connection_suppressed_stays_suppressed_across_pseudo_requests() {
        // Once a connection is suppressed (first write didn't match
        // any rule under default_action="ignore"), even later writes
        // that LOOK like new request lines should not flip suppression
        // back off. Suppression is a connection-level decision.
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "ignore".into(),
            rules: vec![],
        };

        let req1 = b"GET /a HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events, _) = collect_write(&mut st, 1, req1, &wl, "", "");
        assert_eq!(events.len(), 0);
        assert!(st.suppress);

        // Later write that also parses as HTTP/1: still silent.
        let req2 = b"GET /b HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events2, spliced) = collect_write(&mut st, 1, req2, &wl, "", "");
        assert_eq!(events2.len(), 0);
        assert!(spliced.is_none());
        assert!(st.suppress);
        // We didn't accidentally start tracking it.
        assert!(!st.request_started_emitted);
    }

    #[test]
    fn rearm_clears_ignore_suppression_but_not_self_host() {
        let mut map: HashMap<usize, ConnectionState> = HashMap::new();
        // ignore-suppressed connection (re-armable).
        let mut ig = ConnectionState::new();
        ig.suppress = true;
        ig.suppress_rearmable = true;
        ig.first_write_seen = true;
        map.insert(1usize, ig);
        // self-host suppressed connection (permanent).
        let mut sh = ConnectionState::new();
        sh.suppress = true;
        sh.suppress_rearmable = false;
        sh.first_write_seen = true;
        map.insert(2usize, sh);

        rearm_inner(&mut map);

        let ig = map.get(&1).unwrap();
        assert!(!ig.suppress, "ignore suppression must be re-armed (cleared)");
        assert!(!ig.first_write_seen, "first-write decision must re-run");
        let sh = map.get(&2).unwrap();
        assert!(sh.suppress, "self-host suppression must persist across re-arm");
    }

    #[test]
    fn ignore_suppressed_connection_meters_after_rearm_and_add() {
        // Reproduces the live hot-reload case on a pooled keep-alive connection:
        // a request to an unwhitelisted host under default_action="ignore" is
        // suppressed; after the host is added (whitelist swap + re-arm), the
        // NEXT request on the SAME connection is metered.
        let ignore_empty = Whitelist {
            version: 1,
            default_action: "ignore".into(),
            rules: vec![],
        };
        let mut st = ConnectionState::new();
        let req1 = b"POST /v1/chat/completions HTTP/1.1\r\nHost: api.openai.com\r\n\r\n{}";
        let (events1, _) = collect_write(&mut st, 1, req1, &ignore_empty, "", "");
        assert!(events1.is_empty(), "unwhitelisted host under ignore is suppressed");
        assert!(st.suppress && st.suppress_rearmable);

        // Operator adds api.openai.com -> swap + re-arm. Drive the per-connection
        // effect via rearm_inner (the same logic the global swap triggers).
        let mut map = HashMap::new();
        map.insert(1usize, st);
        rearm_inner(&mut map);
        let mut st = map.remove(&1).unwrap();
        assert!(!st.suppress, "re-arm clears the ignore suppression");

        let now = Whitelist {
            version: 1,
            default_action: "ignore".into(),
            rules: vec![rule("api.openai.com", "/", false)],
        };
        let req2 = b"POST /v1/chat/completions HTTP/1.1\r\nHost: api.openai.com\r\n\r\n{}";
        let (events2, _) = collect_write(&mut st, 1, req2, &now, "", "");
        assert!(
            events2
                .iter()
                .any(|e| matches!(e, Event::RequestStarted { .. })),
            "after re-arm + add, the next request must meter: {:?}",
            events2
        );
        assert!(!st.suppress, "now-whitelisted host stays un-suppressed");
    }

    #[test]
    fn http2_detected_persists_across_subsequent_writes() {
        // Once we tag a connection HTTP/2, every subsequent write must
        // NOT attempt HTTP/1 parsing. Otherwise an HPACK frame that
        // happens to start with method-looking bytes could spuriously
        // trigger a RequestStarted event.
        let mut st = ConnectionState::new();
        let wl = empty_wl();

        let preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x00";
        let (events1, _) = collect_write(&mut st, 1, preface, &wl, "", "");
        // h2 metered (no diagnostic): just connection byte accounting.
        assert_eq!(events1.len(), 1);
        assert!(matches!(events1[0], Event::BytesObserved { .. }));
        assert!(st.http2_detected);

        // Subsequent frame whose first bytes happen to look like
        // ASCII (HPACK can carry literal header values etc). MUST NOT
        // become an HTTP/1 RequestStarted.
        let frame = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (events2, _) = collect_write(&mut st, 1, frame, &wl, "", "");
        assert_eq!(events2.len(), 1, "got events: {:?}", events2);
        assert!(matches!(events2[0], Event::BytesObserved { .. }));
        assert!(!events2
            .iter()
            .any(|e| matches!(e, Event::RequestStarted { .. })));
    }

    // --- Real usage wiring into RequestCompleted -----------------------

    /// Build a single h2 frame (9-byte header + payload) for tests.
    fn h2_frame(ftype: u8, flags: u8, stream_id: u32, payload: &[u8]) -> Vec<u8> {
        let len = payload.len();
        let mut f = vec![(len >> 16) as u8, (len >> 8) as u8, len as u8, ftype, flags];
        f.extend_from_slice(&stream_id.to_be_bytes());
        f.extend_from_slice(payload);
        f
    }

    #[test]
    fn h2_stream_completes_with_measured_anthropic_usage() {
        // Drive the demux directly: the public path gates on a process-global
        // env read once, so we exercise the pure per-stream helpers here.
        let mut st = ConnectionState::new();
        st.http2_detected = true;
        st.h2 = Some(Box::new(H2State::new()));
        let mut events: Vec<Event> = Vec::new();

        // Client: preface + HEADERS(stream 1) + DATA(stream 1, request JSON, END).
        let mut tx = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
        tx.extend(h2_frame(crate::h2::FRAME_HEADERS, 0, 1, b"")); // opaque HPACK block
        tx.extend(h2_frame(
            crate::h2::FRAME_DATA,
            0x1, // END_STREAM
            1,
            br#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"hi"}]}"#,
        ));
        {
            let mut sink = |e: Event| events.push(e);
            process_h2_tx(&mut st, 7, &tx, &mut sink);
        }

        // Server: DATA(stream 1) SSE with Anthropic usage, END_STREAM.
        let sse = b"event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":736,\"output_tokens\":1}}}\r\n\r\nevent: message_delta\r\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":21}}\r\n\r\n";
        let rx = h2_frame(crate::h2::FRAME_DATA, 0x1, 1, sse);
        {
            let mut sink = |e: Event| events.push(e);
            process_h2_rx(&mut st, 7, &rx, &mut sink);
        }

        assert!(events.iter().any(|e| matches!(e, Event::RequestStarted { .. })));
        let completed = events.iter().find_map(|e| match e {
            Event::RequestCompleted { tokens_in, tokens_out, tokens_measured, model, .. } => {
                Some((*tokens_in, *tokens_out, *tokens_measured, model.clone()))
            }
            _ => None,
        });
        let (ti, to, measured, model) =
            completed.expect("RequestCompleted emitted for the completed stream");
        assert!(measured, "usage should be provider-measured, not the byte estimate");
        assert_eq!(ti, 736);
        assert_eq!(to, 21);
        assert_eq!(model.as_deref(), Some("claude-haiku-4-5"));
        // The stream is removed once completed.
        assert!(st.h2.as_ref().unwrap().streams.is_empty());
    }

    #[test]
    fn looks_like_h2_server_start_detects_settings_and_rejects_http1() {
        // A server SETTINGS frame (type 0x4, stream 0, len multiple of 6).
        let settings = h2_frame(crate::h2::FRAME_SETTINGS, 0, 0, &[0, 3, 0, 0, 0, 100]);
        assert!(looks_like_h2_server_start(&settings));
        // An HTTP/1 response head must never be misclassified as h2.
        assert!(!looks_like_h2_server_start(b"HTTP/1.1 200 OK\r\n\r\n"));
        // A DATA frame (type 0x0, stream != 0) is not the connection bootstrap.
        let data = h2_frame(crate::h2::FRAME_DATA, 0x1, 1, b"body");
        assert!(!looks_like_h2_server_start(&data));
        // Too short to even hold a 9-byte frame header.
        assert!(!looks_like_h2_server_start(&[0, 0, 0, 4]));
    }

    #[test]
    fn h2_stream_completes_from_read_side_only() {
        // Rx-only connection: only SSL_read is hooked, so the whole h2 exchange
        // must be metered from the server->client side alone. The stream is never
        // opened by the tx path; `process_h2_rx` auto-registers it on the first
        // response DATA and completes it on END_STREAM with the SSE-measured
        // usage. Two DATA frames also exercise the RequestStarted once-guard.
        let mut st = ConnectionState::new();
        st.http2_detected = true;
        st.h2 = Some(Box::new(H2State::new()));
        let mut events: Vec<Event> = Vec::new();

        // Response DATA(stream 1) split across two frames; END_STREAM on the last.
        // No tx frames at all — the stream is unknown to the demux until now.
        let sse1 = b"event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":736,\"output_tokens\":1}}}\r\n\r\n";
        let sse2 = b"event: message_delta\r\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":21}}\r\n\r\n";
        let mut rx = h2_frame(crate::h2::FRAME_DATA, 0, 1, sse1); // no END_STREAM
        rx.extend(h2_frame(crate::h2::FRAME_DATA, 0x1, 1, sse2)); // END_STREAM
        {
            let mut sink = |e: Event| events.push(e);
            process_h2_rx(&mut st, 9, &rx, &mut sink);
        }

        // Exactly one RequestStarted despite two DATA frames auto-registering.
        let started = events
            .iter()
            .filter(|e| matches!(e, Event::RequestStarted { .. }))
            .count();
        assert_eq!(started, 1, "rx-only stream emits exactly one RequestStarted");

        let completed = events.iter().find_map(|e| match e {
            Event::RequestCompleted { tokens_in, tokens_out, tokens_measured, model, .. } => {
                Some((*tokens_in, *tokens_out, *tokens_measured, model.clone()))
            }
            _ => None,
        });
        let (ti, to, measured, model) =
            completed.expect("RequestCompleted emitted from the read side alone");
        assert!(measured, "usage measured from the response SSE, not the byte estimate");
        assert_eq!(ti, 736);
        assert_eq!(to, 21);
        assert_eq!(model.as_deref(), Some("claude-haiku-4-5"));
        // The stream is removed once completed.
        assert!(st.h2.as_ref().unwrap().streams.is_empty());
    }

    #[test]
    fn request_completed_uses_measured_usage_when_scanner_has_it() {
        let mut st = ConnectionState::new();
        let _ = collect_write(
            &mut st,
            1,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody",
            &empty_wl(),
            "",
            "",
        );
        // Attach a scanner fed with a real (non-streaming) Anthropic response.
        let mut rp =
            crate::body_scan::RespParse::new(crate::body_scan::Provider::Anthropic);
        rp.feed(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"usage\":{\"input_tokens\":14,\"output_tokens\":495}}");
        st.resp = Some(rp);

        match build_request_completed(&st, 1).expect("completed") {
            Event::RequestCompleted {
                tokens_in,
                tokens_out,
                status,
                tokens_measured,
                ..
            } => {
                assert_eq!(tokens_in, 14, "measured input should override estimate");
                assert_eq!(tokens_out, 495, "measured output should override estimate");
                assert_eq!(status, Some(200));
                assert!(tokens_measured);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn completed_emitted_guards_against_double_emit() {
        // Once a request's RequestCompleted has been emitted early (on chunked
        // response completion, in observe_read_inner), every deferred site —
        // next-request boundary, SSL_free, and the exit drain — must NOT re-emit
        // it. They all funnel through build_request_completed, so the single
        // `completed_emitted` guard there covers them.
        let mut st = ConnectionState::new();
        let _ = collect_write(
            &mut st,
            1,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody",
            &empty_wl(),
            "",
            "",
        );
        // Before the early emit, a completion builds an event.
        assert!(build_request_completed(&st, 1).is_some(), "first emit allowed");
        // The early-emit path sets this; afterwards the deferred sites get None.
        st.completed_emitted = true;
        assert!(
            build_request_completed(&st, 1).is_none(),
            "no re-emit after completed_emitted"
        );
        // The exit-drain site is a no-op for an already-completed request too.
        let mut map = HashMap::new();
        map.insert(1usize, st);
        let mut emitted = Vec::new();
        flush_all_pending_inner(&mut map, &mut |e| emitted.push(e));
        assert!(
            emitted.is_empty(),
            "exit drain must not re-emit an already-completed request"
        );
    }

    #[test]
    fn request_completed_falls_back_to_estimate_without_scanner() {
        let mut st = ConnectionState::new();
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![rule_with_tokenizer(
                "api.anthropic.com",
                "/v1/",
                false,
                "claude",
            )],
        };
        let _ = collect_write(
            &mut st,
            1,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody",
            &wl,
            "",
            "",
        );
        let _ = collect_read(&mut st, 1, &vec![0u8; 1400]);
        // No scanner attached (parse-usage disabled) -> estimate + no status.
        match build_request_completed(&st, 1).expect("completed") {
            Event::RequestCompleted {
                tokens_out,
                status,
                tokens_measured,
                ..
            } => {
                assert!(tokens_out > 0, "should fall back to byte estimate");
                assert_eq!(status, None);
                assert!(!tokens_measured);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn request_completed_carries_tool_calls() {
        let mut st = ConnectionState::new();
        let _ = collect_write(
            &mut st,
            1,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\nbody",
            &empty_wl(),
            "",
            "",
        );
        let mut rp =
            crate::body_scan::RespParse::new(crate::body_scan::Provider::Anthropic);
        rp.feed(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"content\":[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"x\",\"input\":{}}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}");
        st.resp = Some(rp);

        match build_request_completed(&st, 1).expect("completed") {
            Event::RequestCompleted {
                tool_calls,
                tool_call_errors,
                tool_call_names,
                tokens_in,
                ..
            } => {
                assert_eq!(tool_calls, 1);
                assert_eq!(tool_call_errors, 0);
                assert_eq!(tool_call_names, vec!["x"]);
                assert_eq!(tokens_in, 10);
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn request_completed_counts_tool_errors_from_request_body() {
        let mut st = ConnectionState::new();
        let _ = collect_write(
            &mut st,
            1,
            b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n",
            &empty_wl(),
            "",
            "",
        );
        // The env gate is off in tests, so simulate the captured-request state
        // directly: mark it captured and hand it a freshest-turn error.
        st.parse_req_body = true;
        st.req_body = br#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"boom","is_error":true}]}]}"#.to_vec();

        match build_request_completed(&st, 1).expect("completed") {
            Event::RequestCompleted {
                tool_call_errors, ..
            } => assert_eq!(tool_call_errors, 1),
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn body_start_offset_finds_header_end() {
        let buf = b"POST / HTTP/1.1\r\nHost: x\r\n\r\nBODY";
        let o = body_start_offset(buf).expect("has terminator");
        assert_eq!(&buf[o..], b"BODY");
        assert!(body_start_offset(b"POST / HTTP/1.1\r\nHost: x").is_none());
    }

    #[test]
    fn request_derived_model_and_chat_from_compressed_body() {
        use std::io::Write;
        // A gzip-compressed Anthropic request body. Over h2 the request's
        // Content-Encoding is HPACK-hidden and never forced to identity, so the
        // captured request bytes are compressed; the model + conversation hash
        // must still be resolved by inflating before the parse.
        let plain = br#"{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}"#;
        let mut e =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(plain).unwrap();
        let gz = e.finish().unwrap();
        assert_eq!(crate::decompress::detect(&gz), crate::decompress::Encoding::Gzip);

        let mut st = ConnectionState::new();
        st.request_started_emitted = true;
        st.host = Some("api.anthropic.com".to_string());
        st.parse_req_body = true;
        st.req_body = gz;

        match build_request_completed(&st, 1).expect("completed") {
            Event::RequestCompleted { model, chat_id, .. } => {
                // measured_model is None (no response scanner), so the model must
                // come from the inflated request body.
                assert_eq!(model.as_deref(), Some("claude-sonnet-4-20250514"));
                assert!(chat_id.is_some(), "fingerprint parses from the inflated body");
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }

    #[test]
    fn request_derived_model_from_plain_body_unaffected() {
        // The identity (plain-JSON developer) path is unchanged: no inflate, model
        // resolved directly from the request body.
        let mut st = ConnectionState::new();
        st.request_started_emitted = true;
        st.host = Some("api.anthropic.com".to_string());
        st.parse_req_body = true;
        st.req_body =
            br#"{"model":"claude-opus-4-20250514","messages":[{"role":"user","content":"hi"}]}"#
                .to_vec();
        match build_request_completed(&st, 1).expect("completed") {
            Event::RequestCompleted { model, .. } => {
                assert_eq!(model.as_deref(), Some("claude-opus-4-20250514"));
            }
            other => panic!("expected RequestCompleted, got {:?}", other),
        }
    }
}
