//! Usage + task-metrics sinks: per-request reporting done inline in the
//! reporter's read loop.
//!
//! Both sinks join `RequestStarted` (which carries the host + whitelisted flag)
//! to the matching `RequestCompleted` (which carries the byte/token/latency
//! counts) by `conn_id`, gate on `whitelisted` AND on the request having a
//! resolved model (only actual LLM completions resolve one — a metered host
//! also serves non-LLM endpoints that, over HTTP/2, match the whitelist on host
//! alone; see the model gate in `usage`/`metrics`), then report against the
//! resolved provider:
//!
//!   * usage — per-request LLM usage to the backend `report_llm_usage`
//!     endpoint as a single event carrying `prompt_tokens` + `completion_tokens`
//!     (combined, not split) tagged `source="external"`, plus the model and
//!     provider. Skipped when both token counts are zero, or when the request
//!     resolved no model (a non-LLM call).
//!   * task-metrics — per-request usage scalars to this task's SCALARS tab via
//!     `events.add_batch` (`training_stats_scalar`): one series per configured
//!     field, variant = provider (+ chat ordinal when the request identified its
//!     conversation). The x-axis (`iter`) is a global enumerator that increments
//!     once per captured request, and `timestamp` carries the shim's capture
//!     wall-time — so the SCALARS Wall-Time/Relative axis (and `xaxis=iso_time`
//!     in an embedded Report) plots against real capture time, not send time.
//!     (`iter` is integer-only server-side, so wall-time can't ride it; it rides
//!     the float `timestamp`.) Tool activity is reported two ways: a SIGNAL — a
//!     line at 0 that jumps to +1 when a tool was used in a request and -1 when
//!     a tool result errored (errors dominate), both aggregate (per
//!     provider/chat) and per tool name, so failures read as downward spikes —
//!     and a CUMULATIVE chart: a monotonically-increasing running total of tool
//!     calls and of tool-call errors over the run (a calls line + an errors line
//!     per provider/chat), so the magnitudes-over-time sit alongside the
//!     at-a-glance spikes. No token filter (latency/bytes are meaningful at zero
//!     tokens). Each token field additionally gets a "(cumulative)" chart — the
//!     same per-series running-total treatment, so the per-request spikes sit
//!     alongside the totals accrued over the run.
//!
//! Best-effort throughout: a backend error is logged to stderr and dropped so a
//! reporting failure never stalls metering.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use clearml_snug_messages::Event;
use serde_json::{json, Value};

use crate::api::ClearmlClient;
use crate::log_forward::LogForwarder;

/// Per-task cap on the conn_id -> meta stash. A request always pops its own
/// entry on completion, so this only bounds memory if the shim ever emits a
/// `RequestStarted` without a matching `RequestCompleted`.
const CONN_META_HARD_CAP: usize = 1024;

/// Flush a sink buffer once its serialized payload reaches this size — the
/// size-based half of the "size OR 5s timer OR drain" trigger. Mirrors the
/// agent's `Events.max_packet_size` (clearml_agent/commands/events.py).
const MAX_PACKET_BYTES: usize = 1_048_576; // 1 MiB

/// Per-buffer hard byte cap. Retain-on-failure (a backend outage) can grow a
/// buffer across flush attempts; this bounds it. Mirrors the agent's
/// `api.http.max_req_size` (15 MB) — and because it equals the server's
/// single-request limit, every flush body is a valid one-shot POST, so no
/// newline-chunking (the agent's `send_request_batch`) is needed here.
const MAX_REQ_BYTES: usize = 15_728_640; // 15 MB

/// (field key, scalar metric title). The "requests" field is a computed
/// cumulative per-provider counter; every other field maps to the same-named
/// `RequestCompleted` count. `tool_calls`/`tool_call_errors` stay valid config
/// keys here (either enables tool metering) but are NOT emitted by the generic
/// per-field loop: they drive the tool SIGNAL series instead (see `TOOL_FIELDS`
/// / `metric_events`), so their titles below are only documentation.
const FIELD_SPECS: &[(&str, &str)] = &[
    ("tokens_in", "LLM Input Tokens"),
    ("tokens_out", "LLM Output Tokens"),
    // Anthropic prompt-cache split. These three input series are DISJOINT: "LLM
    // Input Tokens" is FRESH (non-cached) input only, and cache-read / cache-write
    // are the cached buckets, so the three sum to the billable input total. (The
    // usage event reports that combined total as prompt_tokens; only the scalars
    // split it — see metric_events.) Both cache buckets are 0 for OpenAI/Gemini.
    ("cache_read_tokens", "LLM Cache Read Tokens"),
    ("cache_write_tokens", "LLM Cache Write Tokens"),
    ("latency_ms", "LLM Latency (ms)"),
    ("bytes_tx", "LLM Bytes Sent"),
    ("bytes_rx", "LLM Bytes Received"),
    ("requests", "LLM Requests"),
    ("tool_calls", "LLM Tool Calls"),
    ("tool_call_errors", "LLM Tool-Call Errors"),
];

/// Tool-activity fields. Either being selected enables tool metering; both
/// feed the tool SIGNAL series rather than the generic per-field loop, so
/// they're filtered out of it.
const TOOL_FIELDS: &[&str] = &["tool_calls", "tool_call_errors"];
// Tool activity is shown as a SIGNAL, not a count: a line sits at 0, jumps to
// +1 when a tool was used in that request, and DIPS to -1 when a tool result
// errored — errors dominate (-1 wins over +1 in a request with both), so
// failures read instantly as downward spikes. Magnitudes (exact call/error
// counts) still ride the `RequestCompleted` events for usage/aggregation;
// the SCALARS view is the at-a-glance signal.
//   * `TOOL_SIGNAL_METRIC`        — aggregate, one line per provider/chat.
//   * `TOOL_CALLS_BY_TOOL_METRIC` — per tool name, one 0-baseline line each.
const TOOL_SIGNAL_METRIC: &str = "LLM Tool Calls (signal)";
const TOOL_CALLS_BY_TOOL_METRIC: &str = "LLM Tool Calls by Tool";
// Cumulative magnitudes that complement the at-a-glance signal: a
// monotonically-increasing running total of tool calls AND of tool-call errors
// over the run, both drawn on one chart (a `<series> / calls` line and a
// `<series> / errors` line per provider/chat). Where the signal shows WHEN
// tools fired/failed, this shows HOW MANY have accrued over time — emitted
// every request (tool-free requests just repeat the current total as a flat
// segment) so each line is continuous.
const TOOL_CUMULATIVE_METRIC: &str = "LLM Tool Calls (cumulative)";

/// (field key, cumulative metric title). Every token field plots twice: the
/// per-request value on its `FIELD_SPECS` chart and a running total over the run
/// here, so "how much did this call cost" and "how much has this run spent so
/// far" are both one glance away. The cumulative value sums the SAME value the
/// point series plots — so the four cumulative input/output charts stay the
/// disjoint split described on `FIELD_SPECS`. These series accumulate across ALL
/// of a model's calls: the chat dimension is dropped from the variant (a per-chat
/// total would restart on every new conversation, which for one-shot calls is a
/// chart of single points), leaving one climbing line per provider/model.
const CUMULATIVE_SPECS: &[(&str, &str)] = &[
    ("tokens_in", "LLM Input Tokens (cumulative)"),
    ("tokens_out", "LLM Output Tokens (cumulative)"),
    ("cache_read_tokens", "LLM Cache Read Tokens (cumulative)"),
    ("cache_write_tokens", "LLM Cache Write Tokens (cumulative)"),
];

/// Reported when the configured field list is empty or all-unknown: the
/// on/off switch is `report_task_metrics`; the field list only narrows.
const DEFAULT_FIELDS: &[&str] = &[
    "tokens_in",
    "tokens_out",
    "cache_read_tokens",
    "cache_write_tokens",
    "requests",
    "latency_ms",
    "bytes_tx",
    "bytes_rx",
    "tool_calls",
    "tool_call_errors",
];

struct ConnMeta {
    host: String,
    whitelisted: bool,
}

/// The `RequestCompleted` counts the task-metrics sink reports on.
struct Completed {
    tokens_in: u64,
    tokens_out: u64,
    /// Anthropic prompt-cache split of `tokens_in` (already summed into it):
    /// cache-read and cache-write token counts, each its own scalar series. 0
    /// for OpenAI/Gemini and non-cache requests.
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    latency_ms: u64,
    bytes_tx: u64,
    bytes_rx: u64,
    tool_calls: u64,
    tool_call_errors: u64,
    tool_call_names: Vec<String>,
    /// Tool names whose freshest-turn results failed; drive each tool's signal
    /// to -1 on the "LLM Tool Calls by Tool" graph.
    tool_call_error_names: Vec<String>,
    /// Conversation id from the shim; `Some` => the metric series is split per
    /// chat, `None` => keyed by provider alone.
    chat_id: Option<String>,
    /// The model the request used (from the shim). `Some` => the metric series
    /// includes it (a line per model within a provider); `None` (or equal to the
    /// provider label) => keyed by provider alone.
    model: Option<String>,
}

pub struct Sinks {
    task_id: String,
    /// Task owner + project ids for usage attribution; included on each
    /// `report_llm_usage` event when non-empty (else the backend derives them
    /// from the task).
    user: String,
    project: String,
    report_usage: bool,
    report_metrics: bool,
    metric_fields: Vec<&'static str>,
    aggregator_url: Option<String>,
    conn_meta: HashMap<u64, ConnMeta>,
    /// Per-series (provider, or provider+chat when the request carries a chat
    /// id) running request tally — the value plotted for the "LLM Requests"
    /// field. Not the x-axis; that's the global `seq` below.
    metrics_counts: HashMap<String, u64>,
    /// Global x-axis enumerator: incremented once per captured request and used
    /// as `iter` for every series, so points across all chats share one
    /// monotonic, capture-ordered axis (the "global enumerator" x-axis; the
    /// Wall-Time view rides `timestamp`).
    metrics_seq: u64,
    /// Every tool name seen so far this task. Each gets a continuous 0-baseline
    /// line on the per-tool signal graph, so a tool reads as a flat 0 except
    /// where it spikes to +1 (used) / -1 (errored).
    seen_tools: HashSet<String>,
    /// Per-series running cumulative tool-call / tool-call-error totals, keyed
    /// like `metrics_counts` (provider, or provider+model+chat). These are the
    /// two monotonic lines of the `TOOL_CUMULATIVE_METRIC` chart.
    tool_calls_cum: HashMap<String, u64>,
    tool_call_errors_cum: HashMap<String, u64>,
    /// Running token totals — the values of the `CUMULATIVE_SPECS` charts. Keyed
    /// `field \0 base` (the chat-less series key) so one map covers every token
    /// field without colliding across them, and every chat of a model feeds one
    /// total.
    token_cum: HashMap<String, f64>,
    /// Pending `report_llm_usage` events, flushed in one POST. `usage_bytes`
    /// tracks their serialized size (the flush/cap trigger), mirroring the
    /// agent's `count_bytes`.
    usage_buf: Vec<Value>,
    usage_bytes: usize,
    /// Pending `training_stats_scalar` events, flushed in one `events.add_batch`
    /// POST. `metrics_bytes` tracks their serialized size.
    metrics_buf: Vec<Value>,
    metrics_bytes: usize,
    /// Pending verbatim events for the external aggregator, pre-serialized as
    /// NDJSON lines (joined with `\n` at flush). `aggregator_bytes` is exact.
    aggregator_buf: Vec<String>,
    aggregator_bytes: usize,
    /// Events dropped because a buffer hit `MAX_REQ_BYTES` (a sustained outage);
    /// surfaced as a single `[SNUG-WARN]` line on the next flush.
    dropped: u64,
}

impl Sinks {
    pub fn new(
        task_id: String,
        report_usage: bool,
        report_metrics: bool,
        raw_fields: &[String],
        aggregator_url: Option<String>,
        user: String,
        project: String,
    ) -> Self {
        Sinks {
            task_id,
            user,
            project,
            report_usage,
            report_metrics,
            metric_fields: resolve_fields(raw_fields),
            aggregator_url,
            conn_meta: HashMap::new(),
            metrics_counts: HashMap::new(),
            metrics_seq: 0,
            seen_tools: HashSet::new(),
            tool_calls_cum: HashMap::new(),
            tool_call_errors_cum: HashMap::new(),
            token_cum: HashMap::new(),
            usage_buf: Vec::new(),
            usage_bytes: 0,
            metrics_buf: Vec::new(),
            metrics_bytes: 0,
            aggregator_buf: Vec::new(),
            aggregator_bytes: 0,
            dropped: 0,
        }
    }

    /// True if any sink is active. When false the caller skips parsing events
    /// for the sinks entirely (the default zero-cost path).
    pub fn enabled(&self) -> bool {
        self.report_usage || self.report_metrics || self.aggregator_url.is_some()
    }

    /// True once any buffer has accumulated a full ~1MB packet, so the read loop
    /// flushes it without waiting for the 5s timer. Low-rate buffers still ride
    /// the timer / drain, so latency stays bounded either way.
    pub fn should_flush(&self) -> bool {
        self.usage_bytes >= MAX_PACKET_BYTES
            || self.metrics_bytes >= MAX_PACKET_BYTES
            || self.aggregator_bytes >= MAX_PACKET_BYTES
    }

    /// Flush the buffered sink events: one POST per non-empty buffer. Order is
    /// usage -> metrics -> aggregator so the ClearML sinks take priority over
    /// a possibly-slow external aggregator.
    ///
    /// Durability is delegated to the client: the usage/metrics POSTs retry
    /// TRANSIENT failures forever inside `ClearmlClient` (interruptible by the
    /// drain/abort signal), so the buffer
    /// is held until the send actually succeeds. An `Err` returned here is
    /// therefore a PERMANENT error (e.g. a 4xx) or shutdown — so we DROP the
    /// batch rather than retain it (retaining a permanently-failing batch would
    /// re-fail every flush and block the buffer behind it). The aggregator is a
    /// single bounded attempt by design and is likewise dropped on failure.
    /// Best-effort throughout: errors are logged + surfaced, never fatal. Takes
    /// `&mut ClearmlClient` (lock already held by the caller), like
    /// `LogForwarder::flush`.
    pub fn flush(&mut self, client: &mut ClearmlClient, fwd: &mut LogForwarder) {
        if self.dropped > 0 {
            fwd.enqueue_diagnostic(&format!(
                "[SNUG-WARN] sink buffer overflow, dropped {} events",
                self.dropped
            ));
            self.dropped = 0;
        }

        if !self.usage_buf.is_empty() {
            let n = self.usage_buf.len();
            let result = client.report_llm_usage(&self.usage_buf);
            // Cleared on success OR failure: transient failures are retried
            // forever inside the client, so an Err here is permanent /
            // shutdown — retaining would just re-fail forever.
            self.usage_buf.clear();
            self.usage_bytes = 0;
            match result {
                // Success is silent on the task console — the per-request
                // "queued" line already records what was sent; only failures are
                // surfaced (with the dropped count + the error).
                Ok(_) => {}
                Err(e) => {
                    eprintln!("WARNING: SNUG usage-events POST failed: {}", e);
                    fwd.enqueue_diagnostic(&format!("[SNUG-USAGE] ERR (dropped {}) {}", n, e));
                }
            }
        }

        if !self.metrics_buf.is_empty() {
            let n = self.metrics_buf.len();
            let result = client.events_add_batch(&self.metrics_buf);
            self.metrics_buf.clear();
            self.metrics_bytes = 0;
            match result {
                // Surface the server's per-batch accounting (added/errors) so a
                // SILENTLY dropped event is visible in the task log.
                Ok(resp) => log_add_batch_result(&resp, n, fwd),
                Err(e) => {
                    eprintln!("WARNING: SNUG task-metrics send failed: {}", e);
                    fwd.enqueue_diagnostic(&format!("[SNUG-METRICS] ERR (dropped {}) {}", n, e));
                }
            }
        }

        if !self.aggregator_buf.is_empty() {
            // Clone the URL so the buffer mutations below don't conflict with a
            // borrow of `self.aggregator_url`.
            match self.aggregator_url.clone() {
                Some(url) => {
                    let n = self.aggregator_buf.len();
                    let body = self.aggregator_buf.join("\n");
                    let result = client.aggregator_post(&url, body.as_bytes());
                    self.aggregator_buf.clear();
                    self.aggregator_bytes = 0;
                    match result {
                        Ok(()) => fwd.enqueue_diagnostic(&format!("[SNUG-AGG] OK ({} events)", n)),
                        Err(e) => {
                            eprintln!("WARNING: SNUG aggregator POST failed: {}", e);
                            fwd.enqueue_diagnostic(&format!("[SNUG-AGG] ERR (dropped {}) {}", n, e));
                        }
                    }
                }
                // No URL but a non-empty buffer shouldn't happen; clear to avoid
                // a permanent leak.
                None => {
                    self.aggregator_buf.clear();
                    self.aggregator_bytes = 0;
                }
            }
        }
    }

    /// Buffer one event for the active sinks. Pure buffering — NO network I/O;
    /// all POSTs happen in `flush`. The `iter`/`seq`/`ts_ms` stamping stays here
    /// (capture time, capture order), so batching the send never disturbs the
    /// SCALARS x-axis contract.
    pub fn on_event(&mut self, ev: &Event, fwd: &mut LogForwarder) {
        match ev {
            Event::RequestStarted {
                conn_id,
                host,
                whitelisted,
                ..
            } => self.stash(*conn_id, host.clone(), *whitelisted),

            Event::RequestCompleted {
                conn_id,
                ts_ms,
                latency_ms,
                bytes_tx,
                bytes_rx,
                tokens_in,
                tokens_out,
                cache_read_tokens,
                cache_write_tokens,
                tool_calls,
                tool_call_errors,
                tool_call_names,
                tool_call_error_names,
                chat_id,
                model,
                ..
            } => {
                // Aggregator: buffer the completed event verbatim, independent of
                // the whitelist gate the other sinks apply. Flushed as NDJSON.
                if self.aggregator_url.is_some() {
                    if let Ok(line) = serde_json::to_string(ev) {
                        self.enqueue_aggregator(line);
                    }
                }
                // Pop once and share between both sinks: a second pop would
                // starve whichever sink ran second. Only a whitelisted request
                // is reported.
                let meta = self.pop(*conn_id);
                if let Some(m) = meta {
                    if m.whitelisted {
                        if self.report_usage {
                            self.usage(
                                &m.host,
                                model.as_deref(),
                                *tokens_in,
                                *tokens_out,
                                *ts_ms,
                                fwd,
                            );
                        }
                        if self.report_metrics {
                            let c = Completed {
                                tokens_in: *tokens_in,
                                tokens_out: *tokens_out,
                                cache_read_tokens: *cache_read_tokens,
                                cache_write_tokens: *cache_write_tokens,
                                latency_ms: *latency_ms,
                                bytes_tx: *bytes_tx,
                                bytes_rx: *bytes_rx,
                                tool_calls: *tool_calls,
                                tool_call_errors: *tool_call_errors,
                                tool_call_names: tool_call_names.clone(),
                                tool_call_error_names: tool_call_error_names.clone(),
                                chat_id: chat_id.clone(),
                                model: model.clone(),
                            };
                            self.metrics(&m.host, &c, *ts_ms, fwd);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn stash(&mut self, conn_id: u64, host: String, whitelisted: bool) {
        if self.conn_meta.len() >= CONN_META_HARD_CAP && !self.conn_meta.contains_key(&conn_id) {
            if let Some(&victim) = self.conn_meta.keys().next() {
                self.conn_meta.remove(&victim);
            }
        }
        self.conn_meta.insert(conn_id, ConnMeta { host, whitelisted });
    }

    fn pop(&mut self, conn_id: u64) -> Option<ConnMeta> {
        self.conn_meta.remove(&conn_id)
    }

    // ---- usage ----

    fn usage(
        &mut self,
        host: &str,
        model: Option<&str>,
        tokens_in: u64,
        tokens_out: u64,
        ts_ms: u64,
        fwd: &mut LogForwarder,
    ) {
        if tokens_in == 0 && tokens_out == 0 {
            return; // no usage to report
        }
        // Only bill requests that resolved a model. A real LLM completion always
        // does (from the request body, else the SSE `message_start`); the non-LLM
        // endpoints a metered host also serves (telemetry acks, org settings,
        // sync, rate-limit) do not. Over HTTP/2 the shim can't decode HPACK, so
        // those match the whitelist on host alone and would otherwise be billed as
        // a byte-estimate against the bare provider label — so skip them here.
        let model = match model {
            Some(m) if !m.is_empty() => m,
            _ => return,
        };
        let provider = host_to_model_name(host);
        let event = self.usage_event(&provider, model, tokens_in, tokens_out, resolve_ts(ts_ms));
        fwd.enqueue_diagnostic(&format!(
            "[SNUG-USAGE] queued report_llm_usage provider={:?} model={:?} prompt_tokens={} completion_tokens={}",
            provider, model, tokens_in, tokens_out
        ));
        self.enqueue_usage(vec![event]);
    }

    /// Append usage events to the buffer, tracking serialized bytes and dropping
    /// (newest) past `MAX_REQ_BYTES`. Mirrors `LogForwarder::enqueue`.
    fn enqueue_usage(&mut self, events: Vec<Value>) {
        for ev in events {
            let n = serde_json::to_string(&ev).map(|s| s.len() + 1).unwrap_or(0);
            if self.usage_bytes + n > MAX_REQ_BYTES {
                self.dropped += 1;
                continue;
            }
            self.usage_bytes += n;
            self.usage_buf.push(ev);
        }
    }

    /// Build the `report_llm_usage` event for one request: a single row carrying
    /// the combined `prompt_tokens` + `completion_tokens`, tagged
    /// `source="external"` (the SNUG meters the task's own egress, not the
    /// ClearML gateway). `model` is the per-request model (the usage coset);
    /// `provider` is the coarse host label. `task`/`user`/`project` attribute the
    /// usage — `user`/`project` are omitted when the agent didn't supply them
    /// (the backend then derives them from the task). No I/O.
    fn usage_event(
        &self,
        provider: &str,
        model: &str,
        tokens_in: u64,
        tokens_out: u64,
        ts_ms: u64,
    ) -> Value {
        let mut ev = serde_json::Map::new();
        ev.insert("timestamp".to_string(), json!(ts_ms));
        ev.insert("source".to_string(), json!("external"));
        ev.insert("model".to_string(), json!(model));
        ev.insert("provider".to_string(), json!(provider));
        ev.insert("prompt_tokens".to_string(), json!(tokens_in));
        ev.insert("completion_tokens".to_string(), json!(tokens_out));
        ev.insert("task".to_string(), json!(self.task_id));
        if !self.user.is_empty() {
            ev.insert("user".to_string(), json!(self.user));
        }
        if !self.project.is_empty() {
            ev.insert("project".to_string(), json!(self.project));
        }
        Value::Object(ev)
    }

    // ---- task-metrics ----

    fn metrics(&mut self, host: &str, c: &Completed, ts_ms: u64, fwd: &mut LogForwarder) {
        if self.metric_fields.is_empty() {
            return;
        }
        // Same 0-token gate as `usage`: model switches, `count_tokens` pre-flights,
        // and telemetry resolve a model but carry no usage — they must not plot a
        // point or consume a slot on the global x-axis enumerator.
        if c.tokens_in == 0 && c.tokens_out == 0 {
            return;
        }
        // Same LLM gate as `usage`: a model-less RequestCompleted is a non-LLM
        // call on a metered host (host-only whitelist match under HTTP/2), not a
        // real completion, so it must not plot a point or consume a slot on the
        // global x-axis enumerator.
        if c.model.as_deref().map(str::is_empty).unwrap_or(true) {
            return;
        }
        let provider = host_to_model_name(host);
        let (events, iter) = self.metric_events(&provider, c, resolve_ts(ts_ms));
        if events.is_empty() {
            return;
        }
        fwd.enqueue_diagnostic(&format!(
            "[SNUG-METRICS] queued scalars provider={:?} model={:?} chat={:?} iter={} fields={}",
            provider,
            c.model.as_deref(),
            c.chat_id.as_deref(),
            iter,
            events.len()
        ));
        self.enqueue_metrics(events);
    }

    /// Append scalar events to the buffer, tracking serialized bytes and
    /// dropping (newest) past `MAX_REQ_BYTES`. Mirrors `LogForwarder::enqueue`.
    fn enqueue_metrics(&mut self, events: Vec<Value>) {
        for ev in events {
            let n = serde_json::to_string(&ev).map(|s| s.len() + 1).unwrap_or(0);
            if self.metrics_bytes + n > MAX_REQ_BYTES {
                self.dropped += 1;
                continue;
            }
            self.metrics_bytes += n;
            self.metrics_buf.push(ev);
        }
    }

    /// Append one pre-serialized NDJSON line to the aggregator buffer, tracking
    /// bytes and dropping (newest) past `MAX_REQ_BYTES`.
    fn enqueue_aggregator(&mut self, line: String) {
        let n = line.len() + 1;
        if self.aggregator_bytes + n > MAX_REQ_BYTES {
            self.dropped += 1;
            return;
        }
        self.aggregator_bytes += n;
        self.aggregator_buf.push(line);
    }

    /// Build the `training_stats_scalar` events for one request. The series
    /// (variant) is the provider, plus the model when the shim parsed one (so a
    /// task using several models of one provider plots a line per model), plus
    /// the chat ordinal when the request identified its conversation — so one
    /// task's N chats (and N models) become distinct lines. The x-axis (`iter`)
    /// is a single global enumerator incremented once per captured request
    /// (shared across all series, capture-ordered), while `timestamp` (`ts_ms`)
    /// carries the capture wall-time for the Wall-Time view. The per-series
    /// request tally is decoupled from the axis and feeds only the "requests"
    /// value. Mutates both counters; no I/O. Returns (events, iter).
    fn metric_events(&mut self, provider: &str, c: &Completed, ts_ms: u64) -> (Vec<Value>, i64) {
        // Series key: provider, then the model when known (a line per model),
        // then the chat ordinal when present. The model segment is omitted when
        // unknown or identical to the provider label, so an unparsed/unknown
        // model keeps the provider-only series instead of "Anthropic /
        // Anthropic".
        let base = match c.model.as_deref() {
            Some(m) if !m.is_empty() && m != provider => format!("{} / {}", provider, m),
            _ => provider.to_string(),
        };
        // `base` stays live past this: the cumulative token series key on it, so
        // every chat of a model accumulates into one line.
        let variant = match c.chat_id.as_deref() {
            Some(id) if !id.is_empty() => format!("{} / chat {}", base, id),
            _ => base.clone(),
        };
        // Per-series cumulative request count (the "requests" value only).
        let req_count = {
            let n = self.metrics_counts.entry(variant.clone()).or_insert(0);
            *n += 1;
            *n
        };
        // Global enumerator drives the x-axis for every series this request.
        let iter = self.metrics_seq as i64;
        self.metrics_seq += 1;
        // Point values first: the immutable borrow of `metric_fields` has to end
        // before the cumulative pass below mutates the running totals.
        let points: Vec<(&'static str, f64)> = self
            .metric_fields
            .iter()
            // Tool fields share the merged timeline below, not the generic loop.
            .filter(|&&field| !TOOL_FIELDS.contains(&field))
            .map(|&field| {
                let value = match field {
                    "requests" => req_count as f64,
                    // Scalar SPLIT: the "LLM Input Tokens" series is FRESH
                    // (non-cached) input only, so it + cache-read + cache-write are
                    // disjoint and sum to the billable input total. The usage event
                    // still reports that combined total as prompt_tokens (see the
                    // report path / usage_event) — only the scalars split it.
                    "tokens_in" => c
                        .tokens_in
                        .saturating_sub(c.cache_read_tokens)
                        .saturating_sub(c.cache_write_tokens) as f64,
                    "tokens_out" => c.tokens_out as f64,
                    "cache_read_tokens" => c.cache_read_tokens as f64,
                    "cache_write_tokens" => c.cache_write_tokens as f64,
                    "latency_ms" => c.latency_ms as f64,
                    "bytes_tx" => c.bytes_tx as f64,
                    "bytes_rx" => c.bytes_rx as f64,
                    _ => 0.0,
                };
                (field, value)
            })
            .collect();
        let mut events: Vec<Value> = Vec::with_capacity(points.len());
        for (field, value) in points {
            events.push(scalar_event(
                &self.task_id,
                metric_title(field),
                &variant,
                value,
                iter,
                ts_ms,
            ));
            // Token fields also carry a running total over every call of this
            // model — keyed on `base`, so chats merge into one climbing line
            // instead of restarting per conversation. Emitted every request (a
            // zero-token request just repeats the current total as a flat
            // segment) so each line stays continuous.
            if let Some(title) = cumulative_title(field) {
                let total = {
                    let n = self
                        .token_cum
                        .entry(format!("{}\u{0}{}", field, base))
                        .or_insert(0.0);
                    *n += value;
                    *n
                };
                events.push(scalar_event(&self.task_id, title, &base, total, iter, ts_ms));
            }
        }
        // Tool activity as a SIGNAL (0 idle / +1 used / -1 errored; errors win).
        // One aggregate line per (provider/chat), plus a 0-baseline line per tool.
        if self.metric_fields.contains(&"tool_calls")
            || self.metric_fields.contains(&"tool_call_errors")
        {
            events.push(scalar_event(
                &self.task_id,
                TOOL_SIGNAL_METRIC,
                &variant,
                tool_signal(c.tool_calls > 0, c.tool_call_errors > 0),
                iter,
                ts_ms,
            ));
            // Cumulative magnitudes over time: a monotonically-increasing running
            // total of tool calls and of tool-call errors for this series, both on
            // one chart (a calls line + an errors line) so the SIGNAL's spikes are
            // complemented by real totals. Emitted every request (even tool-free
            // ones, which add 0) so each line stays continuous.
            let calls_cum = {
                let n = self.tool_calls_cum.entry(variant.clone()).or_insert(0);
                *n += c.tool_calls;
                *n
            };
            let errors_cum = {
                let n = self.tool_call_errors_cum.entry(variant.clone()).or_insert(0);
                *n += c.tool_call_errors;
                *n
            };
            events.push(scalar_event(
                &self.task_id,
                TOOL_CUMULATIVE_METRIC,
                &format!("{} / calls", variant),
                calls_cum as f64,
                iter,
                ts_ms,
            ));
            events.push(scalar_event(
                &self.task_id,
                TOOL_CUMULATIVE_METRIC,
                &format!("{} / errors", variant),
                errors_cum as f64,
                iter,
                ts_ms,
            ));
            // Per-tool: each tool seen so far gets a continuous 0-baseline line
            // that jumps to +1 the requests it's used and -1 the requests its
            // result errored. The shim resolves each errored tool_use_id back to
            // its tool name within the request body.
            let called: HashSet<&str> = c.tool_call_names.iter().map(String::as_str).collect();
            let errored: HashSet<&str> =
                c.tool_call_error_names.iter().map(String::as_str).collect();
            for name in called.iter().chain(errored.iter()) {
                self.seen_tools.insert((*name).to_string());
            }
            let mut tools: Vec<&String> = self.seen_tools.iter().collect();
            tools.sort(); // stable output order
            for tool in tools {
                let sig = tool_signal(called.contains(tool.as_str()), errored.contains(tool.as_str()));
                events.push(scalar_event(&self.task_id, TOOL_CALLS_BY_TOOL_METRIC, tool, sig, iter, ts_ms));
            }
        }
        (events, iter)
    }
}

/// Coarse host -> friendly provider label for the usage event + scalar series.
/// An unrecognized host reports under its own hostname (the catch-all).
fn host_to_model_name(host: &str) -> String {
    match host {
        "" => "Unknown Model".to_string(),
        "api.openai.com" => "OpenAI".to_string(),
        "api.anthropic.com" => "Anthropic".to_string(),
        "claude.ai" => "Anthropic".to_string(),
        "generativelanguage.googleapis.com" => "Gemini".to_string(),
        other => other.to_string(),
    }
}

/// Filter the configured list to known fields (dedup, order-preserving). An
/// empty or all-unknown selection falls back to all fields.
fn resolve_fields(raw: &[String]) -> Vec<&'static str> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for f in raw {
        if let Some(&(key, _)) = FIELD_SPECS.iter().find(|(k, _)| *k == f.as_str()) {
            if seen.insert(key) {
                out.push(key);
            }
        }
    }
    if out.is_empty() {
        DEFAULT_FIELDS.to_vec()
    } else {
        out
    }
}

/// Title for a resolved field. The fallback is unreachable (fields are
/// validated against `FIELD_SPECS`) but keeps the lookup total.
fn metric_title(field: &str) -> &'static str {
    FIELD_SPECS
        .iter()
        .find(|(k, _)| *k == field)
        .map(|(_, title)| *title)
        .unwrap_or("LLM Metric")
}

/// Running-total chart title for a field, or `None` for fields that only plot
/// per-request values (latency/bytes don't accumulate meaningfully, and
/// "requests" is already a running count).
fn cumulative_title(field: &str) -> Option<&'static str> {
    CUMULATIVE_SPECS
        .iter()
        .find(|(k, _)| *k == field)
        .map(|(_, title)| *title)
}

fn scalar_event(task: &str, metric: &str, variant: &str, value: f64, iter: i64, ts_ms: u64) -> Value {
    json!({
        "type": "training_stats_scalar",
        "task": task,
        "metric": metric,
        "variant": variant,
        "value": value,
        "iter": iter,
        "timestamp": ts_ms,
    })
}

/// Tool-activity signal value: -1 if a result errored (dominant), +1 if a tool
/// was used, else 0. Shared by the aggregate and per-tool signal series so both
/// encode failures the same way (errors dip below the 0 baseline).
fn tool_signal(used: bool, errored: bool) -> f64 {
    if errored {
        -1.0
    } else if used {
        1.0
    } else {
        0.0
    }
}

/// Log the `events.add_batch` accounting. The endpoint returns 200 even when it
/// silently drops individual events (`{data:{added,errors,errors_info}}`), so a
/// rejected plot would otherwise look like it was never sent. Emits the
/// `errors_info` whenever any event was refused. Best-effort, off the hot path.
fn log_add_batch_result(resp: &Value, sent: usize, fwd: &mut LogForwarder) {
    let data = resp.get("data").unwrap_or(resp);
    let added = data.get("added").and_then(|v| v.as_i64()).unwrap_or(-1);
    let errors = data.get("errors").and_then(|v| v.as_i64()).unwrap_or(0);
    if errors > 0 {
        let info = data
            .get("errors_info")
            .map(|v| v.to_string())
            .unwrap_or_default();
        fwd.enqueue_diagnostic(&format!(
            "[SNUG-METRICS] add_batch sent={} added={} errors={} info={}",
            sent, added, errors, info
        ));
    } else {
        fwd.enqueue_diagnostic(&format!(
            "[SNUG-METRICS] add_batch sent={} added={} errors=0",
            sent, added
        ));
    }
}

/// The shim stamps `ts_ms` on every event; fall back to now only if it's absent.
fn resolve_ts(ts_ms: u64) -> u64 {
    if ts_ms != 0 {
        ts_ms
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sinks(fields: &[&str]) -> Sinks {
        let raw: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
        Sinks::new("task-1".into(), true, true, &raw, None, String::new(), String::new())
    }

    // Shared event builders for the usage tests: a whitelisted RequestStarted
    // (the gate) and its matching RequestCompleted carrying model + token counts.
    fn started(conn_id: u64, host: &str) -> Event {
        Event::RequestStarted {
            conn_id,
            ts_ms: 1000,
            host: host.into(),
            path: "/v1/x".into(),
            method: "POST".into(),
            whitelisted: true,
            inject_headers: true,
        }
    }
    fn rc(conn_id: u64, model: Option<&str>, tin: u64, tout: u64) -> Event {
        Event::RequestCompleted {
            conn_id,
            ts_ms: 1000,
            status: Some(200),
            latency_ms: 1,
            bytes_tx: 1,
            bytes_rx: 1,
            tokens_in: tin,
            tokens_out: tout,
            tokens_measured: true,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: model.map(str::to_string),
        }
    }

    #[test]
    fn enabled_reflects_each_sink() {
        let none = Sinks::new("t".into(), false, false, &[], None, String::new(), String::new());
        assert!(!none.enabled());
        let usage = Sinks::new("t".into(), true, false, &[], None, String::new(), String::new());
        assert!(usage.enabled());
        let agg = Sinks::new("t".into(), false, false, &[], Some("http://x".into()), String::new(), String::new());
        assert!(agg.enabled(), "aggregator-only must still be enabled");
    }

    #[test]
    fn host_mapping() {
        assert_eq!(host_to_model_name("api.anthropic.com"), "Anthropic");
        assert_eq!(host_to_model_name("api.openai.com"), "OpenAI");
        assert_eq!(host_to_model_name("generativelanguage.googleapis.com"), "Gemini");
        // The consumer chat wire rolls up under the same provider as the API path.
        assert_eq!(host_to_model_name("claude.ai"), "Anthropic");
        assert_eq!(host_to_model_name("unknown.example.com"), "unknown.example.com");
        assert_eq!(host_to_model_name(""), "Unknown Model");
    }

    #[test]
    fn resolve_fields_known_dedup_order() {
        let r = |v: &[&str]| resolve_fields(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(r(&["requests", "tokens_in"]), vec!["requests", "tokens_in"]);
        assert_eq!(r(&["tokens_in", "tokens_in"]), vec!["tokens_in"]); // dedup
        assert_eq!(r(&["bogus"]), DEFAULT_FIELDS.to_vec()); // all-unknown -> default
        assert_eq!(r(&[]), DEFAULT_FIELDS.to_vec()); // empty -> default
        assert_eq!(r(&["tokens_out", "nope"]), vec!["tokens_out"]); // drop unknown
    }

    #[test]
    fn usage_event_combines_in_and_out_with_source() {
        let s = sinks(&["tokens_in"]);
        let e = s.usage_event("Anthropic", "claude-haiku-4-5", 19, 42, 1000);
        assert_eq!(e["timestamp"], 1000);
        assert_eq!(e["source"], "external");
        assert_eq!(e["provider"], "Anthropic");
        assert_eq!(e["model"], "claude-haiku-4-5");
        // input + output combined into one event, renamed per the schema.
        assert_eq!(e["prompt_tokens"], 19);
        assert_eq!(e["completion_tokens"], 42);
        assert_eq!(e["task"], "task-1"); // sinks() builds task "task-1"
        // user/project omitted when unset; the legacy fields are gone.
        assert!(e.get("user").is_none() && e.get("project").is_none());
        assert!(e.get("usage").is_none() && e.get("label").is_none());
        assert!(e.get("target_url").is_none() && e.get("event_type").is_none());
    }

    #[test]
    fn usage_event_carries_user_and_project_when_set() {
        let s = Sinks::new(
            "task-1".into(),
            true,
            false,
            &[],
            None,
            "user-9".into(),
            "proj-7".into(),
        );
        let e = s.usage_event("OpenAI", "gpt-4o", 1, 2, 1000);
        assert_eq!(e["user"], "user-9");
        assert_eq!(e["project"], "proj-7");
    }

    #[test]
    fn usage_skips_when_no_tokens() {
        // Both token counts zero -> nothing buffered (no usage to report).
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new("task-1".into(), true, false, &[], None, String::new(), String::new());
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, Some("claude-haiku-4-5"), 0, 0), &mut fwd);
        assert!(s.usage_buf.is_empty(), "zero-token request reports nothing");
    }

    #[test]
    fn metrics_skips_when_no_tokens() {
        // A model-resolved request with zero tokens (a model switch, a
        // `count_tokens` pre-flight, telemetry) must not plot a scalar point or
        // advance the x-axis; a real completion still does.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new(
            "task-1".into(),
            false,
            true,
            &["tokens_in".to_string(), "tokens_out".to_string()],
            None,
            String::new(),
            String::new(),
        );
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, Some("claude-haiku-4-5"), 0, 0), &mut fwd);
        assert!(s.metrics_buf.is_empty(), "zero-token request plots no scalars");
        assert_eq!(s.metrics_seq, 0, "x-axis not advanced for a zero-token request");

        s.on_event(&started(2, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(2, Some("claude-haiku-4-5"), 10, 5), &mut fwd);
        assert!(!s.metrics_buf.is_empty(), "a real completion still plots scalars");
    }

    #[test]
    fn estimated_usage_is_reported_unchanged() {
        // Estimated (byte-ratio) counts are reported the same as measured ones:
        // the schema has no est/measured distinction and we report both.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new("task-1".into(), true, false, &[], None, String::new(), String::new());
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(
            &Event::RequestCompleted {
                conn_id: 1,
                ts_ms: 1000,
                status: Some(200),
                latency_ms: 1,
                bytes_tx: 1,
                bytes_rx: 1,
                tokens_in: 5,
                tokens_out: 6,
                tokens_measured: false, // estimate
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                tool_calls: 0,
                tool_call_errors: 0,
                tool_call_names: vec![],
                tool_call_error_names: vec![],
                chat_id: None,
                model: Some("claude-haiku-4-5".into()),
            },
            &mut fwd,
        );
        let e = s.usage_buf.last().expect("estimate buffered");
        assert_eq!(e["prompt_tokens"], 5);
        assert_eq!(e["completion_tokens"], 6);
    }

    #[test]
    fn usage_reported_with_model_skipped_without() {
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        // Usage only (no metrics noise), no network (no client passed).
        let mut s = Sinks::new("task-1".into(), true, false, &[], None, String::new(), String::new());

        // model present -> billed; used verbatim, provider stays the host label.
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, Some("claude-opus-4-20250514"), 10, 0), &mut fwd);
        assert_eq!(s.usage_buf.len(), 1, "a request with a model is billed");
        let e = s.usage_buf.last().expect("usage event buffered");
        assert_eq!(e["model"], "claude-opus-4-20250514");
        assert_eq!(e["provider"], "Anthropic");
        assert_eq!(e["prompt_tokens"], 10);
        assert_eq!(e["completion_tokens"], 0);
        assert_eq!(e["source"], "external");

        // model absent -> not an LLM call -> skipped, even with nonzero
        // (byte-estimate) tokens. Nothing new is buffered.
        s.on_event(&started(2, "api.openai.com"), &mut fwd);
        s.on_event(&rc(2, None, 3, 5), &mut fwd);
        assert_eq!(s.usage_buf.len(), 1, "a model-less request is not billed");

        // empty-string model is treated the same as absent -> skipped.
        s.on_event(&started(3, "api.openai.com"), &mut fwd);
        s.on_event(&rc(3, Some(""), 3, 5), &mut fwd);
        assert_eq!(s.usage_buf.len(), 1, "an empty-model request is not billed");
    }

    #[test]
    fn metrics_skipped_when_model_missing() {
        // The scalar/metrics path gates on a resolved model just like usage: a
        // model-less RequestCompleted (a non-LLM call on a metered host) plots
        // nothing and does not advance the global x-axis enumerator.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new(
            "task-1".into(),
            false,
            true,
            &["tokens_in".to_string()],
            None,
            String::new(),
            String::new(),
        );
        // No model -> skipped.
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, None, 100, 20), &mut fwd);
        assert!(s.metrics_buf.is_empty(), "model-less request plots nothing");
        assert_eq!(s.metrics_seq, 0, "x-axis enumerator not advanced");

        // With a model -> plotted, enumerator advances once.
        s.on_event(&started(2, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(2, Some("claude-haiku-4-5"), 100, 20), &mut fwd);
        assert!(!s.metrics_buf.is_empty(), "a request with a model is plotted");
        assert_eq!(s.metrics_seq, 1, "x-axis advanced once for the LLM request");
    }

    #[test]
    fn model_flows_to_both_usage_and_metrics() {
        // A single RequestCompleted carrying a model must populate the model on
        // BOTH sinks from the same `model` field: the usage event's `model` and
        // the metrics series variant. Guards against one sink dropping it.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new(
            "task-1".into(),
            true,
            true,
            &["tokens_in".to_string()],
            None,
            String::new(),
            String::new(),
        );
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, Some("claude-haiku-4-5"), 100, 20), &mut fwd);

        let u = s.usage_buf.last().expect("usage event buffered");
        assert_eq!(u["model"], "claude-haiku-4-5", "usage carries the model");
        assert_eq!(u["provider"], "Anthropic");

        let m = s
            .metrics_buf
            .iter()
            .find(|e| e["metric"] == "LLM Input Tokens")
            .expect("metric event buffered");
        assert_eq!(
            m["variant"], "Anthropic / claude-haiku-4-5",
            "metrics variant carries the model dimension"
        );
    }

    #[test]
    fn metric_events_variant_includes_model() {
        let mut s = sinks(&["tokens_in"]);
        let mk = |model: Option<&str>, chat: Option<&str>| Completed {
            tokens_in: 10,
            tokens_out: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: chat.map(|c| c.to_string()),
            model: model.map(|m| m.to_string()),
        };
        // Model known -> appears in the variant after the provider.
        let (e, _) = s.metric_events("Anthropic", &mk(Some("claude-haiku-4-5"), None), 1000);
        assert_eq!(e[0]["variant"], "Anthropic / claude-haiku-4-5");
        // Model + chat -> both in the variant.
        let (e, _) = s.metric_events("Anthropic", &mk(Some("claude-haiku-4-5"), Some("3")), 1000);
        assert_eq!(e[0]["variant"], "Anthropic / claude-haiku-4-5 / chat 3");
        // Model unknown -> provider only, chat still applies.
        let (e, _) = s.metric_events("OpenAI", &mk(None, Some("2")), 1000);
        assert_eq!(e[0]["variant"], "OpenAI / chat 2");
        // Model equal to the provider label -> not duplicated.
        let (e, _) = s.metric_events("Anthropic", &mk(Some("Anthropic"), None), 1000);
        assert_eq!(e[0]["variant"], "Anthropic");
    }

    #[test]
    fn metric_events_emit_cache_read_and_write_series() {
        // The input series are a DISJOINT split: "LLM Input Tokens" is FRESH only
        // (tokens_in - cache_read - cache_write), and the cache buckets plot on
        // their own series; the three sum to the billable input total (which the
        // usage event still reports combined as prompt_tokens).
        let mut s = sinks(&["tokens_in", "cache_read_tokens", "cache_write_tokens"]);
        let c = Completed {
            tokens_in: 45302,
            tokens_out: 13,
            cache_read_tokens: 45000,
            cache_write_tokens: 300,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: Some("claude-sonnet-4-5".into()),
        };
        let (e, _) = s.metric_events("Anthropic", &c, 1000);
        let value = |metric: &str| -> f64 {
            e.iter()
                .find(|ev| ev["metric"] == metric && ev["variant"] == "Anthropic / claude-sonnet-4-5")
                .and_then(|ev| ev["value"].as_f64())
                .unwrap_or_else(|| panic!("{metric} series missing"))
        };
        assert_eq!(value("LLM Input Tokens"), 2.0, "fresh only: 45302 - 45000 - 300");
        assert_eq!(value("LLM Cache Read Tokens"), 45000.0);
        assert_eq!(value("LLM Cache Write Tokens"), 300.0);
    }

    #[test]
    fn metric_events_cumulative_token_series() {
        // Every token field plots twice: the per-request value and a
        // monotonically-increasing per-series running total. The cumulative
        // input total sums the FRESH split, so the four cumulative charts stay
        // disjoint just like the point charts.
        let mut s = sinks(&["tokens_in", "tokens_out", "cache_read_tokens", "cache_write_tokens"]);
        let mk = |tin: u64, tout: u64, cr: u64, cw: u64, chat: &str| Completed {
            tokens_in: tin,
            tokens_out: tout,
            cache_read_tokens: cr,
            cache_write_tokens: cw,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: Some(chat.into()),
            model: None,
        };
        let value = |events: &[Value], metric: &str, variant: &str| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == metric && e["variant"] == variant)
                .and_then(|e| e["value"].as_f64())
                .unwrap_or_else(|| panic!("{metric} / {variant} missing"))
        };

        // First call: 10 fresh in (110 - 60 read - 40 write), 30 out. The
        // cumulative variant drops the chat segment the point series carries.
        let (e1, _) = s.metric_events("Anthropic", &mk(110, 30, 60, 40, "1"), 1000);
        assert_eq!(value(&e1, "LLM Input Tokens", "Anthropic / chat 1"), 10.0);
        assert_eq!(value(&e1, "LLM Input Tokens (cumulative)", "Anthropic"), 10.0);
        assert_eq!(value(&e1, "LLM Output Tokens (cumulative)", "Anthropic"), 30.0);
        assert_eq!(value(&e1, "LLM Cache Read Tokens (cumulative)", "Anthropic"), 60.0);
        assert_eq!(value(&e1, "LLM Cache Write Tokens (cumulative)", "Anthropic"), 40.0);

        // Second call adds on top; the point series still shows this call alone.
        let (e2, _) = s.metric_events("Anthropic", &mk(30, 5, 0, 0, "1"), 2000);
        assert_eq!(value(&e2, "LLM Input Tokens", "Anthropic / chat 1"), 30.0, "point value is per-request");
        assert_eq!(value(&e2, "LLM Input Tokens (cumulative)", "Anthropic"), 40.0);
        assert_eq!(value(&e2, "LLM Output Tokens (cumulative)", "Anthropic"), 35.0);
        // Cache buckets got nothing this call: the totals hold flat, not drop out.
        assert_eq!(value(&e2, "LLM Cache Read Tokens (cumulative)", "Anthropic"), 60.0);
        assert_eq!(value(&e2, "LLM Cache Write Tokens (cumulative)", "Anthropic"), 40.0);

        // A DIFFERENT chat of the same model feeds the SAME total — the point
        // series splits per conversation, the running total is across all calls.
        let (e3, _) = s.metric_events("Anthropic", &mk(7, 3, 0, 0, "2"), 3000);
        assert_eq!(value(&e3, "LLM Input Tokens", "Anthropic / chat 2"), 7.0);
        assert_eq!(value(&e3, "LLM Input Tokens (cumulative)", "Anthropic"), 47.0);
        assert_eq!(value(&e3, "LLM Output Tokens (cumulative)", "Anthropic"), 38.0);

        // A different model is its own line.
        let mut c = mk(5, 2, 0, 0, "1");
        c.model = Some("claude-opus-4-5".into());
        let (e4, _) = s.metric_events("Anthropic", &c, 4000);
        assert_eq!(
            value(&e4, "LLM Input Tokens (cumulative)", "Anthropic / claude-opus-4-5"),
            5.0,
            "per-model total, not folded into the model-less line"
        );
    }

    #[test]
    fn metric_events_no_cumulative_for_non_token_fields() {
        // Latency/bytes don't accumulate meaningfully and "requests" is already a
        // running count — none of them get a "(cumulative)" twin.
        let mut s = sinks(&["latency_ms", "bytes_tx", "bytes_rx", "requests"]);
        let c = Completed {
            tokens_in: 10,
            tokens_out: 1,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency_ms: 42,
            bytes_tx: 7,
            bytes_rx: 9,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: None,
        };
        let (e, _) = s.metric_events("Anthropic", &c, 1000);
        assert_eq!(e.len(), 4, "one event per field, no cumulative twins");
        assert!(!e
            .iter()
            .any(|ev| ev["metric"].as_str().is_some_and(|m| m.ends_with("(cumulative)"))));
    }

    #[test]
    fn metric_events_iter_and_cumulative_requests() {
        let mut s = sinks(&["tokens_in", "requests"]);
        let c1 = Completed { tokens_in: 100, tokens_out: 0, cache_read_tokens: 0, cache_write_tokens: 0, latency_ms: 5, bytes_tx: 1, bytes_rx: 2, tool_calls: 0, tool_call_errors: 0, tool_call_names: vec![], tool_call_error_names: vec![], chat_id: None, model: None };
        let requests = |events: &[Value]| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Requests")
                .and_then(|e| e["value"].as_f64())
                .expect("requests series present")
        };
        let (e1, i1) = s.metric_events("Anthropic", &c1, 1000);
        assert_eq!(i1, 0);
        // tokens_in plots twice (point + running total); requests plots once.
        assert_eq!(e1.len(), 3);
        assert_eq!(e1[0]["metric"], "LLM Input Tokens");
        assert_eq!(e1[0]["variant"], "Anthropic");
        assert_eq!(e1[0]["value"], 100.0);
        assert_eq!(e1[0]["task"], "task-1");
        assert_eq!(requests(&e1), 1.0); // first request for this provider

        let c2 = Completed { tokens_in: 50, tokens_out: 0, cache_read_tokens: 0, cache_write_tokens: 0, latency_ms: 5, bytes_tx: 1, bytes_rx: 2, tool_calls: 0, tool_call_errors: 0, tool_call_names: vec![], tool_call_error_names: vec![], chat_id: None, model: None };
        let (e2, i2) = s.metric_events("Anthropic", &c2, 2000);
        assert_eq!(i2, 1); // global enumerator advanced
        assert_eq!(requests(&e2), 2.0); // cumulative request count

        // distinct provider keeps its own per-series request count, but the
        // x-axis is the SAME global enumerator (continues to 2, not a reset).
        let (e3, i3) = s.metric_events("OpenAI", &c2, 3000);
        assert_eq!(i3, 2, "global enumerator is shared across series");
        assert_eq!(requests(&e3), 1.0);
    }

    #[test]
    fn metric_events_group_by_chat() {
        // Cap 2: a chat id splits the series (variant = provider + chat ordinal).
        // The x-axis is now a single global enumerator shared across chats; the
        // per-series request count is decoupled from it.
        let mut s = sinks(&["tokens_in", "requests"]);
        let mk = |id: &str| Completed {
            tokens_in: 10,
            tokens_out: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: Some(id.into()),
            model: None,
        };
        let requests = |events: &[Value]| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Requests")
                .and_then(|e| e["value"].as_f64())
                .expect("requests series present")
        };
        let (trip1, i_trip1) = s.metric_events("Anthropic", &mk("1"), 1000);
        assert_eq!(trip1[0]["variant"], "Anthropic / chat 1");
        assert_eq!(i_trip1, 0);
        assert_eq!(requests(&trip1), 1.0, "chat 1 first request");

        // A different chat is its own series, but the global enumerator advances.
        let (story1, i_story1) = s.metric_events("Anthropic", &mk("2"), 1000);
        assert_eq!(story1[0]["variant"], "Anthropic / chat 2");
        assert_eq!(i_story1, 1, "global enumerator advances across chats");
        assert_eq!(requests(&story1), 1.0, "chat 2 has its own request count");

        // The first chat's next turn advances the global axis again, while its
        // own request count climbs independently.
        let (trip2, i_trip2) = s.metric_events("Anthropic", &mk("1"), 1000);
        assert_eq!(i_trip2, 2, "global enumerator keeps climbing");
        assert_eq!(requests(&trip2), 2.0, "requests cumulative within the chat");
    }

    #[test]
    fn metric_events_per_tool_signal() {
        let mut s = sinks(&["tool_calls", "tool_call_errors"]);
        let mk = |names: Vec<&str>, errs: Vec<&str>| Completed {
            tokens_in: 0,
            tokens_out: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: names.len() as u64,
            tool_call_errors: errs.len() as u64,
            tool_call_names: names.iter().map(|s| s.to_string()).collect(),
            tool_call_error_names: errs.iter().map(|s| s.to_string()).collect(),
            chat_id: Some("1".into()),
            model: None,
        };
        let per_tool = |events: &[Value], tool: &str| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Tool Calls by Tool" && e["variant"] == tool)
                .and_then(|e| e["value"].as_f64())
                .unwrap_or(f64::NAN)
        };

        // get_weather used cleanly, search's result errored.
        let (e1, _) = s.metric_events("Anthropic", &mk(vec!["get_weather", "search"], vec!["search"]), 1000);
        assert_eq!(per_tool(&e1, "get_weather"), 1.0, "used tool -> +1");
        assert_eq!(per_tool(&e1, "search"), -1.0, "errored tool -> -1 (dominates)");
        // Signal-only: no raw-count, error-overlay, or plot variants.
        assert!(
            !e1.iter().any(|e| e["metric"] == "LLM Tool Calls"),
            "no raw-count aggregate metric"
        );
        assert!(
            !e1.iter()
                .any(|e| e["variant"].as_str().is_some_and(|v| v.ends_with("(err)"))),
            "no (err) overlay variants"
        );
        assert!(!e1.iter().any(|e| e["type"] == "plot"), "no plot events");

        // A later request with no tool activity: both already-seen tools sit at
        // the continuous 0 baseline (not absent).
        let (e2, _) = s.metric_events("Anthropic", &mk(vec![], vec![]), 2000);
        assert_eq!(per_tool(&e2, "get_weather"), 0.0, "baseline 0 when unused");
        assert_eq!(per_tool(&e2, "search"), 0.0, "baseline 0 when unused");
    }

    #[test]
    fn metric_events_signal_encodes_minus_one_zero_plus_one() {
        // Aggregate signal line: -1 on error (dominant), +1 on a clean tool use,
        // 0 when the request had no tool activity.
        let mut s = sinks(&["tool_calls", "tool_call_errors"]);
        let mk = |calls: u64, errs: u64| Completed {
            tokens_in: 0,
            tokens_out: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: calls,
            tool_call_errors: errs,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: None,
        };
        let signal = |events: &[Value]| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Tool Calls (signal)")
                .and_then(|e| e["value"].as_f64())
                .expect("signal series present")
        };
        let (clean, _) = s.metric_events("Anthropic", &mk(2, 0), 1000);
        assert_eq!(signal(&clean), 1.0, "clean tool use -> +1");
        let (errored, _) = s.metric_events("Anthropic", &mk(2, 1), 2000);
        assert_eq!(signal(&errored), -1.0, "any error -> -1 (dominates)");
        let (idle, _) = s.metric_events("Anthropic", &mk(0, 0), 3000);
        assert_eq!(signal(&idle), 0.0, "no tool activity -> 0 baseline");
    }

    #[test]
    fn metric_events_cumulative_tool_calls() {
        // The cumulative chart carries a `/ calls` line and a `/ errors` line per
        // series, each a monotonically-increasing running total. Tool-free
        // requests repeat the current total (a flat segment), and distinct chats
        // accumulate independently.
        let mut s = sinks(&["tool_calls", "tool_call_errors"]);
        let mk = |calls: u64, errs: u64, chat: &str| Completed {
            tokens_in: 0,
            tokens_out: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tool_calls: calls,
            tool_call_errors: errs,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: Some(chat.into()),
            model: None,
        };
        let cum = |events: &[Value], variant: &str| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Tool Calls (cumulative)" && e["variant"] == variant)
                .and_then(|e| e["value"].as_f64())
                .unwrap_or(f64::NAN)
        };

        // chat 1: 2 calls, 1 errored.
        let (e1, _) = s.metric_events("Anthropic", &mk(2, 1, "1"), 1000);
        assert_eq!(cum(&e1, "Anthropic / chat 1 / calls"), 2.0);
        assert_eq!(cum(&e1, "Anthropic / chat 1 / errors"), 1.0);

        // chat 1 again: totals climb (4 calls, 1 error so far).
        let (e2, _) = s.metric_events("Anthropic", &mk(2, 0, "1"), 2000);
        assert_eq!(cum(&e2, "Anthropic / chat 1 / calls"), 4.0, "calls accumulate");
        assert_eq!(cum(&e2, "Anthropic / chat 1 / errors"), 1.0, "errors hold flat");

        // chat 1, tool-free request: both lines repeat the current totals so the
        // series stays continuous rather than dropping out.
        let (e3, _) = s.metric_events("Anthropic", &mk(0, 0, "1"), 3000);
        assert_eq!(cum(&e3, "Anthropic / chat 1 / calls"), 4.0, "flat through idle");
        assert_eq!(cum(&e3, "Anthropic / chat 1 / errors"), 1.0, "flat through idle");

        // A different chat is its own independent running total.
        let (e4, _) = s.metric_events("Anthropic", &mk(1, 0, "2"), 4000);
        assert_eq!(cum(&e4, "Anthropic / chat 2 / calls"), 1.0, "per-chat total");
        assert_eq!(cum(&e4, "Anthropic / chat 2 / errors"), 0.0);
    }

    #[test]
    fn join_gates_on_whitelisted() {
        // whitelisted=false -> pop returns meta but caller must not report.
        let mut s = sinks(&["tokens_in"]);
        s.stash(7, "api.anthropic.com".into(), false);
        let m = s.pop(7).expect("stashed");
        assert!(!m.whitelisted);
        assert_eq!(m.host, "api.anthropic.com");
        assert!(s.pop(7).is_none(), "pop removes the entry");
    }

    // ---- batching ----

    #[test]
    fn should_flush_on_byte_threshold() {
        let mut s = sinks(&["tokens_in"]);
        assert!(!s.should_flush(), "empty buffers do not trigger a flush");
        // Accumulate scalar events until the metric buffer crosses ~1MB.
        while s.metrics_bytes < MAX_PACKET_BYTES {
            s.enqueue_metrics(vec![scalar_event("t", "M", "V", 1.0, 0, 1000)]);
        }
        assert!(
            s.should_flush(),
            "crossing MAX_PACKET_BYTES triggers a size-based flush"
        );
    }

    #[test]
    fn enqueue_respects_byte_cap() {
        let mut s = sinks(&["tokens_in"]);
        // Pre-load the byte counter to just under the cap (in-module private
        // access), then prove the next event is refused and counted, not added.
        s.metrics_bytes = MAX_REQ_BYTES - 5;
        s.enqueue_metrics(vec![scalar_event("t", "M", "V", 1.0, 0, 1000)]);
        assert!(s.metrics_buf.is_empty(), "event over the cap is refused");
        assert_eq!(s.dropped, 1, "and counted as dropped");
        assert!(s.metrics_bytes <= MAX_REQ_BYTES, "counter never exceeds the cap");
    }

    #[test]
    fn on_event_buffers_without_sending() {
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        // All three sinks on (usage + metrics + aggregator). No client is
        // passed to on_event, so there is no path to the network here.
        let mut s = Sinks::new(
            "task-1".into(),
            true,
            true,
            &["tokens_in".to_string()],
            Some("http://aggregator.example/ingest".into()),
            String::new(),
            String::new(),
        );
        s.on_event(
            &Event::RequestStarted {
                conn_id: 1,
                ts_ms: 1000,
                host: "api.anthropic.com".into(),
                path: "/v1/messages".into(),
                method: "POST".into(),
                whitelisted: true,
                inject_headers: true,
            },
            &mut fwd,
        );
        s.on_event(
            &Event::RequestCompleted {
                conn_id: 1,
                ts_ms: 1000,
                status: Some(200),
                latency_ms: 5,
                bytes_tx: 10,
                bytes_rx: 20,
                tokens_in: 100,
                tokens_out: 50,
                tokens_measured: true,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                tool_calls: 0,
                tool_call_errors: 0,
                tool_call_names: vec![],
                tool_call_error_names: vec![],
                chat_id: None,
                model: Some("claude-haiku-4-5".into()),
            },
            &mut fwd,
        );
        assert!(!s.usage_buf.is_empty(), "usage event buffered");
        assert!(!s.metrics_buf.is_empty(), "metric events buffered");
        assert!(!s.aggregator_buf.is_empty(), "aggregator event buffered");
        assert!(s.usage_bytes > 0 && s.metrics_bytes > 0 && s.aggregator_bytes > 0);
    }

    #[test]
    fn flush_clears_buffers_on_permanent_failure() {
        use crate::descriptor::Descriptor;
        // No creds: ensure_token can't build an auth header, so the sink POSTs
        // fail fast (no network, no retry loop) — i.e. a permanent/terminal
        // failure. The flush must DROP the batch (clear buffer + reset bytes),
        // NOT retain it: transient failures are retried forever inside the
        // client, so an Err escaping to flush means "don't re-send".
        let d = Descriptor::from_json_str(
            r#"{"api_server":"https://127.0.0.1:1/","task_id":"t"}"#,
        )
        .expect("parse descriptor");
        let mut client = ClearmlClient::from_descriptor(&d);
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        s.enqueue_metrics(vec![scalar_event("t", "M", "V", 1.0, 0, 1000)]);
        s.enqueue_usage(vec![json!({"id": "x", "usage": 1})]);
        assert!(s.metrics_bytes > 0 && s.usage_bytes > 0);

        s.flush(&mut client, &mut fwd);

        assert!(s.metrics_buf.is_empty(), "metrics dropped on permanent failure");
        assert!(s.usage_buf.is_empty(), "usage dropped on permanent failure");
        assert_eq!(s.metrics_bytes, 0, "metrics bytes reset");
        assert_eq!(s.usage_bytes, 0, "usage bytes reset");
    }
}
