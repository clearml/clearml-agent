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
//!     endpoint as a single event carrying the DISJOINT input split
//!     `prompt_tokens` (fresh) + `cache_read_tokens` + `cache_write_tokens`, plus
//!     `completion_tokens`, tagged `source="external"`, plus the model and
//!     provider. Skipped when both token counts are zero, or when the request
//!     resolved no model (a non-LLM call).
//!   * task-metrics — per-SECOND usage scalars to this task's SCALARS tab via
//!     `events.add_batch` (`training_stats_scalar`). The point series (tokens
//!     in/out, cache read/write, latency, bytes) are a CONTINUOUS
//!     per-second time series: each second reports that second's accumulated
//!     traffic (tokens / bytes summed, latency averaged), and a
//!     second with NO traffic reports 0 — so every series is an uninterrupted
//!     rate-over-time line rather than a sparse point-per-call. One series per
//!     configured field, variant = provider (+ model when known, + chat ordinal
//!     when the request identified its conversation). Every series opens with a
//!     single 0 one second BEFORE its first captured second, so its line rises
//!     from 0 instead of jumping to the first real value. The x-axis (`iter`) is
//!     the whole-second offset from that origin (so the axis is elapsed wall-time
//!     in seconds), and `timestamp` carries that second's wall-time for the
//!     SCALARS Wall-Time/Relative axis (and `xaxis=iso_time` in an embedded
//!     Report). Idle seconds are 0-filled by a ~1 Hz clock tick
//!     driven from the reporter loop (`on_tick`), independent of traffic;
//!     requests advance the same clock (`advance_rate_clock`). A series 0-fills
//!     only while active — a variant idle past `RATE_IDLE_RETIRE_SECS` retires
//!     (stops emitting and is pruned), so a finished chat can't 0-fill forever
//!     (chat ids are never reused). It re-appears as a new segment if it resumes.
//!
//!     Three families stay PER-REQUEST (not per-second, not 0-filled), each on its
//!     own charts against a per-captured-request enumerator (`per_request_events`):
//!     each token field's "(cumulative)" running total, the cumulative request
//!     count ("LLM Requests (cumulative)", a running total per provider/model), and tool
//!     activity - a
//!     SIGNAL line (0 idle / +1 used / -1 errored, errors dominate; aggregate per
//!     provider/chat and per tool name, so failures read as downward spikes) plus
//!     a CUMULATIVE chart (a monotonically-increasing running total of tool calls
//!     and of tool-call errors, a calls line + an errors line per provider/chat).
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

/// Backstop on how many seconds a single `advance_rate_clock` call iterates when
/// 0-filling. The 1 Hz tick advances one second at a time, so this never bites in
/// normal operation; it only bounds the loop on a pathological jump (machine
/// sleep, a stalled reporter loop). Emission per series is already bounded by
/// `RATE_IDLE_RETIRE_SECS` (a variant stops 0-filling once idle past the grace
/// window), so this is a loop-iteration guard, not the emission bound.
const MAX_RATE_FILL_SECS: u64 = 3600;

/// Grace window (seconds) a series keeps reporting 0 through idle before it is
/// RETIRED from the per-second 0-fill. Because a chat id is a per-task ordinal
/// that is never reused, every conversation that ever ran would otherwise stay a
/// permanent 0-filled line — a coding-agent task with hundreds of chats would
/// emit hundreds of flat-zero series forever. Retiring an idle series bounds the
/// 0-fill to recently-active conversations: gaps up to this window stay
/// continuous (a normal turn pause reads as 0), but a finished chat stops
/// emitting and is pruned. It re-appears as a new segment if it sends traffic
/// again.
const RATE_IDLE_RETIRE_SECS: u64 = 120;

/// (field key, scalar metric title). Every field here EXCEPT the tool fields and
/// `requests` is a per-second POINT series (see `advance_rate_clock`): each
/// second reports that second's summed count (tokens/bytes) or averaged latency,
/// 0 when idle. Three keys stay valid config entries here but are NOT per-second
/// points, so their titles below are only documentation: `tool_calls` /
/// `tool_call_errors` (either enables tool metering) drive the tool SIGNAL series
/// instead, and `requests` drives a CUMULATIVE running total per provider/model
/// (both via `per_request_events`) - a request-per-second rate is an
/// uninformative 0/1 line, so requests accumulate instead.
const FIELD_SPECS: &[(&str, &str)] = &[
    ("tokens_in", "LLM Input Tokens"),
    ("tokens_out", "LLM Output Tokens"),
    // Prompt-cache split. These three input series are DISJOINT: "LLM Input
    // Tokens" is FRESH (non-cached) input only, and cache-read / cache-write are
    // the cached buckets, so the three sum to the billable input total. The usage
    // event reports the SAME disjoint split (see usage_event / accumulate_rate).
    // The cache buckets are populated for Anthropic, OpenAI, and native Gemini; 0
    // for providers/requests that report no cache breakdown.
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

/// The request-count field. Like the tool fields it is a valid config key but is
/// filtered out of the per-second loop: it plots as a CUMULATIVE running total
/// per provider/model (see `per_request_events`), one climbing line, rather than
/// a per-second rate.
const REQUESTS_FIELD: &str = "requests";

/// Title of the cumulative request-count chart - a monotonically-increasing line
/// per provider/model. Carries the "(cumulative)" suffix like the token and tool
/// cumulative charts (`CUMULATIVE_SPECS`, `TOOL_CUMULATIVE_METRIC`); the
/// `FIELD_SPECS` "LLM Requests" entry is only the field's doc name (never
/// emitted), so this is the real chart title.
const REQUESTS_CUMULATIVE_METRIC: &str = "LLM Requests (cumulative)";
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
/// per-second value on its `FIELD_SPECS` chart and a running total over the run
/// here, so "how much did this call cost" and "how much has this run spent so
/// far" are both one glance away. The cumulative value sums the SAME per-request
/// token counts the point series buckets per second — so the four cumulative
/// input/output charts stay the disjoint split described on `FIELD_SPECS`. These
/// series accumulate across ALL of a model's calls: the chat dimension is dropped
/// from the variant (a per-chat total would restart on every new conversation,
/// which for one-shot calls is a chart of single points), leaving one climbing
/// line per provider/model. Unlike the point series these are PER-REQUEST (a
/// running total is meaningless to 0-fill), so they ride the request enumerator.
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
    /// Prompt-cache split of `tokens_in` (contained within it): cache-read and
    /// cache-write token counts, each its own scalar series. Populated for
    /// Anthropic, OpenAI, and native Gemini; 0 for non-cache requests.
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

/// One second's accumulated traffic for a single series variant — the raw
/// material for the per-second point series. Summed for additive fields
/// (tokens/bytes); latency is kept as sum+count so it can be AVERAGED
/// (a per-second latency SUM would be meaningless — a mean latency for the
/// second's requests is the useful signal). A `Default` (all-zero) bucket is the
/// value a series reports for a second it had no traffic.
#[derive(Default)]
struct RateBucket {
    fresh_in: f64,
    out: f64,
    cache_read: f64,
    cache_write: f64,
    bytes_tx: f64,
    bytes_rx: f64,
    latency_sum: f64,
    latency_count: u64,
}

impl RateBucket {
    /// The per-second value for one configured field. Additive fields return the
    /// second's sum; `latency_ms` returns the mean over the second's requests (0
    /// when none). Fields that never reach the per-second loop (the tool fields
    /// and `requests`, which are filtered out of it) return 0.
    fn value(&self, field: &str) -> f64 {
        match field {
            "tokens_in" => self.fresh_in,
            "tokens_out" => self.out,
            "cache_read_tokens" => self.cache_read,
            "cache_write_tokens" => self.cache_write,
            "bytes_tx" => self.bytes_tx,
            "bytes_rx" => self.bytes_rx,
            "latency_ms" => {
                if self.latency_count > 0 {
                    self.latency_sum / self.latency_count as f64
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
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
    /// Per-request x-axis enumerator for the PER-REQUEST charts only (cumulative
    /// token totals + tool signal/cumulative): incremented once per captured
    /// request so their points share one monotonic, capture-ordered axis. The
    /// per-second point series do NOT use this — they ride the wall-second axis
    /// (`rate_axis_start`). The Wall-Time view rides `timestamp` for both.
    metrics_seq: u64,
    /// Origin of the per-second point-series x-axis (`iter = second -
    /// rate_axis_start`): one second BEFORE the first captured second, so each
    /// series' lead-in 0 sits at `iter = 0` and its first real second at
    /// `iter = 1`. `None` until the first contributing request starts the clock.
    rate_axis_start: Option<u64>,
    /// The wall-second the still-open rate bucket accumulates into. `None` until
    /// the clock starts; advancing past it emits that second + 0-fills the gap.
    rate_cur_sec: Option<u64>,
    /// Traffic accumulated for `rate_cur_sec`, keyed by series variant
    /// (provider/model/chat). Emptied each time the clock advances a second.
    rate_open: HashMap<String, RateBucket>,
    /// Currently-active series variants. A second with no traffic still emits a 0
    /// for each of these, keeping every line continuous over time — until the
    /// variant is idle past `RATE_IDLE_RETIRE_SECS`, when it is pruned from here
    /// (see `advance_rate_clock`) so a finished chat stops 0-filling.
    rate_series: HashSet<String>,
    /// Last wall-second each `rate_series` variant carried real traffic. Drives
    /// the idle-retirement grace window (a variant 0-fills only through
    /// `last_active + RATE_IDLE_RETIRE_SECS`).
    rate_last_active: HashMap<String, u64>,
    /// Variants that have already emitted their first point. A variant emits a
    /// single 0 lead-in (see `rate_lead_in`) the first time it appears here so its
    /// line rises from 0; retirement drops it, so a resumed segment leads in again.
    rate_emitted: HashSet<String>,
    /// Every tool name seen so far this task. Each gets a continuous 0-baseline
    /// line on the per-tool signal graph, so a tool reads as a flat 0 except
    /// where it spikes to +1 (used) / -1 (errored).
    seen_tools: HashSet<String>,
    /// Per-series running cumulative tool-call / tool-call-error totals, keyed by
    /// series variant (provider, or provider+model+chat). These are the two
    /// monotonic lines of the `TOOL_CUMULATIVE_METRIC` chart.
    tool_calls_cum: HashMap<String, u64>,
    tool_call_errors_cum: HashMap<String, u64>,
    /// Running token totals — the values of the `CUMULATIVE_SPECS` charts. Keyed
    /// `field \0 base` (the chat-less series key) so one map covers every token
    /// field without colliding across them, and every chat of a model feeds one
    /// total.
    token_cum: HashMap<String, f64>,
    /// Running request count - the value of the "LLM Requests (cumulative)" chart.
    /// Keyed on the chat-less `base` so a model's chats merge into one climbing
    /// line, matching the token cumulative charts. Incremented once per captured
    /// request.
    requests_cum: HashMap<String, u64>,
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
            metrics_seq: 0,
            rate_axis_start: None,
            rate_cur_sec: None,
            rate_open: HashMap::new(),
            rate_series: HashSet::new(),
            rate_last_active: HashMap::new(),
            rate_emitted: HashSet::new(),
            seen_tools: HashSet::new(),
            tool_calls_cum: HashMap::new(),
            tool_call_errors_cum: HashMap::new(),
            token_cum: HashMap::new(),
            requests_cum: HashMap::new(),
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
                                *cache_read_tokens,
                                *cache_write_tokens,
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
        cache_read: u64,
        cache_write: u64,
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
        let event = self.usage_event(
            &provider, model, tokens_in, tokens_out, cache_read, cache_write, resolve_ts(ts_ms),
        );
        fwd.enqueue_diagnostic(&format!(
            "[SNUG-USAGE] queued report_llm_usage provider={:?} model={:?} \
             prompt_tokens(fresh)={} cache_read={} cache_write={} completion_tokens={}",
            provider, model,
            tokens_in.saturating_sub(cache_read).saturating_sub(cache_write),
            cache_read, cache_write, tokens_out
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
    /// the DISJOINT input split — `prompt_tokens` (fresh, uncached) +
    /// `cache_read_tokens` + `cache_write_tokens` — plus `completion_tokens`, so
    /// the four sum to the billable total for every provider. Tagged
    /// `source="external"` (SNUG meters the task's own egress, not the ClearML
    /// gateway). `model` is the per-request model (the usage coset); `provider` is
    /// the coarse host label. `task`/`user`/`project` attribute the usage —
    /// `user`/`project` are omitted when the agent didn't supply them (the backend
    /// then derives them from the task). No I/O.
    fn usage_event(
        &self,
        provider: &str,
        model: &str,
        tokens_in: u64,
        tokens_out: u64,
        cache_read: u64,
        cache_write: u64,
        ts_ms: u64,
    ) -> Value {
        // FRESH (uncached) input, the same split metric_events plots: prompt +
        // cache_read + cache_write sums back to the billable input total.
        // saturating_sub guards the impossible read + write > tokens_in.
        let fresh = tokens_in.saturating_sub(cache_read).saturating_sub(cache_write);
        let mut ev = serde_json::Map::new();
        ev.insert("timestamp".to_string(), json!(ts_ms));
        ev.insert("source".to_string(), json!("external"));
        ev.insert("model".to_string(), json!(model));
        ev.insert("provider".to_string(), json!(provider));
        ev.insert("prompt_tokens".to_string(), json!(fresh));
        ev.insert("cache_read_tokens".to_string(), json!(cache_read));
        ev.insert("cache_write_tokens".to_string(), json!(cache_write));
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
        // and telemetry resolve a model but carry no usage — they contribute no
        // traffic to the per-second buckets and no per-request point.
        if c.tokens_in == 0 && c.tokens_out == 0 {
            return;
        }
        // Same LLM gate as `usage`: a model-less RequestCompleted is a non-LLM
        // call on a metered host (host-only whitelist match under HTTP/2), not a
        // real completion, so it must not create a series or advance a clock.
        if c.model.as_deref().map(str::is_empty).unwrap_or(true) {
            return;
        }
        let provider = host_to_model_name(host);
        let ts_ms = resolve_ts(ts_ms);
        fwd.enqueue_diagnostic(&format!(
            "[SNUG-METRICS] captured provider={:?} model={:?} chat={:?} in={} out={}",
            provider,
            c.model.as_deref(),
            c.chat_id.as_deref(),
            c.tokens_in,
            c.tokens_out
        ));
        // Point series: fold this request into its wall-second bucket. Advancing
        // the clock to the request's second emits any now-closed seconds,
        // 0-filling idle gaps.
        self.accumulate_rate(&provider, c, ts_ms, fwd);
        // Cumulative token totals + tool signal/cumulative stay per-request.
        let events = self.per_request_events(&provider, c, ts_ms);
        if !events.is_empty() {
            self.enqueue_metrics(events);
        }
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

    /// The series keys for a completed request. `base` = provider, plus the model
    /// when the shim parsed one and it differs from the provider label (a line per
    /// model). `variant` = `base`, plus the chat ordinal when the request
    /// identified its conversation (a line per chat). `base` keys the chat-less
    /// cumulative token totals; `variant` keys the per-second point series and the
    /// tool series. The model segment is omitted when unknown or equal to the
    /// provider label (so an unparsed model keeps the provider-only series, not
    /// "Anthropic / Anthropic").
    fn series_keys(provider: &str, c: &Completed) -> (String, String) {
        let base = match c.model.as_deref() {
            Some(m) if !m.is_empty() && m != provider => format!("{} / {}", provider, m),
            _ => provider.to_string(),
        };
        let variant = match c.chat_id.as_deref() {
            Some(id) if !id.is_empty() => format!("{} / chat {}", base, id),
            _ => base.clone(),
        };
        (base, variant)
    }

    /// The configured point fields — everything except the tool fields (which ride
    /// the signal series) and `requests` (which rides the cumulative running
    /// total), neither a per-second point. These are the series that emit once per
    /// second and 0-fill idle seconds.
    fn rate_generic_fields(&self) -> Vec<&'static str> {
        self.metric_fields
            .iter()
            .copied()
            .filter(|f| !TOOL_FIELDS.contains(f) && *f != REQUESTS_FIELD)
            .collect()
    }

    /// Lead-in events for a series' FIRST emitted second: a single 0 on every
    /// configured field at the second BEFORE `first_sec`, so the chart line rises
    /// from 0 instead of starting at the first real value. Empty once the variant
    /// has already been emitted (tracked in `rate_emitted`); after retirement the
    /// mark is dropped, so a resumed segment leads in again. `axis` is the shared
    /// x-axis origin (one second before the run's first captured second, so the
    /// very first series' lead-in lands at `iter = 0`).
    fn rate_lead_in(
        &mut self,
        variant: &str,
        first_sec: u64,
        axis: u64,
        fields: &[&'static str],
    ) -> Vec<Value> {
        if !self.rate_emitted.insert(variant.to_string()) {
            return Vec::new(); // already has a live segment
        }
        let lead_sec = first_sec.saturating_sub(1);
        let iter = lead_sec.saturating_sub(axis) as i64;
        let ts_ms = lead_sec * 1000;
        fields
            .iter()
            .map(|&field| scalar_event(&self.task_id, metric_title(field), variant, 0.0, iter, ts_ms))
            .collect()
    }

    /// Fold one completed request into the open wall-second's bucket for its
    /// series variant. First advances the per-second clock to the request's second
    /// (emitting the previous open second + 0-filling any idle gap), then
    /// accumulates this request's counts. Registers the variant so idle seconds
    /// 0-fill it too.
    fn accumulate_rate(&mut self, provider: &str, c: &Completed, ts_ms: u64, fwd: &mut LogForwarder) {
        let sec = ts_ms / 1000;
        self.advance_rate_clock(sec, fwd);
        let (_base, variant) = Self::series_keys(provider, c);
        self.rate_series.insert(variant.clone());
        // Mark active this second, resetting the idle-retirement grace window.
        self.rate_last_active.insert(variant.clone(), sec);
        let b = self.rate_open.entry(variant).or_default();
        // FRESH (non-cached) input only, so tokens_in + cache_read + cache_write
        // stay a disjoint split that sums to the billable input total (the same
        // split the cumulative charts carry).
        let fresh = c
            .tokens_in
            .saturating_sub(c.cache_read_tokens)
            .saturating_sub(c.cache_write_tokens);
        b.fresh_in += fresh as f64;
        b.out += c.tokens_out as f64;
        b.cache_read += c.cache_read_tokens as f64;
        b.cache_write += c.cache_write_tokens as f64;
        b.bytes_tx += c.bytes_tx as f64;
        b.bytes_rx += c.bytes_rx as f64;
        b.latency_sum += c.latency_ms as f64;
        b.latency_count += 1;
    }

    /// Advance the per-second point-series clock to `target_sec`: emit the
    /// currently-open second (its accumulated bucket, or 0 for any active series
    /// with no traffic that second), 0-fill every idle second up to but excluding
    /// `target_sec`, then reopen the clock at `target_sec`. A no-op when
    /// `target_sec` is not strictly ahead of the open second — same-second (or
    /// slightly out-of-order) traffic just folds into the open bucket. The first
    /// call anchors the x-axis origin (`rate_axis_start`) and emits nothing.
    ///
    /// A series only 0-fills while it is ACTIVE — within `RATE_IDLE_RETIRE_SECS`
    /// of its last real second; past that it stops emitting and is pruned from
    /// `rate_series` so a finished chat can't 0-fill forever. The second loop is
    /// additionally bounded by `MAX_RATE_FILL_SECS` iterations against a
    /// pathological clock jump.
    ///
    /// The first time a series is emitted it is preceded by a single 0 at the
    /// prior second (its lead-in baseline; see `rate_lead_in`), so every line
    /// rises from 0 rather than starting at its first real value.
    fn advance_rate_clock(&mut self, target_sec: u64, fwd: &mut LogForwarder) {
        let cur = match self.rate_cur_sec {
            None => {
                // First contributing request anchors the axis one second before
                // itself (so each series' lead-in 0 lands at iter 0) and opens the
                // first bucket; there is nothing to emit yet.
                self.rate_axis_start = Some(target_sec.saturating_sub(1));
                self.rate_cur_sec = Some(target_sec);
                return;
            }
            Some(c) => c,
        };
        if target_sec <= cur {
            return;
        }
        let axis = self.rate_axis_start.unwrap_or(cur);
        let fields = self.rate_generic_fields();
        let mut series: Vec<String> = self.rate_series.iter().cloned().collect();
        series.sort(); // stable output order
        let had_real = !self.rate_open.is_empty();
        // A variant 0-fills second `sec` only while still within its grace window.
        let active_at = |last: Option<u64>, sec: u64| -> bool {
            last.map(|l| sec <= l + RATE_IDLE_RETIRE_SECS).unwrap_or(true)
        };
        let mut events: Vec<Value> = Vec::new();
        if !series.is_empty() && !fields.is_empty() {
            // The open second (`cur`) reports its accumulated bucket; an active
            // series with no traffic this second reports 0.
            let iter = (cur - axis) as i64;
            let ts_ms = cur * 1000;
            for variant in &series {
                if !active_at(self.rate_last_active.get(variant).copied(), cur) {
                    continue; // retired: stop emitting
                }
                // First emitted second for this series: lay a single 0 one second
                // earlier so its line rises from 0 instead of the first real value.
                events.extend(self.rate_lead_in(variant, cur, axis, &fields));
                let bucket = self.rate_open.get(variant);
                for &field in &fields {
                    let value = bucket.map(|b| b.value(field)).unwrap_or(0.0);
                    events.push(scalar_event(
                        &self.task_id,
                        metric_title(field),
                        variant,
                        value,
                        iter,
                        ts_ms,
                    ));
                }
            }
            // Idle seconds between `cur` and `target_sec` report 0 for every active
            // series, so each line stays continuous over time. `fill_start` caps
            // the iteration count against a huge jump; the per-variant grace check
            // caps how long any one line 0-fills.
            let fill_start = (cur + 1).max(target_sec.saturating_sub(MAX_RATE_FILL_SECS));
            for sec in fill_start..target_sec {
                let iter = (sec - axis) as i64;
                let ts_ms = sec * 1000;
                for variant in &series {
                    if !active_at(self.rate_last_active.get(variant).copied(), sec) {
                        continue; // retired for this and every later second
                    }
                    for &field in &fields {
                        events.push(scalar_event(
                            &self.task_id,
                            metric_title(field),
                            variant,
                            0.0,
                            iter,
                            ts_ms,
                        ));
                    }
                }
            }
        }
        self.rate_open.clear();
        self.rate_cur_sec = Some(target_sec);
        // Prune variants idle past the grace window so `rate_series` stays bounded
        // to recently-active conversations (chat ids are never reused, so without
        // this the set would grow for the task's whole life).
        let retire_before = target_sec.saturating_sub(RATE_IDLE_RETIRE_SECS);
        {
            let last_active = &self.rate_last_active;
            self.rate_series
                .retain(|v| last_active.get(v).copied().unwrap_or(0) >= retire_before);
        }
        self.rate_last_active.retain(|_, last| *last >= retire_before);
        // Drop the emitted-mark for retired variants so a resumed segment leads in
        // from 0 again (chat ids are never reused, so a variant only "resumes" as
        // a genuinely new segment).
        {
            let series = &self.rate_series;
            self.rate_emitted.retain(|v| series.contains(v));
        }
        if !events.is_empty() {
            // Only announce seconds that carried real traffic; a pure 0-fill tick
            // would otherwise log every idle second.
            if had_real {
                fwd.enqueue_diagnostic(&format!(
                    "[SNUG-METRICS] queued rate scalars sec={} series={} events={}",
                    cur,
                    series.len(),
                    events.len()
                ));
            }
            self.enqueue_metrics(events);
        }
    }

    /// Emit the still-open second's accumulated bucket at drain, so the final
    /// captured second isn't lost waiting for a tick that will never fire. Only
    /// emits when the open bucket holds real traffic (an idle open second needs no
    /// trailing 0-point). No clock advance — the process is exiting.
    pub fn flush_final_rate(&mut self, fwd: &mut LogForwarder) {
        let sec = match self.rate_cur_sec {
            Some(s) if !self.rate_open.is_empty() => s,
            _ => return,
        };
        let axis = self.rate_axis_start.unwrap_or(sec);
        let iter = (sec - axis) as i64;
        let ts_ms = sec * 1000;
        let fields = self.rate_generic_fields();
        let mut variants: Vec<String> = self.rate_open.keys().cloned().collect();
        variants.sort();
        let mut events: Vec<Value> = Vec::new();
        for variant in &variants {
            // A series draining before any tick emitted it still leads in from 0.
            events.extend(self.rate_lead_in(variant, sec, axis, &fields));
            if let Some(b) = self.rate_open.get(variant) {
                for &field in &fields {
                    events.push(scalar_event(
                        &self.task_id,
                        metric_title(field),
                        variant,
                        b.value(field),
                        iter,
                        ts_ms,
                    ));
                }
            }
        }
        self.rate_open.clear();
        if !events.is_empty() {
            fwd.enqueue_diagnostic(&format!(
                "[SNUG-METRICS] queued final rate sec={} events={}",
                sec,
                events.len()
            ));
            self.enqueue_metrics(events);
        }
    }

    /// Build the PER-REQUEST `training_stats_scalar` events for one completed
    /// request: each configured token field's cumulative running total (keyed on
    /// the chat-less `base`, so a model's chats merge into one climbing line), the
    /// cumulative request count ("LLM Requests (cumulative)", also keyed on `base`), plus the
    /// tool signal / cumulative / per-tool series. The point series are NOT
    /// here — they are per-second (see `accumulate_rate`). The x-axis (`iter`) is a
    /// per-captured-request enumerator shared by these charts; `timestamp`
    /// (`ts_ms`) carries the capture wall-time. Mutates the running totals; no I/O.
    fn per_request_events(&mut self, provider: &str, c: &Completed, ts_ms: u64) -> Vec<Value> {
        let (base, variant) = Self::series_keys(provider, c);
        let iter = self.metrics_seq as i64;
        self.metrics_seq += 1;
        let mut events: Vec<Value> = Vec::new();

        // Cumulative token totals. Collect (field, title, per-request value) first
        // so the immutable borrow of `metric_fields` ends before mutating
        // `token_cum`. The value is the SAME per-request count the point series
        // buckets per second, so the totals stay the disjoint input split.
        let cum_inputs: Vec<(&'static str, &'static str, f64)> = self
            .metric_fields
            .iter()
            .filter(|&&field| !TOOL_FIELDS.contains(&field))
            .filter_map(|&field| {
                cumulative_title(field).map(|title| {
                    let value = match field {
                        "tokens_in" => c
                            .tokens_in
                            .saturating_sub(c.cache_read_tokens)
                            .saturating_sub(c.cache_write_tokens)
                            as f64,
                        "tokens_out" => c.tokens_out as f64,
                        "cache_read_tokens" => c.cache_read_tokens as f64,
                        "cache_write_tokens" => c.cache_write_tokens as f64,
                        _ => 0.0,
                    };
                    (field, title, value)
                })
            })
            .collect();
        for (field, title, value) in cum_inputs {
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

        // Cumulative request count: one climbing line per provider/model (keyed on
        // the chat-less `base`, so a model's chats merge into a single running
        // total, matching the token cumulative charts). Requests plot ONLY as this
        // cumulative - a per-second request rate is an uninformative 0/1 signal, so
        // unlike the token fields there is no per-second twin. Emitted once per
        // captured request (+1).
        if self.metric_fields.contains(&REQUESTS_FIELD) {
            let total = {
                let n = self.requests_cum.entry(base.clone()).or_insert(0);
                *n += 1;
                *n
            };
            events.push(scalar_event(
                &self.task_id,
                REQUESTS_CUMULATIVE_METRIC,
                &base,
                total as f64,
                iter,
                ts_ms,
            ));
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
        events
    }

    /// The ~1 Hz idle tick, called from the reporter loop each iteration with the
    /// current wall-time. Advances the per-second clock so idle seconds 0-fill
    /// even when no traffic arrives. A no-op until the first request has started
    /// the clock, and when task-metrics are off.
    pub fn on_tick(&mut self, now_ms: u64, fwd: &mut LogForwarder) {
        if !self.report_metrics || self.rate_cur_sec.is_none() {
            return;
        }
        self.advance_rate_clock(now_ms / 1000, fwd);
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

/// Running-total chart title for a field, or `None` for fields with no
/// `CUMULATIVE_SPECS` twin. Latency and bytes are per-second rate signals that
/// don't accumulate into a meaningful running total; `requests` DOES plot a
/// running total but via its own dedicated series (`per_request_events` /
/// `requests_cum`, titled "LLM Requests (cumulative)"), not a `CUMULATIVE_SPECS` twin, so it
/// returns `None` here too.
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

    /// An all-zero `Completed`; tests set the fields they care about.
    fn completed() -> Completed {
        Completed {
            tokens_in: 0,
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
            chat_id: None,
            model: None,
        }
    }

    /// Drain the buffered scalar events (and reset the byte counter) so a test can
    /// assert on what a clock advance / tick / drain emitted, then continue.
    fn take_metrics(s: &mut Sinks) -> Vec<Value> {
        s.metrics_bytes = 0;
        std::mem::take(&mut s.metrics_buf)
    }

    /// The float value of the series (metric, variant) at its LATEST iter - the
    /// current value, past the single 0 lead-in every series opens with. Panics
    /// if absent so a missing series fails loudly.
    fn scalar_value(evs: &[Value], metric: &str, variant: &str) -> f64 {
        evs.iter()
            .filter(|e| e["metric"] == metric && e["variant"] == variant)
            .max_by_key(|e| e["iter"].as_i64().unwrap_or(i64::MIN))
            .and_then(|e| e["value"].as_f64())
            .unwrap_or_else(|| panic!("scalar {metric} / {variant} missing"))
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
        // No cache -> fresh == tokens_in, both cache buckets 0.
        let e = s.usage_event("Anthropic", "claude-haiku-4-5", 19, 42, 0, 0, 1000);
        assert_eq!(e["timestamp"], 1000);
        assert_eq!(e["source"], "external");
        assert_eq!(e["provider"], "Anthropic");
        assert_eq!(e["model"], "claude-haiku-4-5");
        // Disjoint input split + output in one event, renamed per the schema.
        assert_eq!(e["prompt_tokens"], 19);
        assert_eq!(e["cache_read_tokens"], 0);
        assert_eq!(e["cache_write_tokens"], 0);
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
        let e = s.usage_event("OpenAI", "gpt-4o", 1, 2, 0, 0, 1000);
        assert_eq!(e["user"], "user-9");
        assert_eq!(e["project"], "proj-7");
    }

    #[test]
    fn usage_event_splits_prompt_into_fresh_and_cache_buckets() {
        // Anthropic-style: tokens_in is the billable total; prompt_tokens is the
        // FRESH remainder and the two cache buckets ride their own fields, so
        // prompt + read + write == tokens_in (the disjoint contract).
        let s = sinks(&["tokens_in"]);
        let e = s.usage_event("Anthropic", "claude-sonnet-4-5", 45305, 13, 45000, 300, 1000);
        assert_eq!(e["prompt_tokens"], 5, "fresh = 45305 - 45000 - 300");
        assert_eq!(e["cache_read_tokens"], 45000);
        assert_eq!(e["cache_write_tokens"], 300);
        assert_eq!(e["completion_tokens"], 13);
        let sum = e["prompt_tokens"].as_u64().unwrap()
            + e["cache_read_tokens"].as_u64().unwrap()
            + e["cache_write_tokens"].as_u64().unwrap();
        assert_eq!(sum, 45305, "buckets are disjoint and sum to the input total");
    }

    #[test]
    fn usage_event_openai_subset_yields_fresh() {
        // OpenAI: cached_tokens is a subset of prompt_tokens, so tokens_in (=
        // prompt_tokens, cache-inclusive) minus cache_read gives fresh.
        let s = sinks(&["tokens_in"]);
        let e = s.usage_event("OpenAI", "gpt-4o-mini", 13219, 1, 13184, 0, 1000);
        assert_eq!(e["prompt_tokens"], 35, "fresh = 13219 - 13184");
        assert_eq!(e["cache_read_tokens"], 13184);
        assert_eq!(e["cache_write_tokens"], 0);
    }

    #[test]
    fn usage_event_clamps_when_cache_exceeds_input() {
        // Defensive: an impossible read + write > tokens_in must not underflow;
        // prompt_tokens clamps to 0.
        let s = sinks(&["tokens_in"]);
        let e = s.usage_event("OpenAI", "gpt-4o", 10, 2, 8, 5, 1000);
        assert_eq!(e["prompt_tokens"], 0);
        assert_eq!(e["cache_read_tokens"], 8);
        assert_eq!(e["cache_write_tokens"], 5);
    }

    #[test]
    fn usage_reports_cache_through_on_event() {
        // End-to-end through on_event: a RequestCompleted carrying cache tokens
        // yields a queued usage event with the disjoint split (proves the wiring,
        // not just the pure builder).
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
                tokens_in: 45305,
                tokens_out: 13,
                tokens_measured: true,
                cache_read_tokens: 45000,
                cache_write_tokens: 300,
                tool_calls: 0,
                tool_call_errors: 0,
                tool_call_names: vec![],
                tool_call_error_names: vec![],
                chat_id: None,
                model: Some("claude-sonnet-4-5".into()),
            },
            &mut fwd,
        );
        let e = s.usage_buf.last().expect("usage event buffered");
        assert_eq!(e["prompt_tokens"], 5);
        assert_eq!(e["cache_read_tokens"], 45000);
        assert_eq!(e["cache_write_tokens"], 300);
        assert_eq!(e["completion_tokens"], 13);
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
        // `count_tokens` pre-flight, telemetry) must not start the per-second
        // clock, create a series, or advance the request enumerator; a real
        // completion still does.
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
        assert!(s.metrics_buf.is_empty(), "zero-token request buffers no scalars");
        assert_eq!(s.metrics_seq, 0, "request enumerator not advanced");
        assert!(s.rate_cur_sec.is_none(), "per-second clock not started");

        s.on_event(&started(2, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(2, Some("claude-haiku-4-5"), 10, 5), &mut fwd);
        // The real completion advances the request enumerator (its cumulative
        // token scalars) and starts the per-second clock (the point series flush
        // on the next tick).
        assert_eq!(s.metrics_seq, 1, "request enumerator advanced once");
        assert!(s.rate_cur_sec.is_some(), "per-second clock started");
        assert!(!s.metrics_buf.is_empty(), "cumulative token scalars buffered");
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
        // model-less RequestCompleted (a non-LLM call on a metered host) creates
        // no series, doesn't start the per-second clock, and doesn't advance the
        // request enumerator.
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
        assert!(s.metrics_buf.is_empty(), "model-less request buffers nothing");
        assert_eq!(s.metrics_seq, 0, "request enumerator not advanced");
        assert!(s.rate_cur_sec.is_none(), "per-second clock not started");

        // With a model -> the request enumerator advances and the clock starts.
        s.on_event(&started(2, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(2, Some("claude-haiku-4-5"), 100, 20), &mut fwd);
        assert!(!s.metrics_buf.is_empty(), "a request with a model is buffered");
        assert_eq!(s.metrics_seq, 1, "request enumerator advanced once");
        assert!(s.rate_cur_sec.is_some(), "per-second clock started");
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
        // Request at second 1; the point series flushes on the next-second tick.
        s.on_event(&rc(1, Some("claude-haiku-4-5"), 100, 20), &mut fwd);

        let u = s.usage_buf.last().expect("usage event buffered");
        assert_eq!(u["model"], "claude-haiku-4-5", "usage carries the model");
        assert_eq!(u["provider"], "Anthropic");

        // The cumulative token scalar buffers immediately (per-request).
        let cum = s
            .metrics_buf
            .iter()
            .find(|e| e["metric"] == "LLM Input Tokens (cumulative)")
            .expect("cumulative token scalar buffered");
        assert_eq!(cum["variant"], "Anthropic / claude-haiku-4-5");

        // The per-second point series flushes once the clock advances past its
        // second; its variant carries the model dimension too.
        s.on_tick(2000, &mut fwd);
        let point = s
            .metrics_buf
            .iter()
            .find(|e| e["metric"] == "LLM Input Tokens")
            .expect("per-second point scalar buffered after the tick");
        assert_eq!(
            point["variant"], "Anthropic / claude-haiku-4-5",
            "point series variant carries the model dimension"
        );
    }

    #[test]
    fn series_keys_base_and_variant() {
        // `base` = provider (+ model when known); `variant` = base (+ chat). The
        // point series and tool series key on `variant`; the cumulative totals on
        // the chat-less `base`.
        let mk = |model: Option<&str>, chat: Option<&str>| {
            let mut c = completed();
            c.model = model.map(|m| m.to_string());
            c.chat_id = chat.map(|c| c.to_string());
            c
        };
        let (base, variant) = Sinks::series_keys("Anthropic", &mk(Some("claude-haiku-4-5"), None));
        assert_eq!(base, "Anthropic / claude-haiku-4-5");
        assert_eq!(variant, "Anthropic / claude-haiku-4-5");
        // Model + chat -> chat only in the variant.
        let (base, variant) = Sinks::series_keys("Anthropic", &mk(Some("claude-haiku-4-5"), Some("3")));
        assert_eq!(base, "Anthropic / claude-haiku-4-5");
        assert_eq!(variant, "Anthropic / claude-haiku-4-5 / chat 3");
        // Model unknown -> provider only; chat still applies to the variant.
        let (base, variant) = Sinks::series_keys("OpenAI", &mk(None, Some("2")));
        assert_eq!(base, "OpenAI");
        assert_eq!(variant, "OpenAI / chat 2");
        // Model equal to the provider label -> not duplicated.
        let (base, variant) = Sinks::series_keys("Anthropic", &mk(Some("Anthropic"), None));
        assert_eq!(base, "Anthropic");
        assert_eq!(variant, "Anthropic");
    }

    #[test]
    fn rate_point_series_split_fresh_cache() {
        // The per-second point split is DISJOINT: "LLM Input Tokens" is FRESH only
        // (tokens_in - cache_read - cache_write); cache read/write are their own
        // series; the three sum to the billable input total. The request lands in
        // second 1 and flushes on the next-second tick.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in", "cache_read_tokens", "cache_write_tokens"]);
        let mut c = completed();
        c.tokens_in = 45302;
        c.tokens_out = 13;
        c.cache_read_tokens = 45000;
        c.cache_write_tokens = 300;
        c.model = Some("claude-sonnet-4-5".into());
        s.accumulate_rate("Anthropic", &c, 1000, &mut fwd);
        s.on_tick(2000, &mut fwd);
        let e = take_metrics(&mut s);
        let v = |m: &str| scalar_value(&e, m, "Anthropic / claude-sonnet-4-5");
        assert_eq!(v("LLM Input Tokens"), 2.0, "fresh only: 45302 - 45000 - 300");
        assert_eq!(v("LLM Cache Read Tokens"), 45000.0);
        assert_eq!(v("LLM Cache Write Tokens"), 300.0);
        // The request's second is iter 1 (iter 0 is the lead-in baseline, one
        // second earlier); its point carries the real value at its wall-time.
        let pt = e
            .iter()
            .find(|x| x["metric"] == "LLM Input Tokens" && x["iter"] == 1)
            .unwrap();
        assert_eq!(pt["value"], 2.0);
        assert_eq!(pt["timestamp"], 1000);
        assert_eq!(pt["task"], "task-1");
        // The series opens with a single 0 one second before, so the line rises
        // from 0 instead of jumping to the first real value.
        let lead = e
            .iter()
            .find(|x| x["metric"] == "LLM Input Tokens" && x["iter"] == 0)
            .unwrap();
        assert_eq!(lead["value"], 0.0);
        assert_eq!(lead["timestamp"], 0);
    }

    #[test]
    fn rate_sums_within_second_latency_averaged() {
        // Two requests in the same wall-second: additive fields sum, latency is
        // averaged over them.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in", "tokens_out", "bytes_tx", "bytes_rx", "latency_ms"]);
        let mk = |tin, tout, tx, rx, lat| {
            let mut c = completed();
            c.tokens_in = tin;
            c.tokens_out = tout;
            c.bytes_tx = tx;
            c.bytes_rx = rx;
            c.latency_ms = lat;
            c.model = Some("m".into());
            c
        };
        s.accumulate_rate("Anthropic", &mk(100, 10, 5, 7, 100), 1000, &mut fwd);
        s.accumulate_rate("Anthropic", &mk(50, 20, 5, 3, 300), 1500, &mut fwd);
        s.on_tick(2000, &mut fwd);
        let e = take_metrics(&mut s);
        let v = |m: &str| scalar_value(&e, m, "Anthropic / m");
        assert_eq!(v("LLM Input Tokens"), 150.0, "tokens summed");
        assert_eq!(v("LLM Output Tokens"), 30.0);
        assert_eq!(v("LLM Bytes Sent"), 10.0, "bytes summed");
        assert_eq!(v("LLM Bytes Received"), 10.0);
        assert_eq!(v("LLM Latency (ms)"), 200.0, "averaged: (100 + 300) / 2");
    }

    #[test]
    fn rate_zero_fills_idle_seconds() {
        // THE core behavior: after a request in second 1, idle seconds report 0
        // for every configured series so each line is continuous over time.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mut c = completed();
        c.tokens_in = 100;
        c.model = Some("m".into());
        s.accumulate_rate("Anthropic", &c, 1000, &mut fwd); // second 1, clock starts

        // Tick to second 2: emits the lead-in 0 (iter 0) + second 1's real data
        // (iter 1); the axis origin sits one second before the first request.
        s.on_tick(2000, &mut fwd);
        let e1 = take_metrics(&mut s);
        assert_eq!(scalar_value(&e1, "LLM Input Tokens", "Anthropic / m"), 100.0);
        let p1 = e1
            .iter()
            .find(|x| x["metric"] == "LLM Input Tokens" && x["iter"] == 1)
            .unwrap();
        assert_eq!(p1["value"], 100.0);
        assert_eq!(p1["timestamp"], 1000);
        // The lead-in 0 baseline one second earlier.
        let lead = e1
            .iter()
            .find(|x| x["metric"] == "LLM Input Tokens" && x["iter"] == 0)
            .unwrap();
        assert_eq!(lead["value"], 0.0);

        // Tick to second 4: seconds 2 and 3 each report 0 for both series.
        s.on_tick(4000, &mut fwd);
        let e2 = take_metrics(&mut s);
        let zeros: Vec<&Value> = e2.iter().filter(|x| x["metric"] == "LLM Input Tokens").collect();
        assert_eq!(zeros.len(), 2, "seconds 2 and 3 each get a point");
        assert!(zeros.iter().all(|z| z["value"] == 0.0), "idle seconds report 0");
        assert_eq!(zeros[0]["iter"], 2);
        assert_eq!(zeros[0]["timestamp"], 2000);
        assert_eq!(zeros[1]["iter"], 3);
        assert_eq!(zeros[1]["timestamp"], 3000);
    }

    #[test]
    fn rate_zero_fills_every_known_series() {
        // An idle second 0-fills EVERY series seen so far (all chats), so no line
        // drops out during a lull.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mk = |chat: &str, tin: u64| {
            let mut c = completed();
            c.chat_id = Some(chat.into());
            c.tokens_in = tin;
            c.model = Some("m".into());
            c
        };
        s.accumulate_rate("Anthropic", &mk("1", 10), 1000, &mut fwd);
        s.accumulate_rate("Anthropic", &mk("2", 20), 1500, &mut fwd);
        s.on_tick(2000, &mut fwd); // flush second 1 (both chats)
        let _ = take_metrics(&mut s);
        // Second 2 is idle: both chat series must emit a 0.
        s.on_tick(3000, &mut fwd);
        let e = take_metrics(&mut s);
        assert_eq!(scalar_value(&e, "LLM Input Tokens", "Anthropic / m / chat 1"), 0.0);
        assert_eq!(scalar_value(&e, "LLM Input Tokens", "Anthropic / m / chat 2"), 0.0);
    }

    #[test]
    fn requests_cumulative_per_base() {
        // "requests" plots a CUMULATIVE running total per provider/model, not a
        // per-second count: every captured request adds 1 to the base line, and a
        // model's separate chats feed the SAME total (the chat dimension is
        // dropped from the key, like the token cumulative charts). It rides
        // per_request_events, not the per-second rate path.
        let mut s = sinks(&["requests"]);
        let mk = |chat: &str| {
            let mut c = completed();
            c.chat_id = Some(chat.into());
            c.tokens_in = 10;
            c.model = Some("m".into());
            c
        };
        // First request on chat 1: the base line reads 1, keyed on the chat-less
        // base (never on a per-chat variant).
        let e1 = s.per_request_events("Anthropic", &mk("1"), 1000);
        assert_eq!(scalar_value(&e1, "LLM Requests (cumulative)", "Anthropic / m"), 1.0);
        assert!(
            !e1.iter().any(|x| x["variant"] == "Anthropic / m / chat 1"),
            "requests cumulative is keyed on base, not per chat"
        );
        // A second request on the same chat climbs the total to 2.
        let e2 = s.per_request_events("Anthropic", &mk("1"), 2000);
        assert_eq!(scalar_value(&e2, "LLM Requests (cumulative)", "Anthropic / m"), 2.0);
        // A different chat of the same model feeds the SAME running total.
        let e3 = s.per_request_events("Anthropic", &mk("2"), 3000);
        assert_eq!(scalar_value(&e3, "LLM Requests (cumulative)", "Anthropic / m"), 3.0, "chats share one total");
        // A different model is its own climbing line, starting at 1.
        let mut c = mk("1");
        c.model = Some("m2".into());
        let e4 = s.per_request_events("Anthropic", &c, 4000);
        assert_eq!(scalar_value(&e4, "LLM Requests (cumulative)", "Anthropic / m2"), 1.0, "per-model total");

        // And it is NOT on the per-second rate path: the rate loop emits no
        // "LLM Requests (cumulative)" point series.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        s.accumulate_rate("Anthropic", &mk("1"), 5000, &mut fwd);
        s.on_tick(6000, &mut fwd);
        let rate = take_metrics(&mut s);
        assert!(
            !rate.iter().any(|x| x["metric"] == "LLM Requests (cumulative)"),
            "requests does not emit a per-second point series"
        );
    }

    #[test]
    fn rate_idle_fill_bounded_by_grace() {
        // A long idle (or a clock jump) must not enqueue an unbounded 0-fill: a
        // series 0-fills only through its grace window, then retires. The real
        // second survives and the total point count is bounded by the grace window.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mut c = completed();
        c.tokens_in = 5;
        c.model = Some("m".into());
        s.accumulate_rate("Anthropic", &c, 1000, &mut fwd); // second 1, last_active=1
        // Jump well past the grace window (but within the loop backstop so the cap
        // isn't what bounds it — the grace window is).
        let jump_sec = 1 + RATE_IDLE_RETIRE_SECS + 50;
        s.on_tick(jump_sec * 1000, &mut fwd);
        let e = take_metrics(&mut s);
        let points: Vec<&Value> = e.iter().filter(|x| x["metric"] == "LLM Input Tokens").collect();
        // Lead-in 0 + second 1 (value 5) + exactly RATE_IDLE_RETIRE_SECS 0-fill
        // seconds (2..=1+grace).
        assert_eq!(points.len(), RATE_IDLE_RETIRE_SECS as usize + 2, "0-fill bounded by the grace window");
        assert!(points.iter().any(|p| p["value"] == 5.0), "the real second survives");
        assert_eq!(
            points.iter().filter(|p| p["value"] == 0.0).count(),
            RATE_IDLE_RETIRE_SECS as usize + 1,
            "the lead-in 0 plus exactly the grace window of 0-fills, then it stops"
        );
        // The idle series is retired from the active set.
        assert!(!s.rate_series.contains("Anthropic / m"), "idle series pruned");
        assert!(!s.rate_last_active.contains_key("Anthropic / m"));
    }

    #[test]
    fn rate_retired_series_reappears_on_resumption() {
        // A retired chat that resumes re-registers and 0-fills again (a new
        // segment after the idle gap).
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mk = |tin: u64| {
            let mut c = completed();
            c.tokens_in = tin;
            c.model = Some("m".into());
            c.chat_id = Some("1".into());
            c
        };
        s.accumulate_rate("Anthropic", &mk(10), 1000, &mut fwd); // second 1
        let far = 1 + RATE_IDLE_RETIRE_SECS + 50;
        s.on_tick(far * 1000, &mut fwd); // retires chat 1
        assert!(!s.rate_series.contains("Anthropic / m / chat 1"), "chat retired");
        let _ = take_metrics(&mut s);

        // The chat resumes much later: it is registered again and its bucket flushes.
        let resume = far + 100;
        s.accumulate_rate("Anthropic", &mk(7), resume * 1000, &mut fwd);
        assert!(s.rate_series.contains("Anthropic / m / chat 1"), "resumed chat re-registered");
        s.on_tick((resume + 1) * 1000, &mut fwd);
        let e = take_metrics(&mut s);
        assert_eq!(scalar_value(&e, "LLM Input Tokens", "Anthropic / m / chat 1"), 7.0);
    }

    #[test]
    fn rate_active_chat_zero_fills_through_short_gap() {
        // A gap SHORTER than the grace window keeps the line continuous at 0 (a
        // normal conversation turn pause must not retire the chat).
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mk = |tin: u64| {
            let mut c = completed();
            c.tokens_in = tin;
            c.model = Some("m".into());
            c.chat_id = Some("1".into());
            c
        };
        s.accumulate_rate("Anthropic", &mk(10), 1000, &mut fwd); // second 1
        // Idle for less than the grace window, then advance.
        let gap = 1 + RATE_IDLE_RETIRE_SECS / 2;
        s.on_tick(gap * 1000, &mut fwd);
        assert!(s.rate_series.contains("Anthropic / m / chat 1"), "still active within grace");
        let e = take_metrics(&mut s);
        // The lead-in 0 + the real second (1, value 10) + a continuous 0-fill of
        // seconds 2..gap.
        let zeros: Vec<&Value> = e
            .iter()
            .filter(|x| x["metric"] == "LLM Input Tokens" && x["value"] == 0.0)
            .collect();
        assert_eq!(zeros.len(), (gap - 1) as usize, "lead-in 0 + continuous 0-fill through the short gap");
        // The chat can still resume seamlessly.
        s.accumulate_rate("Anthropic", &mk(3), gap * 1000, &mut fwd);
        assert_eq!(*s.rate_last_active.get("Anthropic / m / chat 1").unwrap(), gap);
    }

    #[test]
    fn rate_point_series_reported_via_on_event() {
        // Full path: a whitelisted RequestStarted + RequestCompleted through
        // on_event (applying the model/token gates), then a tick flushes the
        // per-second point.
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
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, Some("claude-haiku-4-5"), 100, 20), &mut fwd); // ts_ms=1000 -> second 1
        s.on_tick(2000, &mut fwd);
        let e = take_metrics(&mut s);
        assert_eq!(scalar_value(&e, "LLM Input Tokens", "Anthropic / claude-haiku-4-5"), 100.0);
    }

    #[test]
    fn requests_cumulative_reported_via_on_event() {
        // Full path: successive whitelisted completions through on_event climb the
        // "LLM Requests (cumulative)" line (keyed on provider/model). Unlike the
        // per-second points it needs no tick - it emits per request.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new(
            "task-1".into(),
            false,
            true,
            &["requests".to_string()],
            None,
            String::new(),
            String::new(),
        );
        s.on_event(&started(1, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(1, Some("claude-haiku-4-5"), 100, 20), &mut fwd);
        let e1 = take_metrics(&mut s);
        assert_eq!(scalar_value(&e1, "LLM Requests (cumulative)", "Anthropic / claude-haiku-4-5"), 1.0);
        s.on_event(&started(2, "api.anthropic.com"), &mut fwd);
        s.on_event(&rc(2, Some("claude-haiku-4-5"), 50, 10), &mut fwd);
        let e2 = take_metrics(&mut s);
        assert_eq!(
            scalar_value(&e2, "LLM Requests (cumulative)", "Anthropic / claude-haiku-4-5"),
            2.0,
            "climbs across requests"
        );
    }

    #[test]
    fn rate_series_leads_in_from_zero() {
        // Each series opens with a single 0 one second before its first captured
        // second, so its line rises from 0 instead of the first real value. A
        // second chat that starts later gets its own lead-in too.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mk = |chat: &str, tin: u64| {
            let mut c = completed();
            c.chat_id = Some(chat.into());
            c.tokens_in = tin;
            c.model = Some("m".into());
            c
        };
        // Chat 1's first request is at second 1; the axis origin is second 0.
        s.accumulate_rate("Anthropic", &mk("1", 100), 1000, &mut fwd);
        s.on_tick(2000, &mut fwd);
        let e1 = take_metrics(&mut s);
        let lead1 = e1
            .iter()
            .find(|x| {
                x["metric"] == "LLM Input Tokens"
                    && x["variant"] == "Anthropic / m / chat 1"
                    && x["iter"] == 0
            })
            .expect("chat 1 lead-in at iter 0");
        assert_eq!(lead1["value"], 0.0);
        assert_eq!(lead1["timestamp"], 0, "one second before the first request");
        assert_eq!(scalar_value(&e1, "LLM Input Tokens", "Anthropic / m / chat 1"), 100.0);

        // A different chat begins at second 5: it leads in from second 4, not the
        // run origin, so only its own line dips to 0 there.
        s.accumulate_rate("Anthropic", &mk("2", 50), 5000, &mut fwd);
        s.on_tick(6000, &mut fwd);
        let e2 = take_metrics(&mut s);
        let lead2 = e2
            .iter()
            .find(|x| {
                x["metric"] == "LLM Input Tokens"
                    && x["variant"] == "Anthropic / m / chat 2"
                    && x["value"] == 0.0
            })
            .expect("chat 2 lead-in");
        assert_eq!(lead2["iter"], 4, "chat 2 leads in one second before its first (second 5)");
        assert_eq!(lead2["timestamp"], 4000);
        assert_eq!(scalar_value(&e2, "LLM Input Tokens", "Anthropic / m / chat 2"), 50.0);
        // The lead-in is once-per-segment: chat 1 does not lead in again.
        assert!(
            !e2.iter()
                .any(|x| x["variant"] == "Anthropic / m / chat 1" && x["iter"] == 0),
            "chat 1 does not repeat its lead-in"
        );
    }

    #[test]
    fn per_request_cumulative_token_series() {
        // The cumulative token totals accumulate across ALL of a model's calls
        // (chat dimension dropped from the variant), summing the FRESH input split
        // so the four cumulative charts stay disjoint. NO point series here — those
        // are per-second.
        let mut s = sinks(&["tokens_in", "tokens_out", "cache_read_tokens", "cache_write_tokens"]);
        let mk = |tin, tout, cr, cw, chat: &str| {
            let mut c = completed();
            c.tokens_in = tin;
            c.tokens_out = tout;
            c.cache_read_tokens = cr;
            c.cache_write_tokens = cw;
            c.chat_id = Some(chat.into());
            c
        };
        let e1 = s.per_request_events("Anthropic", &mk(110, 30, 60, 40, "1"), 1000);
        assert_eq!(scalar_value(&e1, "LLM Input Tokens (cumulative)", "Anthropic"), 10.0); // 110-60-40
        assert_eq!(scalar_value(&e1, "LLM Output Tokens (cumulative)", "Anthropic"), 30.0);
        assert_eq!(scalar_value(&e1, "LLM Cache Read Tokens (cumulative)", "Anthropic"), 60.0);
        assert_eq!(scalar_value(&e1, "LLM Cache Write Tokens (cumulative)", "Anthropic"), 40.0);
        assert!(
            !e1.iter().any(|x| x["metric"] == "LLM Input Tokens"),
            "per_request_events emits no point series"
        );

        // Second call adds on top; cache buckets hold flat when unused.
        let e2 = s.per_request_events("Anthropic", &mk(30, 5, 0, 0, "1"), 2000);
        assert_eq!(scalar_value(&e2, "LLM Input Tokens (cumulative)", "Anthropic"), 40.0);
        assert_eq!(scalar_value(&e2, "LLM Output Tokens (cumulative)", "Anthropic"), 35.0);
        assert_eq!(scalar_value(&e2, "LLM Cache Read Tokens (cumulative)", "Anthropic"), 60.0);

        // A different chat of the same model feeds the SAME total (chat-less base).
        let e3 = s.per_request_events("Anthropic", &mk(7, 3, 0, 0, "2"), 3000);
        assert_eq!(scalar_value(&e3, "LLM Input Tokens (cumulative)", "Anthropic"), 47.0);

        // A different model is its own line.
        let mut c = mk(5, 2, 0, 0, "1");
        c.model = Some("claude-opus-4-5".into());
        let e4 = s.per_request_events("Anthropic", &c, 4000);
        assert_eq!(
            scalar_value(&e4, "LLM Input Tokens (cumulative)", "Anthropic / claude-opus-4-5"),
            5.0,
            "per-model total, not folded into the model-less line"
        );
    }

    #[test]
    fn per_request_empty_for_non_token_non_tool_fields() {
        // Latency and bytes have no cumulative twin and aren't tool/request fields,
        // so per_request_events emits nothing for them - they live only on the
        // per-second point charts.
        let mut s = sinks(&["latency_ms", "bytes_tx", "bytes_rx"]);
        let mut c = completed();
        c.tokens_in = 10;
        c.tokens_out = 1;
        c.latency_ms = 42;
        c.bytes_tx = 7;
        c.bytes_rx = 9;
        let e = s.per_request_events("Anthropic", &c, 1000);
        assert!(e.is_empty(), "no cumulative or tool events for these fields");
    }

    #[test]
    fn per_request_iter_advances_across_calls() {
        // The per-request charts (cumulative token totals) ride a single
        // enumerator shared across all series and providers.
        let mut s = sinks(&["tokens_in"]);
        let mut c = completed();
        c.tokens_in = 100;
        let e1 = s.per_request_events("Anthropic", &c, 1000);
        let cum1 = e1.iter().find(|x| x["metric"] == "LLM Input Tokens (cumulative)").unwrap();
        assert_eq!(cum1["iter"], 0);
        assert_eq!(s.metrics_seq, 1);
        let e2 = s.per_request_events("Anthropic", &c, 2000);
        let cum2 = e2.iter().find(|x| x["metric"] == "LLM Input Tokens (cumulative)").unwrap();
        assert_eq!(cum2["iter"], 1, "enumerator advances per request");
        let e3 = s.per_request_events("OpenAI", &c, 3000);
        let cum3 = e3.iter().find(|x| x["metric"] == "LLM Input Tokens (cumulative)").unwrap();
        assert_eq!(cum3["iter"], 2, "enumerator shared across providers");
    }

    #[test]
    fn per_request_per_tool_signal() {
        let mut s = sinks(&["tool_calls", "tool_call_errors"]);
        let mk = |names: Vec<&str>, errs: Vec<&str>| {
            let mut c = completed();
            c.tool_calls = names.len() as u64;
            c.tool_call_errors = errs.len() as u64;
            c.tool_call_names = names.iter().map(|s| s.to_string()).collect();
            c.tool_call_error_names = errs.iter().map(|s| s.to_string()).collect();
            c.chat_id = Some("1".into());
            c
        };
        let per_tool = |events: &[Value], tool: &str| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Tool Calls by Tool" && e["variant"] == tool)
                .and_then(|e| e["value"].as_f64())
                .unwrap_or(f64::NAN)
        };

        // get_weather used cleanly, search's result errored.
        let e1 = s.per_request_events("Anthropic", &mk(vec!["get_weather", "search"], vec!["search"]), 1000);
        assert_eq!(per_tool(&e1, "get_weather"), 1.0, "used tool -> +1");
        assert_eq!(per_tool(&e1, "search"), -1.0, "errored tool -> -1 (dominates)");
        assert!(
            !e1.iter().any(|e| e["metric"] == "LLM Tool Calls"),
            "no raw-count aggregate metric"
        );
        assert!(!e1.iter().any(|e| e["type"] == "plot"), "no plot events");

        // A later request with no tool activity: both already-seen tools sit at
        // the continuous 0 baseline.
        let e2 = s.per_request_events("Anthropic", &mk(vec![], vec![]), 2000);
        assert_eq!(per_tool(&e2, "get_weather"), 0.0, "baseline 0 when unused");
        assert_eq!(per_tool(&e2, "search"), 0.0, "baseline 0 when unused");
    }

    #[test]
    fn per_request_signal_encodes_minus_one_zero_plus_one() {
        // Aggregate signal line: -1 on error (dominant), +1 on a clean tool use,
        // 0 when the request had no tool activity.
        let mut s = sinks(&["tool_calls", "tool_call_errors"]);
        let mk = |calls: u64, errs: u64| {
            let mut c = completed();
            c.tool_calls = calls;
            c.tool_call_errors = errs;
            c
        };
        let signal = |events: &[Value]| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Tool Calls (signal)")
                .and_then(|e| e["value"].as_f64())
                .expect("signal series present")
        };
        assert_eq!(signal(&s.per_request_events("Anthropic", &mk(2, 0), 1000)), 1.0, "clean -> +1");
        assert_eq!(signal(&s.per_request_events("Anthropic", &mk(2, 1), 2000)), -1.0, "error -> -1");
        assert_eq!(signal(&s.per_request_events("Anthropic", &mk(0, 0), 3000)), 0.0, "idle -> 0");
    }

    #[test]
    fn per_request_cumulative_tool_calls() {
        // The cumulative chart carries a `/ calls` line and a `/ errors` line per
        // series, each a monotonically-increasing running total. Tool-free requests
        // repeat the current total, and distinct chats accumulate independently.
        let mut s = sinks(&["tool_calls", "tool_call_errors"]);
        let mk = |calls: u64, errs: u64, chat: &str| {
            let mut c = completed();
            c.tool_calls = calls;
            c.tool_call_errors = errs;
            c.chat_id = Some(chat.into());
            c
        };
        let cum = |events: &[Value], variant: &str| -> f64 {
            events
                .iter()
                .find(|e| e["metric"] == "LLM Tool Calls (cumulative)" && e["variant"] == variant)
                .and_then(|e| e["value"].as_f64())
                .unwrap_or(f64::NAN)
        };

        let e1 = s.per_request_events("Anthropic", &mk(2, 1, "1"), 1000);
        assert_eq!(cum(&e1, "Anthropic / chat 1 / calls"), 2.0);
        assert_eq!(cum(&e1, "Anthropic / chat 1 / errors"), 1.0);

        let e2 = s.per_request_events("Anthropic", &mk(2, 0, "1"), 2000);
        assert_eq!(cum(&e2, "Anthropic / chat 1 / calls"), 4.0, "calls accumulate");
        assert_eq!(cum(&e2, "Anthropic / chat 1 / errors"), 1.0, "errors hold flat");

        let e3 = s.per_request_events("Anthropic", &mk(0, 0, "1"), 3000);
        assert_eq!(cum(&e3, "Anthropic / chat 1 / calls"), 4.0, "flat through idle");
        assert_eq!(cum(&e3, "Anthropic / chat 1 / errors"), 1.0, "flat through idle");

        let e4 = s.per_request_events("Anthropic", &mk(1, 0, "2"), 4000);
        assert_eq!(cum(&e4, "Anthropic / chat 2 / calls"), 1.0, "per-chat total");
        assert_eq!(cum(&e4, "Anthropic / chat 2 / errors"), 0.0);
    }

    #[test]
    fn flush_final_rate_emits_open_second() {
        // At drain the still-open second's data must not be lost waiting for a tick.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mut c = completed();
        c.tokens_in = 42;
        c.model = Some("m".into());
        s.accumulate_rate("Anthropic", &c, 1000, &mut fwd); // second 1, open bucket has data
        assert!(s.metrics_buf.is_empty(), "nothing flushed yet (open second)");
        s.flush_final_rate(&mut fwd);
        let e = take_metrics(&mut s);
        assert_eq!(scalar_value(&e, "LLM Input Tokens", "Anthropic / m"), 42.0, "open second emitted");
        // Idempotent: the bucket is cleared, so a second call emits nothing.
        s.flush_final_rate(&mut fwd);
        assert!(s.metrics_buf.is_empty(), "no double-emit");
    }

    #[test]
    fn flush_final_rate_noop_when_idle() {
        // A drain during an idle open second emits no trailing 0-point.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        let mut c = completed();
        c.tokens_in = 10;
        c.model = Some("m".into());
        s.accumulate_rate("Anthropic", &c, 1000, &mut fwd);
        s.on_tick(3000, &mut fwd); // advance past second 1; open bucket now empty
        let _ = take_metrics(&mut s);
        s.flush_final_rate(&mut fwd);
        assert!(s.metrics_buf.is_empty(), "no trailing 0-point at drain");
    }

    #[test]
    fn on_tick_noop_before_first_request() {
        // With no series yet the tick has nothing to 0-fill and must not start the
        // clock — the leading idle before the first call produces no points.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = sinks(&["tokens_in"]);
        s.on_tick(1000, &mut fwd);
        s.on_tick(5000, &mut fwd);
        assert!(s.metrics_buf.is_empty(), "no series yet -> nothing to 0-fill");
        assert!(s.rate_cur_sec.is_none(), "clock unstarted until the first request");
    }

    #[test]
    fn on_tick_noop_when_metrics_off() {
        // The rate clock only runs when task-metrics are on.
        let mut fwd = LogForwarder::new("t".into(), "w".into());
        let mut s = Sinks::new("task-1".into(), true, false, &[], None, String::new(), String::new());
        s.on_tick(1000, &mut fwd);
        assert!(s.rate_cur_sec.is_none(), "clock stays off when report_metrics is false");
        assert!(s.metrics_buf.is_empty());
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
