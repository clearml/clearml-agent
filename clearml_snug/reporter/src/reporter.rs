//! The reporting thread: consumes metered `Event`s from an in-process
//! `mpsc::Receiver<Event>`, forwards each to the task console, runs the sinks,
//! and keeps the backend token fresh.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use base64::Engine as _;
use clearml_snug_messages::Event;

use crate::api::ClearmlClient;
use crate::descriptor::Descriptor;
use crate::log_forward::LogForwarder;
use crate::poll::{self, PollCallbacks};
use crate::sinks::Sinks;

/// Flush the log buffer at least this often.
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);
/// Channel-receive poll cadence so the flush + token-refresh timers fire even
/// when no events arrive (a long-idle task still wakes ~1×/s — negligible CPU).
const RECV_TIMEOUT: Duration = Duration::from_secs(1);
/// How often to run the proactive token-freshness check (traffic-independent).
const TOKEN_CHECK_INTERVAL: Duration = Duration::from_secs(60);

/// Owns the reporting thread (and the optional control-plane poll thread).
/// Held in a shim process-global so the `exit(3)` hook can drain + join it.
pub struct ReporterHandle {
    reporter_join: Option<JoinHandle<()>>,
    poll_join: Option<JoinHandle<()>>,
    /// Set to ask the reporter loop to drain the channel + do a final flush.
    drain: Arc<AtomicBool>,
    /// Set by the reporter loop right before it returns.
    done: Arc<AtomicBool>,
    /// Set to stop the poll thread.
    poll_stop: Arc<AtomicBool>,
}

/// Spawn the reporting thread (and, if `poll_cb` is `Some`, the control-plane
/// poll thread). Returns immediately; all network I/O happens on the spawned
/// thread(s). Never blocks the caller, never panics.
pub fn start_reporter(
    d: Descriptor,
    rx: Receiver<Event>,
    poll_cb: Option<PollCallbacks>,
) -> ReporterHandle {
    // Shared with the poll thread, which also drives token refresh via the API.
    let client = Arc::new(Mutex::new(ClearmlClient::from_descriptor(&d)));
    let drain = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let poll_stop = Arc::new(AtomicBool::new(false));

    // Let the client's retry loop observe the drain signal, so a data-plane
    // forever-retry bails within ~250ms at shutdown instead of stalling the
    // bounded exit drain budget. (The reporter's own drain check sits at the
    // bottom of run_read_loop, after handle_event — so a forever-retry inside
    // handle_event would never reach it without this.)
    if let Ok(mut c) = client.lock() {
        c.set_abort_signal(Arc::clone(&drain));
    }

    let reporter_join = {
        let client = Arc::clone(&client);
        let drain = Arc::clone(&drain);
        let done = Arc::clone(&done);
        let fwd = LogForwarder::new(d.task_id.clone(), d.worker_id.clone());
        let sinks = Sinks::new(
            d.task_id.clone(),
            d.report_usage_events,
            d.report_task_metrics,
            &d.task_metrics_fields,
            d.aggregator_url.clone(),
            d.user.clone(),
            d.project.clone(),
        );
        std::thread::Builder::new()
            .name("snug-reporter".to_string())
            .spawn(move || run_read_loop(rx, drain, done, client, fwd, sinks))
            .ok()
    };

    let poll_join = match poll_cb {
        Some(cb) => {
            let client = Arc::clone(&client);
            let stop = Arc::clone(&poll_stop);
            let task_id = d.task_id.clone();
            let interval = d.poll_interval_sec;
            std::thread::Builder::new()
                .name("snug-poll".to_string())
                .spawn(move || poll::run_poll_loop(client, task_id, interval, stop, cb))
                .ok()
        }
        None => None,
    };

    ReporterHandle {
        reporter_join,
        poll_join,
        drain,
        done,
        poll_stop,
    }
}

impl ReporterHandle {
    /// Ask the reporter to drain whatever is queued + do a final synchronous
    /// HTTP flush, then wait up to `timeout` for it to finish. Returns `true` if
    /// it finished within budget, `false` if we gave up waiting (the thread is
    /// then leaked — the process is about to `exit(3)` regardless). Bounded so
    /// it never hangs `exit`; idempotent-safe and panic-free.
    pub fn flush_and_join(mut self, timeout: Duration) -> bool {
        self.poll_stop.store(true, Ordering::SeqCst);
        self.drain.store(true, Ordering::SeqCst);

        let deadline = Instant::now() + timeout;
        while !self.done.load(Ordering::SeqCst) {
            if Instant::now() >= deadline {
                return false; // give up; leak the thread, the process is exiting
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        // `done` is set, so the thread has returned (or is about to). Join is
        // cheap now; both joins are best-effort.
        if let Some(j) = self.reporter_join.take() {
            let _ = j.join();
        }
        if let Some(j) = self.poll_join.take() {
            let _ = j.join();
        }
        true
    }
}

fn run_read_loop(
    rx: Receiver<Event>,
    drain: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    client: Arc<Mutex<ClearmlClient>>,
    mut fwd: LogForwarder,
    mut sinks: Sinks,
) {
    let mut last_flush = Instant::now();
    let mut last_token_check = Instant::now();
    // conn_ids whose host matched the whitelist — used to gate raw `[SNUG]`
    // event forwarding so non-whitelisted traffic produces no console log.
    // Keyed on RequestStarted (the only event carrying `whitelisted`);
    // BytesObserved / RequestCompleted carry just conn_id.
    let mut wl_conns: HashSet<u64> = HashSet::new();
    loop {
        match rx.recv_timeout(RECV_TIMEOUT) {
            Ok(ev) => handle_event(&ev, &mut fwd, &mut sinks, &mut wl_conns),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                // Every sender dropped (shouldn't happen before exit). Treat it
                // as a drain signal so we flush what we have and stop cleanly.
                drain.store(true, Ordering::SeqCst);
            }
        }

        // Flush the log buffer + the sink buffers under one client lock when
        // either is full (size trigger) or the 5s timer fires.
        if fwd.should_flush() || sinks.should_flush() || last_flush.elapsed() >= FLUSH_INTERVAL {
            if let Ok(mut c) = client.lock() {
                fwd.flush(&mut *c);
                sinks.flush(&mut *c, &mut fwd);
            }
            last_flush = Instant::now();
        }

        // Proactive, traffic-independent token freshness: a long-running task
        // with sparse LLM calls must never let its token lapse. Cheap (a mutex
        // lock + an expiry comparison; only POSTs when actually near expiry).
        if last_token_check.elapsed() >= TOKEN_CHECK_INTERVAL {
            if let Ok(mut c) = client.lock() {
                c.maybe_refresh_token();
            }
            last_token_check = Instant::now();
        }

        if drain.load(Ordering::SeqCst) {
            // Drain everything already queued, then a final flush, then stop.
            while let Ok(ev) = rx.try_recv() {
                handle_event(&ev, &mut fwd, &mut sinks, &mut wl_conns);
            }
            if let Ok(mut c) = client.lock() {
                fwd.flush(&mut *c);
                // Single bounded pass — no next tick to retry, so unsent events
                // are lost (the process is exiting; bounded by EXIT_DRAIN_TIMEOUT).
                sinks.flush(&mut *c, &mut fwd);
                // sinks.flush enqueued its per-batch OK/ERR diagnostics into fwd;
                // one trailing log flush delivers them before we stop.
                fwd.flush(&mut *c);
            }
            break;
        }
    }
    done.store(true, Ordering::SeqCst);
}

/// Forward one event to the task console and — when a reporting sink is active —
/// feed the usage / task-metrics / aggregator sinks. The NDJSON text is
/// produced HERE (on the reporter thread, off the user's hot path) and fed to
/// `LogForwarder`, whose `classify_prefix` keys off the `"kind":"…"` tag.
/// Prefix for all call-history console lines.
const SNUG_CALL: &str = "[SNUG-CALL]";

fn handle_event(ev: &Event, fwd: &mut LogForwarder, sinks: &mut Sinks, wl_conns: &mut HashSet<u64>) {
    match ev {
        // Captured calls are rendered DECODED + human-readable (not the raw
        // base64 NDJSON) so the task console shows the actual HTTP text.
        Event::CallHistoryEntry { .. } => render_call_history(ev, fwd),
        // Mode-switch / dump-summary notices: one verbatim [SNUG-CALL] row so
        // off/collect/dump/continuous transitions are visible on the console.
        Event::CallHistoryNotice { text, .. } => {
            fwd.enqueue_diagnostic(&format!("{SNUG_CALL} {text}"));
        }
        _ => {
            // `should_forward_raw` runs first so it keeps `wl_conns` maintained
            // regardless of the debug gate; the per-request event JSON is then
            // suppressed unless debug logging is on (see `is_per_request_event`).
            let forward =
                should_forward_raw(ev, wl_conns) && (debug_log_enabled() || !is_per_request_event(ev));
            if forward {
                if let Ok(text) = serde_json::to_string(ev) {
                    fwd.enqueue(&text);
                }
            }
        }
    }
    if sinks.enabled() {
        sinks.on_event(ev, fwd);
    }
}

/// Whether a raw `[SNUG]` event line should be forwarded to the task console.
///
/// `RequestStarted` (the per-request opening line) and `RequestCompleted` (the
/// per-request summary) reach the console; the per-write `BytesObserved` events
/// are suppressed, because a busy connection emits thousands of them and they
/// flood the task log. The useful `[SNUG-USAGE]` / `[SNUG-METRICS]` diagnostics
/// ride a separate path (`enqueue_diagnostic`) and are unaffected.
///
/// A non-whitelisted host is monitored (bytes counted) but never reported, so
/// neither its `RequestStarted` nor its `RequestCompleted` produces a `[SNUG]`
/// log. `RequestStarted` carries the `whitelisted` flag and gates on it directly;
/// `RequestCompleted` carries only conn_id, so we track which conn_ids are
/// whitelisted from `RequestStarted` and gate the completion on that set.
/// Non-request events (e.g. ShimDiagnostic) are not connection-gated.
fn should_forward_raw(ev: &Event, wl_conns: &mut HashSet<u64>) -> bool {
    match ev {
        Event::RequestStarted { conn_id, whitelisted, .. } => {
            if *whitelisted {
                wl_conns.insert(*conn_id);
            } else {
                wl_conns.remove(conn_id);
            }
            // Forwarded only for a whitelisted host, and always tracked so the
            // matching RequestCompleted (which lacks the flag) can gate on it.
            *whitelisted
        }
        // Per-write byte reports are far too frequent for the task console.
        Event::BytesObserved { .. } => false,
        Event::RequestCompleted { conn_id, .. } => {
            let w = wl_conns.contains(conn_id);
            // End of request; a keep-alive connection re-inserts on its next
            // RequestStarted, so dropping here bounds the set.
            wl_conns.remove(conn_id);
            w
        }
        _ => true,
    }
}

/// True for the per-request event JSON — `RequestStarted` and `RequestCompleted`
/// — which reaches the task console (as a raw `[SNUG] {json}` line) only under
/// debug logging. These two lines fire for every LLM call and flood a busy
/// task's log, and their numbers are already summarized by the always-on
/// `[SNUG-USAGE]` / `[SNUG-METRICS]` diagnostics. Other raw events (e.g.
/// `ShimDiagnostic`) are rare and always forward.
fn is_per_request_event(ev: &Event) -> bool {
    matches!(ev, Event::RequestStarted { .. } | Event::RequestCompleted { .. })
}

/// Whether verbose per-request console forwarding is enabled. Mirrors the shim's
/// `log::debug_enabled` (the gate behind `snug_log!`): the reporter is a
/// separate crate that can't reach the shim's `log` module, so it reads the same
/// env vars directly. Cached for the process lifetime.
fn debug_log_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| debug_log_enabled_from(|k| std::env::var(k).ok()))
}

/// Inner form of `debug_log_enabled` taking an env getter, so the truthiness
/// logic is unit-testable without the process-wide `OnceLock` cache. Truthy =
/// value in {1, true, yes, on}, case-insensitive (matches the shim).
fn debug_log_enabled_from(get: impl Fn(&str) -> Option<String>) -> bool {
    let truthy = |k: &str| {
        get(k)
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    };
    truthy("CLEARML_SNUG_DEBUG_LOG") || truthy("CLEARML_SNUG_H2_DEBUG")
}

/// Render a captured request/response pair into decoded, human-readable console
/// lines with a `[SNUG-CALL]` prefix: a header, then the decoded request, then
/// the decoded response.
fn render_call_history(ev: &Event, fwd: &mut LogForwarder) {
    let Event::CallHistoryEntry {
        seq,
        host,
        path,
        method,
        status,
        request_b64,
        response_b64,
        request_truncated,
        response_truncated,
        response_compressed,
        chat_id,
        ..
    } = ev
    else {
        return;
    };

    const C: &str = SNUG_CALL;
    // Identify the call by its chat ordinal (the task's running chat number,
    // which matches the SCALARS series and reflects the call's position in the
    // task — e.g. the 8th call reads "chat #8" even if it's the 1st captured).
    // Fall back to the capture sequence when there's no chat id (usage parsing
    // off / non-LLM / unparseable body).
    let id = match chat_id {
        Some(c) => format!("chat #{c}"),
        None => format!("#{seq}"),
    };
    let status_s = match status {
        Some(s) => s.to_string(),
        None => "-".to_string(),
    };
    let mut flags = String::new();
    if *request_truncated {
        flags.push_str(" [req-truncated]");
    }
    if *response_truncated {
        flags.push_str(" [resp-truncated]");
    }
    if *response_compressed {
        flags.push_str(" [resp-compressed]");
    }
    fwd.enqueue_diagnostic(&format!(
        "{C} {id} {method} {host}{path} -> {status_s}{flags}"
    ));
    if !request_b64.is_empty() {
        fwd.enqueue_diagnostic(&format!("{C} {id} >>> REQUEST"));
        for line in decode_lines(request_b64) {
            fwd.enqueue_diagnostic(&line);
        }
    }
    if !response_b64.is_empty() {
        fwd.enqueue_diagnostic(&format!("{C} {id} <<< RESPONSE"));
        for line in decode_lines(response_b64) {
            fwd.enqueue_diagnostic(&line);
        }
    }
}

/// Decode a base64 chunk into console lines: UTF-8 text split on newlines, or a
/// single placeholder for binary/compressed bytes.
fn decode_lines(b64: &str) -> Vec<String> {
    let bytes = match base64::engine::general_purpose::STANDARD.decode(b64) {
        Ok(b) => b,
        Err(_) => return vec!["<undecodable base64>".to_string()],
    };
    match std::str::from_utf8(&bytes) {
        Ok(s) => s.lines().map(|l| l.to_string()).collect(),
        Err(_) => vec![format!("<{} bytes, non-UTF8 (compressed/binary)>", bytes.len())],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clearml_snug_messages::Direction;
    use std::sync::mpsc::sync_channel;

    // A descriptor with no backend creds and sinks off: the reporter runs but
    // never makes a successful backend call (login fails fast / is never
    // needed), so this exercises the channel→loop→drain lifecycle without a
    // live server.
    fn bare_descriptor() -> Descriptor {
        Descriptor::from_json_str(
            r#"{"api_server":"https://127.0.0.1:1/","task_id":"t-test"}"#,
        )
        .expect("parse descriptor")
    }

    fn req_started(conn_id: u64, whitelisted: bool) -> Event {
        Event::RequestStarted {
            conn_id,
            ts_ms: 0,
            host: "h".into(),
            path: "/".into(),
            method: "POST".into(),
            whitelisted,
            inject_headers: false,
        }
    }

    fn bytes(conn_id: u64) -> Event {
        Event::BytesObserved {
            conn_id,
            ts_ms: 0,
            direction: Direction::Tx,
            bytes: 1,
            tokens_est: 1,
        }
    }

    fn req_completed(conn_id: u64) -> Event {
        Event::RequestCompleted {
            conn_id,
            ts_ms: 0,
            status: Some(200),
            latency_ms: 0,
            bytes_tx: 0,
            bytes_rx: 0,
            tokens_in: 0,
            tokens_out: 0,
            tokens_measured: false,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: None,
        }
    }

    #[test]
    fn raw_forward_gated_on_whitelisted_connection() {
        let mut wl = HashSet::new();

        // Whitelisted connection (conn 1): RequestStarted + RequestCompleted
        // forward; the per-write BytesObserved is suppressed (console flood).
        // RequestStarted also records the conn as whitelisted so its
        // RequestCompleted can gate on it.
        assert!(should_forward_raw(&req_started(1, true), &mut wl));
        assert!(!should_forward_raw(&bytes(1), &mut wl));
        assert!(should_forward_raw(&req_completed(1), &mut wl));

        // Non-whitelisted connection (conn 2): nothing forwards.
        assert!(!should_forward_raw(&req_started(2, false), &mut wl));
        assert!(!should_forward_raw(&bytes(2), &mut wl));
        assert!(!should_forward_raw(&req_completed(2), &mut wl));

        // Bytes never forward regardless of whitelist state.
        assert!(!should_forward_raw(&bytes(3), &mut wl));

        // A whitelisted RequestStarted forwards AND records the conn; the
        // matching RequestCompleted forwards and clears it, so a reused
        // keep-alive conn_id is re-evaluated by its next RequestStarted.
        assert!(should_forward_raw(&req_started(1, true), &mut wl));
        assert!(should_forward_raw(&req_completed(1), &mut wl));
        assert!(!should_forward_raw(&req_completed(1), &mut wl)); // completed cleared it
        assert!(!should_forward_raw(&req_started(1, false), &mut wl)); // now non-whitelisted
        assert!(!should_forward_raw(&req_completed(1), &mut wl));
    }

    fn shim_diag() -> Event {
        Event::ShimDiagnostic {
            ts_ms: 0,
            kind_detail: "http2_unsupported".into(),
            conn_id: None,
            dropped_events: None,
            host: None,
        }
    }

    #[test]
    fn only_request_events_are_debug_gated() {
        // The raw `[SNUG] {json}` flood is exactly RequestStarted +
        // RequestCompleted; other raw events (ShimDiagnostic) stay always-on and
        // BytesObserved is suppressed by should_forward_raw regardless.
        assert!(is_per_request_event(&req_started(1, true)));
        assert!(is_per_request_event(&req_completed(1)));
        assert!(!is_per_request_event(&bytes(1)));
        assert!(!is_per_request_event(&shim_diag()));
    }

    #[test]
    fn debug_gate_matches_shim_truthiness() {
        let from = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<String, String> =
                pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            debug_log_enabled_from(|k| map.get(k).cloned())
        };
        assert!(!from(&[]));
        assert!(from(&[("CLEARML_SNUG_DEBUG_LOG", "1")]));
        assert!(from(&[("CLEARML_SNUG_DEBUG_LOG", "On")]));
        assert!(from(&[("CLEARML_SNUG_H2_DEBUG", "true")])); // h2 gate folded in
        assert!(!from(&[("CLEARML_SNUG_DEBUG_LOG", "0")]));
        assert!(!from(&[("CLEARML_SNUG_DEBUG_LOG", "false")]));
    }

    #[test]
    fn channel_lifecycle_drains_and_joins() {
        // Sinks off + empty buffer => the loop never touches the network, so
        // this exercises spawn → drain → join purely. `_tx` stays alive so the
        // channel isn't disconnected before we signal drain.
        let (_tx, rx) = sync_channel::<Event>(64);
        let handle = start_reporter(bare_descriptor(), rx, None);
        assert!(
            handle.flush_and_join(Duration::from_secs(5)),
            "reporter should drain and join within budget"
        );
    }

    #[test]
    fn flush_and_join_returns_after_sender_dropped() {
        let (tx, rx) = sync_channel::<Event>(8);
        let handle = start_reporter(bare_descriptor(), rx, None);
        drop(tx); // disconnect: the loop should treat this as a drain signal
        assert!(handle.flush_and_join(Duration::from_secs(5)));
    }
}
