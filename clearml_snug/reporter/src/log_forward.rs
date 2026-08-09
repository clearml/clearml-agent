//! Log-forwarding sink: push the shim's raw NDJSON event lines to the task's
//! CONSOLE as `[SNUG]`-prefixed log events via `events.add_batch`.
//!
//! Buffering: flush at 50 lines or on the caller's 5s timer, with a hard cap so
//! a backend outage can't grow memory unbounded.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::api::ClearmlClient;

const FLUSH_BATCH: usize = 50;
const BUFFER_HARD_CAP: usize = 1000;

pub struct LogForwarder {
    task_id: String,
    worker_id: String,
    lines: Vec<String>,
    dropped: u64,
}

impl LogForwarder {
    pub fn new(task_id: String, worker_id: String) -> Self {
        let worker_id = if worker_id.is_empty() {
            "clearml-snug-reporter".to_string()
        } else {
            worker_id
        };
        LogForwarder {
            task_id,
            worker_id,
            lines: Vec::new(),
            dropped: 0,
        }
    }

    pub fn should_flush(&self) -> bool {
        self.lines.len() >= FLUSH_BATCH
    }

    /// Prefix a raw NDJSON event line and queue it as a task-console log line.
    pub fn enqueue(&mut self, raw_line: &str) {
        if self.lines.len() >= BUFFER_HARD_CAP {
            self.dropped += 1;
            return;
        }
        self.lines.push(format!("{} {}", classify_prefix(raw_line), raw_line));
    }

    /// Queue an already-formatted line verbatim — no event classification. Used
    /// for the sinks' own `[SNUG-USAGE]` / `[SNUG-METRICS]` diagnostics, which
    /// are plain text (not shim NDJSON) and so bypass event classification.
    pub fn enqueue_diagnostic(&mut self, line: &str) {
        if self.lines.len() >= BUFFER_HARD_CAP {
            self.dropped += 1;
            return;
        }
        self.lines.push(line.to_string());
    }

    /// Flush queued lines to the task console via `events.add_batch`.
    /// Best-effort: a backend error is logged to stderr and the lines dropped,
    /// never fatal (a broken backend must not kill metering).
    pub fn flush(&mut self, client: &mut ClearmlClient) {
        if self.lines.is_empty() && self.dropped == 0 {
            return;
        }
        let mut lines = std::mem::take(&mut self.lines);
        if self.dropped > 0 {
            lines.push(format!(
                "[SNUG-WARN] reporter buffer overflow, dropped {} lines",
                self.dropped
            ));
            self.dropped = 0;
        }
        let base = now_ms();
        let events: Vec<Value> = lines
            .into_iter()
            .enumerate()
            .map(|(i, msg)| log_event(&self.task_id, &self.worker_id, &msg, base + i as i64))
            .collect();
        if let Err(e) = client.events_add_batch(&events) {
            eprintln!("WARNING: SNUG reporter log forwarding failed: {}", e);
        }
    }
}

/// Choose the console prefix from the event `kind`. A cheap
/// substring scan rather than a full JSON parse: classification is cosmetic, and
/// the shim emits compact NDJSON with `kind` as a leading, unique field, so this
/// keeps the log-forward hot path parse-free (the sinks do the one typed parse).
fn classify_prefix(raw_line: &str) -> &'static str {
    if raw_line.contains("\"kind\":\"CallHistoryEntry\"")
        || raw_line.contains("\"kind\":\"CallHistoryNotice\"")
    {
        // Normally call-history events are rendered decoded
        // (reporter::render_call_history / the CallHistoryNotice arm of
        // handle_event) rather than forwarded raw, so this arm is a defensive
        // fallback.
        "[SNUG-CALL]"
    } else if raw_line.contains("\"kind\":\"ShimDiagnostic\"") {
        "[SNUG-DIAG]"
    } else if raw_line.contains("\"kind\":\"") {
        "[SNUG]"
    } else {
        "[SNUG-WARN] non-JSON:"
    }
}

fn log_event(task: &str, worker: &str, msg: &str, ts_ms: i64) -> Value {
    json!({
        "type": "log",
        "level": "INFO",
        "task": task,
        "worker": worker,
        "msg": msg,
        "timestamp": ts_ms,
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_event_kinds() {
        assert_eq!(classify_prefix(r#"{"kind":"RequestStarted"}"#), "[SNUG]");
        assert_eq!(classify_prefix(r#"{"kind":"RequestCompleted"}"#), "[SNUG]");
        assert_eq!(classify_prefix(r#"{"kind":"CallHistoryEntry"}"#), "[SNUG-CALL]");
        assert_eq!(classify_prefix(r#"{"kind":"ShimDiagnostic"}"#), "[SNUG-DIAG]");
        assert_eq!(classify_prefix("not json"), "[SNUG-WARN] non-JSON:");
    }

    #[test]
    fn log_event_has_console_log_shape() {
        let e = log_event("t1", "w1", "[SNUG] hi", 123);
        assert_eq!(e["type"], "log");
        assert_eq!(e["level"], "INFO");
        assert_eq!(e["task"], "t1");
        assert_eq!(e["worker"], "w1");
        assert_eq!(e["msg"], "[SNUG] hi");
        assert_eq!(e["timestamp"], 123);
    }

    #[test]
    fn enqueue_prefixes_and_buffers() {
        let mut f = LogForwarder::new("t".into(), "".into());
        assert_eq!(f.worker_id, "clearml-snug-reporter"); // empty -> default
        f.enqueue(r#"{"kind":"RequestCompleted","tokens_in":5}"#);
        assert_eq!(f.lines.len(), 1);
        assert!(f.lines[0].starts_with("[SNUG] {"));
    }

    #[test]
    fn enqueue_diagnostic_is_verbatim() {
        // Pre-formatted diagnostics are queued as-is, NOT reclassified as
        // non-JSON (which would mislabel them "[SNUG-WARN] non-JSON: ...").
        let mut f = LogForwarder::new("t".into(), "w".into());
        f.enqueue_diagnostic("[SNUG-USAGE] OK");
        assert_eq!(f.lines, vec!["[SNUG-USAGE] OK".to_string()]);
    }
}
