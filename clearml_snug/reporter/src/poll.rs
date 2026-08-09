//! Control plane: poll the task's User Properties (hyperparams section
//! `properties`) and apply changes by calling back into the shim — the
//! call-history capture mode + the explicit dump trigger.
//!
//! User Properties are the ClearML field an operator can edit WHILE THE TASK
//! RUNS, so they're the live control surface for the 4 capture modes. Runs on
//! its own thread inside the task process. The reporter crate can't depend on
//! the shim (that would be a dependency cycle), so the shim injects plain `fn`
//! pointers (`PollCallbacks`) that mutate its atomics directly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clearml_snug_messages::CallHistoryMode;
use serde_json::{Map, Value};

use crate::api::ClearmlClient;

// User-Property keys (match clearml_agent.helper.snug.SNUG_USERPROP_*).
const PROP_CALL_HISTORY: &str = "_snug_call_history";
const PROP_POLL_RATE: &str = "_snug_user_properties_poll_rate";
const PROP_WHITELIST: &str = "_snug_whitelist";

/// Shim-supplied callbacks the poll thread invokes on a property change. Plain
/// `fn` pointers (the shim's `pub fn`s) keep the reporter crate free of any
/// dependency on the shim.
#[derive(Clone, Copy)]
pub struct PollCallbacks {
    /// Flip the capture mode. The shim's implementation also handles the `Dump`
    /// edge: entering `dump` prints the backlog once, then settles into
    /// `Collect` (the sliding window continues).
    pub set_call_history_mode: fn(CallHistoryMode),
    /// Apply a `_snug_whitelist` change: merge the (raw) additions onto the
    /// immutable base whitelist and atomically swap the result in. The reporter
    /// passes the property value through verbatim; the shim owns all parsing.
    pub reload_whitelist: fn(&str),
}

#[derive(Default, PartialEq, Debug)]
struct RuntimeFields {
    call_history: Option<CallHistoryMode>,
    poll_sec: Option<u64>,
    /// Raw `_snug_whitelist` value, passed to the shim verbatim. `None` = the
    /// key is absent (no opinion); `Some("")` = the operator cleared it.
    whitelist: Option<String>,
}

/// What a poll tick should do, derived purely from (last, current). Extracted so
/// the edge-trigger logic is unit-testable without the global atomics / network.
#[derive(Default, PartialEq, Debug)]
struct PollActions {
    set_mode: Option<CallHistoryMode>,
    /// After applying a `dump`, write the `_snug_call_history` property back to
    /// `collect` so the operator needn't switch back manually (and the next
    /// `dump` is a fresh edge). True iff this tick switched INTO `dump`.
    revert_dump_to_collect: bool,
    new_interval_sec: Option<u64>,
    /// The raw `_snug_whitelist` value to (re)apply this tick, set only when it
    /// changed. `Some("")` means the operator cleared it (revert to base).
    reload_whitelist: Option<String>,
}

fn parse_mode(s: &str) -> Option<CallHistoryMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "off" => Some(CallHistoryMode::Off),
        "collect" => Some(CallHistoryMode::Collect),
        "dump" => Some(CallHistoryMode::Dump),
        "continuous" => Some(CallHistoryMode::Continuous),
        _ => None,
    }
}

fn as_u64(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

/// Resolve the effective fields from a task User-Properties map (`name ->
/// value`). Unrecognized / absent keys yield `None` (no opinion → no change).
fn extract(props: &Map<String, Value>) -> RuntimeFields {
    let call_history = props
        .get(PROP_CALL_HISTORY)
        .and_then(|v| v.as_str())
        .and_then(parse_mode);
    let poll_sec = props.get(PROP_POLL_RATE).and_then(as_u64);
    // Raw value passed through verbatim; the shim owns all parsing/merging.
    // `None` = key absent (no opinion); `Some("")` = operator cleared it.
    let whitelist = props
        .get(PROP_WHITELIST)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    RuntimeFields {
        call_history,
        poll_sec,
        whitelist,
    }
}

/// Diff last vs current into the actions to take. A mode is applied only when it
/// changed. `dump` is one-shot per transition INTO it: the shim prints the
/// backlog once and settles into `Collect`, and the reporter then writes the
/// property back to `collect` (`revert_dump_to_collect`) so the operator doesn't
/// switch back by hand and a later `dump` is a fresh edge.
fn decide(last: &RuntimeFields, current: &RuntimeFields) -> PollActions {
    let mut a = PollActions::default();
    if let Some(m) = current.call_history {
        if Some(m) != last.call_history {
            a.set_mode = Some(m);
            a.revert_dump_to_collect = m == CallHistoryMode::Dump;
        }
    }
    if let Some(ps) = current.poll_sec {
        if ps >= 1 && Some(ps) != last.poll_sec {
            a.new_interval_sec = Some(ps);
        }
    }
    // Whitelist additions: re-apply on any change to the raw value, including a
    // transition to Some("") (operator cleared the field → the shim reverts to
    // base). No action when the key is absent (None) or unchanged.
    if current.whitelist != last.whitelist {
        if let Some(w) = &current.whitelist {
            a.reload_whitelist = Some(w.clone());
        }
    }
    a
}

fn read_fields(client: &Arc<Mutex<ClearmlClient>>, task_id: &str) -> Option<RuntimeFields> {
    let mut c = client.lock().ok()?;
    match c.get_task_user_properties(task_id) {
        Ok(props) => Some(extract(&props)),
        Err(e) => {
            eprintln!("WARNING: SNUG user-property poll failed: {}", e);
            None
        }
    }
}

/// Poll until `stop` is set: each tick reads the task's User Properties and
/// applies any change via the shim callbacks. Best-effort; never panics. Its
/// regular `get_task_user_properties` calls also serve as a second token-refresh
/// heartbeat (via `ensure_token`), independent of the reporter thread's timer.
pub fn run_poll_loop(
    client: Arc<Mutex<ClearmlClient>>,
    task_id: String,
    initial_interval_sec: f64,
    stop: Arc<AtomicBool>,
    cb: PollCallbacks,
) {
    let mut interval = Duration::from_secs_f64(initial_interval_sec.max(1.0));
    // Seed the baseline so the first poll doesn't emit a spurious mode flip for
    // a property already set when the task started.
    let mut last = read_fields(&client, &task_id).unwrap_or_default();

    while !stop.load(Ordering::SeqCst) {
        if !interruptible_sleep(interval, &stop) {
            return;
        }
        let current = match read_fields(&client, &task_id) {
            Some(f) => f,
            None => continue, // backend hiccup; retry next tick
        };
        let actions = decide(&last, &current);
        if let Some(m) = actions.set_mode {
            (cb.set_call_history_mode)(m);
        }
        if let Some(ref w) = actions.reload_whitelist {
            (cb.reload_whitelist)(w);
        }
        if let Some(ps) = actions.new_interval_sec {
            interval = Duration::from_secs(ps);
        }
        last = current;
        if actions.revert_dump_to_collect {
            // The shim already printed the backlog + settled to Collect. Reflect
            // that in the property so the operator needn't switch back, and so a
            // later `dump` is a fresh edge. Baseline `last` to Collect regardless
            // of the write result: on success the next poll reads "collect" (a
            // no-op); on a (rare) write failure the property stays "dump", the
            // next poll re-dumps — benign, and the backend is likely down anyway.
            if let Ok(mut c) = client.lock() {
                if let Err(e) = c.set_task_user_property(&task_id, PROP_CALL_HISTORY, "collect") {
                    eprintln!("WARNING: SNUG dump auto-revert to collect failed: {}", e);
                }
            }
            last.call_history = Some(CallHistoryMode::Collect);
        }
    }
}

/// Sleep up to `dur`, waking early (returning false) if `stop` is set.
fn interruptible_sleep(dur: Duration, stop: &AtomicBool) -> bool {
    let deadline = Instant::now() + dur;
    while Instant::now() < deadline {
        if stop.load(Ordering::SeqCst) {
            return false;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(Duration::from_millis(200)));
    }
    !stop.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(json: &str) -> Map<String, Value> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn extract_reads_the_four_modes() {
        assert_eq!(
            extract(&props(r#"{"_snug_call_history":"off"}"#)).call_history,
            Some(CallHistoryMode::Off)
        );
        assert_eq!(
            extract(&props(r#"{"_snug_call_history":"collect"}"#)).call_history,
            Some(CallHistoryMode::Collect)
        );
        assert_eq!(
            extract(&props(r#"{"_snug_call_history":"dump"}"#)).call_history,
            Some(CallHistoryMode::Dump)
        );
        assert_eq!(
            extract(&props(r#"{"_snug_call_history":"continuous"}"#)).call_history,
            Some(CallHistoryMode::Continuous)
        );
    }

    #[test]
    fn extract_absent_or_bogus_is_none() {
        assert_eq!(extract(&props("{}")).call_history, None);
        assert_eq!(
            extract(&props(r#"{"_snug_call_history":"bogus"}"#)).call_history,
            None
        );
    }

    #[test]
    fn extract_reads_poll_rate_as_int_or_string() {
        assert_eq!(
            extract(&props(r#"{"_snug_user_properties_poll_rate":2}"#)).poll_sec,
            Some(2)
        );
        assert_eq!(
            extract(&props(r#"{"_snug_user_properties_poll_rate":"3"}"#)).poll_sec,
            Some(3)
        );
    }

    #[test]
    fn decide_fires_set_mode_only_on_change() {
        let off = RuntimeFields {
            call_history: Some(CallHistoryMode::Off),
            ..Default::default()
        };
        let dump = RuntimeFields {
            call_history: Some(CallHistoryMode::Dump),
            ..Default::default()
        };
        // Off -> Dump fires the flip (which prints the backlog once).
        assert_eq!(decide(&off, &dump).set_mode, Some(CallHistoryMode::Dump));
        // Dump -> Dump (property parked) does NOT re-fire (no change).
        assert_eq!(decide(&dump, &dump).set_mode, None);
    }

    #[test]
    fn decide_marks_dump_for_auto_revert_to_collect() {
        let collect = RuntimeFields {
            call_history: Some(CallHistoryMode::Collect),
            ..Default::default()
        };
        let dump = RuntimeFields {
            call_history: Some(CallHistoryMode::Dump),
            ..Default::default()
        };
        let cont = RuntimeFields {
            call_history: Some(CallHistoryMode::Continuous),
            ..Default::default()
        };
        // Switching INTO dump flags the auto-revert (reporter writes the property
        // back to "collect"); other mode switches do not.
        assert!(decide(&collect, &dump).revert_dump_to_collect);
        assert!(!decide(&collect, &cont).revert_dump_to_collect);
        assert!(!decide(&dump, &dump).revert_dump_to_collect);
    }

    #[test]
    fn decide_changes_interval_on_poll_sec_change() {
        let a = RuntimeFields::default();
        let b = RuntimeFields {
            poll_sec: Some(2),
            ..Default::default()
        };
        assert_eq!(decide(&a, &b).new_interval_sec, Some(2));
        assert_eq!(decide(&b, &b).new_interval_sec, None);
    }

    #[test]
    fn extract_reads_whitelist_and_distinguishes_absent_from_cleared() {
        // present with a value (passed through verbatim — JSON or host list)
        assert_eq!(
            extract(&props(r#"{"_snug_whitelist":"a.com, b.com"}"#)).whitelist,
            Some("a.com, b.com".to_string())
        );
        // absent → None (no opinion; the shim is left alone)
        assert_eq!(extract(&props("{}")).whitelist, None);
        // present but empty → Some("") (the clear signal, distinct from absent)
        assert_eq!(
            extract(&props(r#"{"_snug_whitelist":""}"#)).whitelist,
            Some(String::new())
        );
    }

    #[test]
    fn decide_fires_whitelist_reload_on_change_and_clear() {
        let none = RuntimeFields::default();
        let some_a = RuntimeFields {
            whitelist: Some("a.com".into()),
            ..Default::default()
        };
        let some_b = RuntimeFields {
            whitelist: Some("b.com".into()),
            ..Default::default()
        };
        let cleared = RuntimeFields {
            whitelist: Some(String::new()),
            ..Default::default()
        };
        // first set fires with the value
        assert_eq!(decide(&none, &some_a).reload_whitelist, Some("a.com".into()));
        // a change fires with the new value
        assert_eq!(
            decide(&some_a, &some_b).reload_whitelist,
            Some("b.com".into())
        );
        // a clear (→ "") fires so the shim reverts to base
        assert_eq!(
            decide(&some_a, &cleared).reload_whitelist,
            Some(String::new())
        );
    }

    #[test]
    fn decide_no_whitelist_reload_when_unchanged_or_absent() {
        let some_a = RuntimeFields {
            whitelist: Some("a.com".into()),
            ..Default::default()
        };
        assert_eq!(decide(&some_a, &some_a).reload_whitelist, None); // unchanged
        let none = RuntimeFields::default();
        assert_eq!(decide(&none, &none).reload_whitelist, None); // both absent
        // key removed (Some → None): no opinion, don't thrash back to base
        assert_eq!(decide(&some_a, &none).reload_whitelist, None);
    }
}
