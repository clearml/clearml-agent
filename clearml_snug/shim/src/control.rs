//! Shim control state: the call-history capture mode + its size knobs, plus the
//! tokenizer / parse-usage env flags.
//!
//! The call-history mode is read by the hot path (`state.rs`) to decide whether
//! to capture/emit request/response pairs, and is mutated by the reporter's
//! control-plane poll thread via the `set_call_history_mode` setter (see
//! `clearml_snug_reporter::poll` + `crate::call_history`).

use std::sync::atomic::{AtomicU8, Ordering};

use clearml_snug_messages::CallHistoryMode;

/// Current call-history capture mode. 0 = Off, 1 = Collect, 2 = Dump,
/// 3 = Continuous. AtomicU8 so the hot path can read with `Relaxed` ordering.
static CALL_HISTORY_MODE: AtomicU8 = AtomicU8::new(0);

/// Cached per-direction capture cap. Read once from
/// `CLEARML_SNUG_CALL_HISTORY_CAP_BYTES`; default 256 KiB. `0` = uncapped.
static CALL_HISTORY_CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Cached ring-buffer depth. Read once from `CLEARML_SNUG_CALL_HISTORY_BUFFER`;
/// default 50.
static CALL_HISTORY_BUFFER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Cached redaction flag. Read once from `CLEARML_SNUG_CALL_HISTORY_REDACT`;
/// default true (mask credentials before buffering/printing).
static CALL_HISTORY_REDACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn mode_to_u8(m: CallHistoryMode) -> u8 {
    match m {
        CallHistoryMode::Off => 0,
        CallHistoryMode::Collect => 1,
        CallHistoryMode::Dump => 2,
        CallHistoryMode::Continuous => 3,
    }
}

fn u8_to_mode(v: u8) -> CallHistoryMode {
    match v {
        1 => CallHistoryMode::Collect,
        2 => CallHistoryMode::Dump,
        3 => CallHistoryMode::Continuous,
        _ => CallHistoryMode::Off,
    }
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

/// Current call-history mode. Cheap atomic load; safe to call from any thread.
pub fn call_history_mode() -> CallHistoryMode {
    u8_to_mode(CALL_HISTORY_MODE.load(Ordering::Relaxed))
}

/// Set the initial call-history mode from `CLEARML_SNUG_CALL_HISTORY` at ctor
/// time. After that, only the poll thread (via `set_call_history_mode`) mutates
/// it. Unrecognized / unset → Off.
pub fn set_initial_call_history_mode_from_env() {
    if let Some(m) = std::env::var("CLEARML_SNUG_CALL_HISTORY")
        .ok()
        .and_then(|s| parse_mode(&s))
    {
        CALL_HISTORY_MODE.store(mode_to_u8(m), Ordering::Relaxed);
    }
}

/// Flip the call-history mode. Called by the reporter's poll thread on a runtime
/// `_snug_call_history` change. A plain `fn(CallHistoryMode)` so it can be passed
/// as a `PollCallbacks` function pointer.
pub fn set_call_history_mode(mode: CallHistoryMode) {
    CALL_HISTORY_MODE.store(mode_to_u8(mode), Ordering::Relaxed);
}

/// Maximum bytes captured per direction (request, response) of a call-history
/// pair. `0` means uncapped.
pub fn call_history_cap_bytes() -> usize {
    *CALL_HISTORY_CAP.get_or_init(|| {
        std::env::var("CLEARML_SNUG_CALL_HISTORY_CAP_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(262144)
    })
}

/// Number of most-recent request/response pairs the ring buffer retains.
pub fn call_history_buffer_size() -> usize {
    *CALL_HISTORY_BUFFER.get_or_init(|| {
        std::env::var("CLEARML_SNUG_CALL_HISTORY_BUFFER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(50)
    })
}

/// Whether to redact credentials from captured requests before buffering/
/// printing. Default true; set `CLEARML_SNUG_CALL_HISTORY_REDACT=0` to disable
/// (strongly discouraged — leaks provider API keys into the task log).
pub fn call_history_redact() -> bool {
    *CALL_HISTORY_REDACT.get_or_init(|| {
        match std::env::var("CLEARML_SNUG_CALL_HISTORY_REDACT") {
            Ok(s) => !matches!(s.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
            Err(_) => true,
        }
    })
}

/// Default tokenizer name applied to connections the whitelist doesn't match.
/// Read once from `CLEARML_SNUG_DEFAULT_TOKENIZER`. Falls back to "approx" if
/// unset or unrecognized (the tokens module also treats unknown names as
/// "approx", so a typo here is at worst a no-op).
static DEFAULT_TOKENIZER: std::sync::OnceLock<String> = std::sync::OnceLock::new();

pub fn default_tokenizer() -> &'static str {
    DEFAULT_TOKENIZER
        .get_or_init(|| {
            std::env::var("CLEARML_SNUG_DEFAULT_TOKENIZER")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "approx".to_string())
        })
        .as_str()
}

/// Whether the shim should parse provider-reported token usage out of response
/// bodies. Read once from `CLEARML_SNUG_PARSE_USAGE` ("1" or "true"). The agent
/// only sets this when a reporting sink is enabled, so default-off keeps the
/// no-sink config free of any body-parsing cost on the hot path.
static PARSE_USAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn parse_usage_enabled() -> bool {
    *PARSE_USAGE.get_or_init(|| {
        std::env::var("CLEARML_SNUG_PARSE_USAGE")
            .ok()
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Serializes tests that mutate the process-global `CALL_HISTORY_MODE` atomic
/// (here + in `crate::state`), so `cargo test`'s parallelism can't make them
/// observe each other's mode. Test-only.
#[cfg(test)]
pub(crate) static MODE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_call_history_mode_flips_all_four() {
        let _g = MODE_TEST_LOCK.lock().unwrap();
        for m in [
            CallHistoryMode::Off,
            CallHistoryMode::Collect,
            CallHistoryMode::Dump,
            CallHistoryMode::Continuous,
        ] {
            set_call_history_mode(m);
            assert_eq!(call_history_mode(), m);
        }
        set_call_history_mode(CallHistoryMode::Off);
    }

    #[test]
    fn parse_mode_accepts_the_four_values_case_insensitively() {
        assert_eq!(parse_mode("off"), Some(CallHistoryMode::Off));
        assert_eq!(parse_mode(" Collect "), Some(CallHistoryMode::Collect));
        assert_eq!(parse_mode("DUMP"), Some(CallHistoryMode::Dump));
        assert_eq!(parse_mode("continuous"), Some(CallHistoryMode::Continuous));
        assert_eq!(parse_mode("bogus"), None);
        assert_eq!(parse_mode(""), None);
    }

    #[test]
    fn u8_round_trips_modes() {
        for m in [
            CallHistoryMode::Off,
            CallHistoryMode::Collect,
            CallHistoryMode::Dump,
            CallHistoryMode::Continuous,
        ] {
            assert_eq!(u8_to_mode(mode_to_u8(m)), m);
        }
        // Out-of-range u8 decodes to Off.
        assert_eq!(u8_to_mode(99), CallHistoryMode::Off);
    }
}
