//! Diagnostic log sink for the shim's own lines (`[snug] init`, `[snug-event]`,
//! `[snug-inline]`, `[snug-h2dbg]`).
//!
//! Two macros feed one sink:
//!   * `snug_log!` — routine per-process diagnostics; a NO-OP unless
//!     `debug_enabled()`. The shim is preloaded into dozens of non-metering
//!     helper processes (multi-process desktop hosts — Electron/Chromium and
//!     helper/sandbox children), so routine chatter stays silent by default.
//!   * `snug_err!` — errors / FATAL and the single reporting-process `[snug]
//!     init` line; ALWAYS written.
//!
//! The sink normally goes to stderr, but in embedded hosts stderr is unreliable:
//! a multi-process desktop host (e.g. Electron/Chromium) may re-exec and redirect
//! it, and child services it spawns get an stderr that is a socket piped back to
//! the parent — so the lines never reach any file an operator can read. When
//! `CLEARML_SNUG_LOG_FILE` is set, every shim process appends its lines to that
//! one path instead (each line pid-tagged), giving a single readable stream
//! across the whole process tree. Unset → stderr, unchanged.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::OnceLock;

use parking_lot::Mutex;

fn sink() -> &'static Option<Mutex<File>> {
    static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    SINK.get_or_init(|| {
        std::env::var("CLEARML_SNUG_LOG_FILE").ok().and_then(|p| {
            OpenOptions::new().create(true).append(true).open(&p).ok().map(Mutex::new)
        })
    })
}

/// True iff verbose per-process `[snug]` diagnostics are enabled, gating
/// `snug_log!`. Driven by `CLEARML_SNUG_DEBUG_LOG`, with the older h2-spike gate
/// `CLEARML_SNUG_H2_DEBUG` folded in for back-compat. Cached for the process.
pub fn debug_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| debug_enabled_from(|k| std::env::var(k).ok()))
}

/// Inner form of `debug_enabled` taking an env getter, so the truthiness logic
/// is unit-testable without the process-wide `OnceLock` caching the first read.
/// Truthy = value in {1, true, yes, on}, case-insensitive.
fn debug_enabled_from(get: impl Fn(&str) -> Option<String>) -> bool {
    let truthy = |k: &str| {
        get(k)
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    };
    truthy("CLEARML_SNUG_DEBUG_LOG") || truthy("CLEARML_SNUG_H2_DEBUG")
}

/// Write one diagnostic line to the log file (pid-tagged) if
/// `CLEARML_SNUG_LOG_FILE` is set, else to stderr.
pub fn line(args: std::fmt::Arguments) {
    match sink() {
        Some(m) => {
            let mut f = m.lock();
            let _ = writeln!(f, "[pid {}] {}", std::process::id(), args);
        }
        None => eprintln!("{args}"),
    }
}

/// Routine per-process diagnostic line. A NO-OP unless `debug_enabled()`, so the
/// shim's non-metering helper processes stay silent by default; routes through
/// the shared `line()` sink when enabled. `format_args!` is lazy, so nothing is
/// formatted when disabled.
macro_rules! snug_log {
    ($($a:tt)*) => {
        if $crate::log::debug_enabled() {
            $crate::log::line(format_args!($($a)*))
        }
    };
}

/// Always-on diagnostic line for errors / FATAL and the single reporting-process
/// `[snug] init` line. Routes through the same `line()` sink regardless of
/// `debug_enabled()`.
macro_rules! snug_err {
    ($($a:tt)*) => { $crate::log::line(format_args!($($a)*)) };
}

#[cfg(test)]
mod tests {
    use super::debug_enabled_from;
    use std::collections::HashMap;

    fn from(pairs: &[(&str, &str)]) -> bool {
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        debug_enabled_from(|k| map.get(k).cloned())
    }

    #[test]
    fn off_when_no_env_set() {
        assert!(!from(&[]));
    }

    #[test]
    fn on_via_debug_log_flag() {
        assert!(from(&[("CLEARML_SNUG_DEBUG_LOG", "1")]));
        assert!(from(&[("CLEARML_SNUG_DEBUG_LOG", "true")]));
        assert!(from(&[("CLEARML_SNUG_DEBUG_LOG", "YES")]));
        assert!(from(&[("CLEARML_SNUG_DEBUG_LOG", "On")]));
    }

    #[test]
    fn on_via_h2_debug_flag_back_compat() {
        assert!(from(&[("CLEARML_SNUG_H2_DEBUG", "1")]));
        assert!(from(&[("CLEARML_SNUG_H2_DEBUG", "true")]));
    }

    #[test]
    fn off_for_non_truthy_values() {
        assert!(!from(&[("CLEARML_SNUG_DEBUG_LOG", "0")]));
        assert!(!from(&[("CLEARML_SNUG_DEBUG_LOG", "false")]));
        assert!(!from(&[("CLEARML_SNUG_DEBUG_LOG", "")]));
    }
}
