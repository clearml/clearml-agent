//! Event emission: hand the metered `Event` to the in-process reporting channel
//! (`channel::try_send`), which forwards it to the reporter thread or — when no
//! reporter was started (e.g. an operator running `curl` under LD_PRELOAD with
//! no descriptor) — falls back to a `[snug-event] {json}` stderr line so events
//! still surface somewhere.

use clearml_snug_messages::Event;

/// Ship `event` to the reporter. Takes the event BY VALUE and moves it into the
/// bounded channel — no JSON serialization on the user's hot path (the reporter
/// serializes it off-thread for log-forwarding). Non-blocking; see `channel`.
pub fn emit(event: Event) {
    // Lazily start the in-process reporter the first time this process produces
    // a completed request (the only event carrying usage worth reporting). This
    // confines the reporter thread to the process that actually meters LLM
    // traffic — multi-process desktop hosts (Electron/Chromium and helper/sandbox
    // children) load the shim but never emit RequestCompleted, so they never
    // spawn a reporter (which would abort them). No-op after the first call and in
    // processes with no descriptor.
    if matches!(event, Event::RequestCompleted { .. }) {
        crate::reporter_handle::ensure_started();
    }
    crate::channel::try_send(event);
}
