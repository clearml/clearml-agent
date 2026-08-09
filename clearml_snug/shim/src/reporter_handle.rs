//! Process-global holder for the reporter thread handle, so the `exit(3)` hook
//! can drain + join the reporter (bounded) before chaining to the real exit —
//! flushing the final `RequestCompleted` events the exit-time `state` flush just
//! enqueued.

use std::sync::mpsc::sync_channel;
use std::sync::Mutex;
use std::sync::Once;
use std::sync::OnceLock;
use std::time::Duration;

use clearml_snug_messages::Event;
use clearml_snug_reporter::{start_reporter, Descriptor, PollCallbacks, ReporterHandle};

static HANDLE: OnceLock<Mutex<Option<ReporterHandle>>> = OnceLock::new();

/// Descriptor + poll callbacks stored at ctor for a DEFERRED reporter start.
/// The reporter thread is not spawned at load time; only when this process
/// actually produces a metered usage event (`ensure_started`). This keeps
/// reporter threads out of the many shim-loaded processes that never meter — in
/// particular multi-process desktop hosts (Electron/Chromium and their
/// helper/sandbox children), which abort when a reporter's threads + outbound TLS
/// appear inside them.
static PENDING: OnceLock<Mutex<Option<(Descriptor, PollCallbacks)>>> = OnceLock::new();
static START_ONCE: Once = Once::new();

/// Store the descriptor for a deferred reporter start (see `PENDING`). Called
/// once from the ctor when a descriptor is available. Does NOT spawn threads.
pub fn store_pending(descriptor: Descriptor, poll_cb: PollCallbacks) {
    let cell = PENDING.get_or_init(|| Mutex::new(None));
    if let Ok(mut g) = cell.lock() {
        *g = Some((descriptor, poll_cb));
    }
}

/// Start the in-process reporter the first time this process produces a metered
/// usage event. Idempotent (a `Once`); a no-op in processes with no stored
/// descriptor. Called from `meter::emit` on a `RequestCompleted`.
pub fn ensure_started() {
    START_ONCE.call_once(|| {
        let pending = PENDING
            .get()
            .and_then(|c| c.lock().ok().and_then(|mut g| g.take()));
        let (descriptor, poll_cb) = match pending {
            Some(p) => p,
            None => return, // no descriptor in this process -> stderr fallback.
        };
        // Install the channel BEFORE spawning so the triggering event (sent
        // right after this returns) reaches the reporter, not the stderr path.
        let (tx, rx) = sync_channel::<Event>(crate::channel::CHANNEL_CAPACITY);
        crate::channel::install(tx);
        let handle = start_reporter(descriptor, rx, Some(poll_cb));
        install(handle);
        // SAFETY: getpid is async-signal-safe and infallible.
        let pid = unsafe { libc::getpid() };
        // Fires ONCE, only in a process that actually meters (this runs on the
        // first metered event). This is the always-on "shim active here" signal;
        // the ctor init line is debug-only because a descriptor is broadcast
        // tree-wide and so can't distinguish the metering process from the many
        // idle helpers.
        snug_err!(
            "[snug] init pid={} rules={} self_hosts={} reporter=in_process",
            pid,
            crate::whitelist::current().rules.len(),
            crate::self_host::current().len()
        );
    });
}

/// Store the reporter handle. Called once from the ctor after `start_reporter`.
pub fn install(handle: ReporterHandle) {
    let cell = HANDLE.get_or_init(|| Mutex::new(None));
    if let Ok(mut g) = cell.lock() {
        *g = Some(handle);
    }
}

/// Signal the reporter to drain + final-flush, then join it within `timeout`.
/// Idempotent: `take()`s the handle, so a second `exit()` (or a call when no
/// reporter was installed) is a no-op. Never panics — `exit(3)` must not unwind.
pub fn flush_and_join(timeout: Duration) {
    if let Some(cell) = HANDLE.get() {
        let taken = cell.lock().ok().and_then(|mut g| g.take());
        if let Some(handle) = taken {
            let _ = handle.flush_and_join(timeout);
        }
    }
}
