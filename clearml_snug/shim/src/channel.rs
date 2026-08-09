//! In-process event channel: the shim's hot path hands metered `Event`s to the
//! reporter thread over a bounded `mpsc::sync_channel`.
//!
//! `try_send` is non-blocking and drops on a full queue (counted in `DROPPED`),
//! so the user task's `SSL_write`/`SSL_read` is NEVER stalled on the network or a
//! slow reporter. When no reporter is installed (no descriptor was handed off —
//! e.g. an operator running `curl` under LD_PRELOAD directly), events fall back
//! to a `[snug-event] {json}` stderr line so they still surface somewhere.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::OnceLock;

use clearml_snug_messages::Event;

/// Bounded queue depth. A full queue drops the event (the user's hot path must
/// never block on the reporter).
pub const CHANNEL_CAPACITY: usize = 8192;

/// Cumulative count of events dropped due to backpressure (channel full).
/// Surfaced via a ShimDiagnostic in a future change if needed.
pub static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Sending end, installed once by the ctor after the reporter (which owns the
/// receiver) is spawned. `None` until then / when no reporter is started.
static TX: OnceLock<SyncSender<Event>> = OnceLock::new();

/// Install the sending end of the event channel. Idempotent (a second call is
/// ignored); called once from the shim ctor.
pub fn install(tx: SyncSender<Event>) {
    let _ = TX.set(tx);
}

/// Hand an event to the reporter thread. Non-blocking. Drops (counted) on a
/// full queue; falls back to a stderr line when no reporter is installed.
pub fn try_send(event: Event) {
    match TX.get() {
        Some(tx) => match tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
            // Reporter thread gone (shouldn't happen before process exit).
            Err(TrySendError::Disconnected(_)) => {}
        },
        None => {
            if let Ok(s) = serde_json::to_string(&event) {
                snug_log!("[snug-event] {}", s);
            }
        }
    }
}
