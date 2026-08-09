//! In-process SNUG reporting library, linked by the shim.
//!
//! Consumes the shim's metered `Event`s from an in-process `mpsc` channel on a
//! background thread, and reports them to the ClearML backend itself — forwarding
//! each event to the task's console as a `[SNUG]`-prefixed log
//! (`events.add_batch`) and feeding the usage / task-metrics / aggregator
//! sinks.
//!
//! The shim spawns the reporter from its `#[ctor]` (passing the receiving end of
//! the event channel) and drains+joins it from the `exit(3)` hook.
//!
//! TLS is rustls + ring: a pure-Rust stack that reads/writes raw TCP via libc
//! and never calls the OpenSSL symbols the shim hooks, so the reporter's own
//! backend traffic is invisible to the shim — no recursion, no self-metering.

mod api;
pub mod descriptor;
mod log_forward;
mod poll;
mod reporter;
mod sinks;

pub use descriptor::Descriptor;
pub use poll::PollCallbacks;
pub use reporter::{start_reporter, ReporterHandle};
