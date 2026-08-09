//! ClearML SNUG.
//!
//! Hooks OpenSSL's `SSL_write`, `SSL_read`, and `SSL_free` via the preload
//! mechanism (`LD_PRELOAD` + `#[no_mangle]` exports on Linux;
//! `DYLD_INSERT_LIBRARIES` + fishhook GOT rebinding on macOS, see
//! hooks/macos.rs), parses the metered traffic, and hands each `Event` to the
//! in-process
//! reporter (the `clearml_snug_reporter` crate it links) over a bounded channel.
//! The reporter runs on a background thread the ctor spawns and the `exit(3)`
//! hook drains + joins; it reports to the ClearML backend itself, in the task
//! process. A control-plane poll thread inside the reporter mutates the shim's
//! atomics (the `CALL_HISTORY_MODE` capture mode) directly via callbacks when a
//! runtime property changes.
//!
//! See the crate README for the event schema and overall design.

#![allow(clippy::missing_safety_doc)]

#[macro_use]
mod log;
mod body_scan;
mod call_history;
mod channel;
mod control;
mod decompress;
mod descriptor_handoff;
mod h2;
mod hooks;
mod init;
mod inject;
mod meter;
mod parser;
mod reentrancy;
mod reporter_handle;
mod self_host;
mod session;
mod state;
mod tokens;
mod whitelist;

// Re-export the hook functions at the crate root so the linker pulls them into
// the cdylib. The globally-visible set is narrowed by #[no_mangle] on each hook
// (SSL_{read,write,free,read_ex,write_ex} + exit); a CI nm guard enforces it.
//
// Linux only: there the hooks ARE the exported `SSL_*` symbols. macOS exports
// NO `SSL_*` — it rebinds each image's `SSL_*` GOT slots onto private functions
// via a dyld add-image callback (hooks/macos.rs), so dlsym still finds libssl's
// reals; an inverse nm guard checks that no `SSL_*` is exported.
#[cfg(target_os = "linux")]
pub use hooks::openssl::{SSL_free, SSL_read, SSL_write};
