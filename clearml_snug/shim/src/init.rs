//! Constructor that runs once per process, before `main`, courtesy of `#[ctor]`
//! (which expands to a `.init_array` / `.ctors` entry).
//!
//! Ordering:
//!   1. seed `control::CALL_HISTORY_MODE` from `CLEARML_SNUG_CALL_HISTORY` and
//!      force-load the whitelist (both BEFORE any hook can fire),
//!   2. read the agent's handoff descriptor from the inherited memfd,
//!   3. on success: install the event channel + spawn the in-process reporter
//!      (and its control-plane poll thread); else fall back to stderr,
//!   4. log a single `[snug] init pid=... call_history=... rules=K reporter=...`
//!      line for first-line debugging (only in a process that will report, or
//!      when debug logging is on; non-reporting helpers stay silent).

use std::env;
use std::sync::OnceLock;

use ctor::ctor;

use clearml_snug_reporter::PollCallbacks;

#[ctor]
fn shim_init() {
    let call_history =
        env::var("CLEARML_SNUG_CALL_HISTORY").unwrap_or_else(|_| "off".to_string());

    // SAFETY: getpid is async-signal-safe and has no failure mode.
    let pid = unsafe { libc::getpid() };

    // Whitelist load (also seeds the ArcSwap for runtime hot-reload).
    let wl = crate::whitelist::initialize();

    // Seed the call-history mode from env BEFORE the reporter/poll thread starts
    // so hot-path reads see the right initial value.
    crate::control::set_initial_call_history_mode_from_env();

    // macOS: register the dyld add-image callback that rebinds each image's
    // SSL_* GOT slots onto our hooks (see hooks/macos.rs). Register EARLY —
    // before Python imports `ssl` — so interposition is in place by the first
    // TLS call. No-op on Linux (the LD_PRELOAD `#[no_mangle]` exports do the
    // interception there). (exit(3) is NOT rebound here — see the atexit drain
    // below.)
    #[cfg(target_os = "macos")]
    crate::hooks::macos::install();

    // macOS: the reliable exit-time drain. The fishhook `_exit` rewrite misses
    // CPython's libSystem-internal exit(3), so register an atexit(3) handler too.
    #[cfg(target_os = "macos")]
    crate::hooks::exit::install_atexit_drain();

    // Read the agent's handoff descriptor (creds + task + sink config) from the
    // inherited fd (a memfd on Linux, an unlinked temp file on macOS); `None` =>
    // no reporter, so events fall back to stderr (e.g. an operator running curl
    // under LD_PRELOAD / DYLD_INSERT_LIBRARIES directly).
    let reporter_status = match crate::descriptor_handoff::read_and_close() {
        Some(descriptor) => {
            // Install the backend self-hosts BEFORE any hook can fire, so the
            // task's own ClearML SDK traffic to its backend is excluded from
            // metering/usage for the connection's whole lifetime.
            crate::self_host::install(descriptor.self_hosts.clone());
            // The poll thread mutates the shim's atomics directly via these
            // function pointers. `set_call_history_mode` also drives the
            // edge-triggered dump: entering `dump` prints the backlog once, then
            // settles into `collect`.
            let poll_cb = PollCallbacks {
                set_call_history_mode: crate::call_history::set_call_history_mode_and_maybe_dump,
                // Apply a `_snug_whitelist` change: merge the additions onto the
                // immutable base whitelist and atomically hot-swap (affects new
                // connections). Clearing the property reverts to base.
                reload_whitelist: crate::whitelist::apply_whitelist_additions,
            };
            // DEFER the reporter: store the descriptor + callbacks now, but spawn
            // the reporter (+ its channel + poll thread) only when this process
            // first produces a metered usage event (reporter_handle::ensure_started,
            // driven from meter::emit). The shim loads into MANY processes that
            // never meter — multi-process desktop hosts (Electron/Chromium and
            // helper/sandbox children) — which abort when a reporter's threads +
            // outbound TLS appear inside them; deferral confines the reporter to
            // the process that actually sees LLM traffic.
            crate::reporter_handle::store_pending(descriptor, poll_cb);
            "deferred"
        }
        None => "stderr",
    };

    // The per-process ctor init line is DEBUG-ONLY (`snug_log!`): the shim is
    // preloaded into dozens of multi-process desktop hosts (Electron/Chromium and
    // helper/sandbox children) that hold a descriptor (reporter="deferred") but
    // never meter — the credential is broadcast
    // tree-wide, so descriptor presence can't tell the one metering process from
    // the idle helpers, and keying on it floods. A process instead announces
    // itself once, on its first actual metered event, from
    // `reporter_handle::ensure_started`.
    snug_log!(
        "[snug] init pid={} call_history={} rules={} self_hosts={} reporter={}",
        pid,
        call_history,
        wl.rules.len(),
        crate::self_host::current().len(),
        reporter_status
    );
}

// No #[dtor] here on purpose. The exit-time flush + reporter drain lives in
// `hooks/exit.rs` — a libc `exit(3)` interposer on Linux, an `atexit(3)` handler
// on macOS; see that module for the rationale.

pub fn project_id() -> &'static str {
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| env::var("CLEARML_PROJECT_ID").unwrap_or_default())
}

pub fn task_id() -> &'static str {
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| env::var("CLEARML_TASK_ID").unwrap_or_default())
}
