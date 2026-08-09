//! Language-agnostic process-exit interposer via libc `exit(3)`.
//!
//! Together with `state::observe_free` (driven by `SSL_free`), this is the
//! shim's exit-time flush. `observe_free` handles the common case where the host
//! closes its SSL connections before exit; `exit()` handles connections still in
//! a keep-alive pool at process teardown. Both call `flush_all_pending_requests`,
//! so the emitted event is identical.
//!
//! The drain itself (`exit_drain`) is shared; how we get invoked differs:
//!   * **Linux**: export `#[no_mangle] exit`; `__libc_start_main` calls `exit`
//!     through a PLT relocation that our LD_PRELOAD export shadows, and we chain
//!     via `dlsym(RTLD_NEXT, "exit")`.
//!   * **macOS**: an `atexit(3)` handler (`install_atexit_drain`, registered
//!     from the ctor) runs `exit_drain()` from inside `exit(3)`. We do NOT use
//!     the fishhook here: CPython's final `exit(3)` originates inside the
//!     libSystem umbrella (an intra-dylib call, not an interposable GOT slot),
//!     so a `_exit` GOT rewrite misses it — verified empirically (the fishhook
//!     never fired; the dropped trailing request only reappeared via atexit).
//!
//! Running the flush BEFORE chaining means the Rust runtime, parking_lot, and
//! the reporter thread are still alive.
//!
//! ## What we deliberately don't hook
//!
//!   * Python-finalize hooks (`#[ctor::dtor]`, `Py_FinalizeEx`, `Py_BytesMain`)
//!     - all tried, all rejected. Either uninvokable under the preload mechanism
//!     or strictly redundant with `exit(3)`. One mechanism covers Python and
//!     non-Python hosts.
//!   * `_exit(2)` / `_Exit()` / `abort(3)` / uncaught signals - by design these
//!     terminate immediately without running atexit handlers. Callers reach them
//!     when they explicitly want to skip cleanup; hooking would impose work on a
//!     path that asked for none. Rare in practice.

use std::time::Duration;

use crate::reentrancy::enter_hook;

/// Upper bound on how long the exit drain waits for the reporter to drain +
/// final-flush before giving up. Bounded so a hung backend can't wedge process
/// teardown; comfortably under the docker drain budget.
const EXIT_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Flush pending state + drain/join the in-process reporter so the just-enqueued
/// events are reported (not merely queued) before exit. Shared by the Linux
/// `exit` interposer and the macOS `atexit(3)` handler.
pub(crate) unsafe fn exit_drain() {
    // Run once. The drain can be reached from more than one path (the Linux
    // `exit` export; on macOS the atexit handler below) — and we never want to
    // flush/join twice. flush_and_join is already idempotent, but guarding here
    // keeps the whole drain single-shot.
    use std::sync::atomic::{AtomicBool, Ordering};
    static DRAINED: AtomicBool = AtomicBool::new(false);
    if DRAINED.swap(true, Ordering::SeqCst) {
        return;
    }

    // Reentrancy guard scoped to the flush only - skip if already held (would
    // mean exit() was called from inside an SSL hook on this thread, an unusual
    // case where losing the trailing event beats corrupting shared state). This
    // MUST run first: it enqueues the final keep-alive RequestCompleted event(s)
    // that the reporter drain below then actually POSTs.
    if let Some(_guard) = enter_hook() {
        crate::state::flush_all_pending_requests();
    }

    // Bounded so a hung backend can't wedge teardown. Runs OUTSIDE the
    // reentrancy guard: the reporter is a separate thread doing its own rustls
    // TLS, which never re-enters these OpenSSL hooks. No-op if no reporter was
    // started. Only graceful exit(3) drains — _exit/abort/signals skip this by
    // design (the crash-resilience trade-off of in-process reporting).
    crate::reporter_handle::flush_and_join(EXIT_DRAIN_TIMEOUT);
}

// ---- Linux: exit(3) interposer via LD_PRELOAD export + dlsym(RTLD_NEXT) ----

#[cfg(target_os = "linux")]
type ExitFn = unsafe extern "C" fn(libc::c_int) -> !;

#[cfg(target_os = "linux")]
static REAL_EXIT: std::sync::OnceLock<Option<ExitFn>> = std::sync::OnceLock::new();

/// Resolve `exit(3)` from the next library in the dlsym chain. Returns `None`
/// only if libc is unreachable through `RTLD_NEXT` - defensive so we never
/// recurse into our own hook.
#[cfg(target_os = "linux")]
fn real_exit() -> Option<ExitFn> {
    *REAL_EXIT.get_or_init(|| unsafe {
        let p = libc::dlsym(libc::RTLD_NEXT, b"exit\0".as_ptr() as *const _);
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute::<*mut std::os::raw::c_void, ExitFn>(p))
        }
    })
}

/// `void exit(int status);` from libc. Flush + drain, then chain to the real
/// `exit()`.
#[cfg(target_os = "linux")]
#[no_mangle]
pub unsafe extern "C" fn exit(status: libc::c_int) -> ! {
    exit_drain();
    match real_exit() {
        Some(f) => f(status),
        // Fallback if dlsym failed (shouldn't happen - if our hook is
        // PLT-reachable, the real symbol must be too). Use raw _exit(2) so the
        // process still terminates with the right code; skips atexit, but
        // that's better than looping.
        None => libc::_exit(status),
    }
}

// ---- macOS: drain via atexit(3) (reliable) ----
//
// On macOS the fishhook `_exit` GOT rewrite does NOT catch CPython's final
// `exit(3)` — that call originates inside the libSystem umbrella (an intra-dylib
// call, not an interposable GOT slot), so the rebind misses it and the trailing
// request is never drained. `atexit(3)` is the reliable mechanism: libc runs
// registered handlers from inside `exit(3)` regardless of who called it, while
// the reporter thread is still alive. Same coverage as the intended exit(3)
// hook — _exit(2)/abort/signals still skip it, by design.

#[cfg(target_os = "macos")]
extern "C" fn snug_atexit() {
    unsafe { exit_drain() };
}

/// Register the atexit drain. Called once from the ctor on macOS.
#[cfg(target_os = "macos")]
pub fn install_atexit_drain() {
    unsafe {
        libc::atexit(snug_atexit);
    }
}
