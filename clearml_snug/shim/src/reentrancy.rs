//! Thread-local reentrancy guard.
//!
//! A hook may indirectly cause the same SSL call again - for example if the
//! shim ever logs via a library that opens a TLS connection (it doesn't
//! today, but a future fallback path might), or if the allocator phones
//! home over HTTPS, our `SSL_write` hook would recursively invoke itself,
//! stack-overflow, and crash the host process.
//!
//! Hot path:
//!   1. `enter_hook()` returns `Some(HookGuard)` on the first entry per
//!      thread; the caller does observation work, then drops the guard.
//!   2. Subsequent calls on the same thread (re-entry) get `None` and
//!      fast-path straight to the real symbol with no observation.
//!
//! With `panic = "abort"` (see shim/Cargo.toml [profile.release]) a panic
//! aborts the whole process; Drop won't run, but we don't care because
//! the process is dead. The Drop here exists for the normal-return path.

use std::cell::Cell;

thread_local! {
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that clears the reentrancy flag on drop. Returned by
/// `enter_hook()` only on first entry.
pub struct HookGuard {
    // Field is private and unconstructible from outside this module, so
    // callers can't fabricate guards without going through `enter_hook()`.
    _private: (),
}

impl Drop for HookGuard {
    fn drop(&mut self) {
        IN_HOOK.with(|c| c.set(false));
    }
}

/// Try to enter the hook on this thread.
///
/// Returns `Some(HookGuard)` if we weren't already inside a hook on this
/// thread; the guard's `Drop` clears the flag. Returns `None` if we already
/// were - the caller must skip observation and call the real symbol
/// directly to avoid infinite recursion.
#[inline]
pub fn enter_hook() -> Option<HookGuard> {
    IN_HOOK.with(|c| {
        if c.get() {
            None
        } else {
            c.set(true);
            Some(HookGuard { _private: () })
        }
    })
}
