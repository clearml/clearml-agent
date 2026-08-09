//! Shared OpenSSL hook bodies, parameterized on the resolved REAL function.
//!
//! The observe + chain logic is byte-for-byte identical on every platform; only
//! (a) how we get *called* and (b) how the real function is resolved differ:
//!
//!   * **Linux** (`hooks/openssl.rs`): we EXPORT `#[no_mangle]` `SSL_*` and the
//!     host's `_ssl` binds to them under `LD_PRELOAD`; the real is resolved via
//!     `dlsym(RTLD_NEXT, ...)`.
//!   * **macOS** (`hooks/macos.rs`): we export nothing; a dyld add-image
//!     callback rewrites each image's `SSL_*` GOT slot to our private `snug_*`
//!     functions, and the real is the address read straight out of the slot.
//!
//! Both funnel into the `handle_*` functions here, so the (subtle) header-splice
//! / out-param contracts live in exactly one place.

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::reentrancy::enter_hook;
use crate::state;

pub type SslWriteFn = unsafe extern "C" fn(*mut c_void, *const c_void, c_int) -> c_int;
pub type SslReadFn = unsafe extern "C" fn(*mut c_void, *mut c_void, c_int) -> c_int;
pub type SslFreeFn = unsafe extern "C" fn(*mut c_void);

// OpenSSL 1.1.1+ added the `_ex` variants - same semantics as SSL_read /
// SSL_write but with `size_t` lengths and out-param bytes-actually-IO'd.
// Python's `_ssl.so` (CPython 3.10+ built against OpenSSL 1.1.1+ or 3.x)
// imports ONLY `SSL_read_ex` / `SSL_write_ex`; the legacy `SSL_read` /
// `SSL_write` symbols are never called from Python's SSL data path, so
// without these hooks Python tasks generate zero SNUG events.
pub type SslWriteExFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, libc::size_t, *mut libc::size_t) -> c_int;
pub type SslReadExFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, libc::size_t, *mut libc::size_t) -> c_int;

/// `int SSL_write(SSL *ssl, const void *buf, int num);`
///
/// Splice contract: if `state::observe_write` returns a spliced buffer
/// (whitelist match + inject_headers=true), we call the real `SSL_write` with
/// that longer buffer but report the *original* `num` back to the libssl caller.
/// Reporting the larger length would make higher layers think they've written
/// more than they asked - assertion failure territory.
#[inline]
pub(crate) unsafe fn handle_write(
    real: SslWriteFn,
    ssl: *mut c_void,
    buf: *const c_void,
    num: c_int,
) -> c_int {
    let _guard = match enter_hook() {
        Some(g) => g,
        None => return real(ssl, buf, num),
    };

    let spliced = if !buf.is_null() && num > 0 {
        let slice = std::slice::from_raw_parts(buf as *const u8, num as usize);
        state::observe_write(ssl as usize, slice)
    } else {
        None
    };

    match spliced {
        Some(buf2) => {
            // Contract: caller MUST see the original `num`, not the
            // post-splice length.
            let _ = real(ssl, buf2.as_ptr() as *const c_void, buf2.len() as c_int);
            num
        }
        None => real(ssl, buf, num),
    }
}

/// `int SSL_read(SSL *ssl, void *buf, int num);`
///
/// Observation happens AFTER the real call: we report the number of bytes
/// actually delivered to the caller, not the number requested. Negative return
/// values (WANT_READ, error) are passed through unobserved.
#[inline]
pub(crate) unsafe fn handle_read(
    real: SslReadFn,
    ssl: *mut c_void,
    buf: *mut c_void,
    num: c_int,
) -> c_int {
    let _guard = match enter_hook() {
        Some(g) => g,
        None => return real(ssl, buf, num),
    };

    let result = real(ssl, buf, num);
    if result > 0 && !buf.is_null() {
        // SAFETY: real returned a non-negative count of bytes it wrote into buf;
        // reading exactly `result` bytes back out matches what the caller will
        // see. We pass the slice (rather than just the length) so observe_read
        // can capture the full response bytes for call-history when on.
        let slice = std::slice::from_raw_parts(buf as *const u8, result as usize);
        state::observe_read(ssl as usize, slice);
    }
    result
}

/// `void SSL_free(SSL *ssl);`
///
/// Observe BEFORE the real free so the conn-state entry is evicted and any
/// `RequestCompleted` emitted while the pointer is still meaningful. libssl may
/// immediately reuse the `SSL*` allocation; observing afterwards would leak the
/// old conn's counts into the new one.
#[inline]
pub(crate) unsafe fn handle_free(real: SslFreeFn, ssl: *mut c_void) {
    let _guard = match enter_hook() {
        Some(g) => g,
        None => {
            real(ssl);
            return;
        }
    };

    state::observe_free(ssl as usize);
    real(ssl);
}

/// `int SSL_write_ex(SSL *ssl, const void *buf, size_t num, size_t *written);`
///
/// OpenSSL 1.1.1+ API. CPython 3.10+'s `_ssl` calls this instead of
/// `SSL_write`. On success we lie about `*written` exactly like `SSL_write`'s
/// "return original num" contract when splicing.
#[inline]
pub(crate) unsafe fn handle_write_ex(
    real: SslWriteExFn,
    ssl: *mut c_void,
    buf: *const c_void,
    num: libc::size_t,
    written: *mut libc::size_t,
) -> c_int {
    let _guard = match enter_hook() {
        Some(g) => g,
        None => return real(ssl, buf, num, written),
    };

    let spliced = if !buf.is_null() && num > 0 {
        let slice = std::slice::from_raw_parts(buf as *const u8, num);
        state::observe_write(ssl as usize, slice)
    } else {
        None
    };

    match spliced {
        Some(buf2) => {
            // Call the real fn with the longer buffer. Capture `inner_written`
            // locally and (on success) report `num` to the caller via `*written`
            // - mirrors SSL_write's "return original num when spliced" rule. If
            // the real call wrote fewer than buf2.len() bytes we can't faithfully
            // know how many were the user's vs the splice header; reporting `num`
            // keeps higher layers' bookkeeping sane.
            let mut inner_written: libc::size_t = 0;
            let rc = real(
                ssl,
                buf2.as_ptr() as *const c_void,
                buf2.len(),
                &mut inner_written as *mut libc::size_t,
            );
            if !written.is_null() && rc == 1 {
                *written = num;
            } else if !written.is_null() {
                *written = inner_written;
            }
            rc
        }
        None => real(ssl, buf, num, written),
    }
}

/// `int SSL_read_ex(SSL *ssl, void *buf, size_t num, size_t *readbytes);`
///
/// OpenSSL 1.1.1+ API. Observation happens AFTER the real call, using
/// `*readbytes` for how many bytes were actually delivered.
#[inline]
pub(crate) unsafe fn handle_read_ex(
    real: SslReadExFn,
    ssl: *mut c_void,
    buf: *mut c_void,
    num: libc::size_t,
    readbytes: *mut libc::size_t,
) -> c_int {
    let _guard = match enter_hook() {
        Some(g) => g,
        None => return real(ssl, buf, num, readbytes),
    };

    let rc = real(ssl, buf, num, readbytes);
    if rc == 1 && !buf.is_null() && !readbytes.is_null() {
        let n = *readbytes;
        if n > 0 {
            let slice = std::slice::from_raw_parts(buf as *const u8, n);
            state::observe_read(ssl as usize, slice);
        }
    }
    rc
}
