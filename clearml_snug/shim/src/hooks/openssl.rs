//! Linux OpenSSL interception via `LD_PRELOAD`.
//!
//! We EXPORT `#[no_mangle]` `SSL_*`; the host's `_ssl` binds to ours under
//! `LD_PRELOAD`, and we chain through to the real symbol via
//! `dlsym(RTLD_NEXT, ...)` (cached in a `OnceLock`). The observe + chain logic
//! itself lives in `hooks::ssl_body`, parameterized on the resolved real fn, so
//! it is shared verbatim with the macOS fishhook path (`hooks/macos.rs`).
//!
//! This module is compiled on Linux only (see `hooks/mod.rs`); macOS exports no
//! `SSL_*` and interposes via dyld GOT rewriting instead.
//!
//! Loud-failure policy: if `dlsym` returns NULL even after the versioned +
//! explicit-handle fallbacks, we `libc::abort()`. A shim that can't find its
//! host's real function is broken in a way silent corruption would only worsen.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::OnceLock;

use crate::hooks::ssl_body::{
    self, SslFreeFn, SslReadExFn, SslReadFn, SslWriteExFn, SslWriteFn,
};

static REAL_SSL_WRITE: OnceLock<SslWriteFn> = OnceLock::new();
static REAL_SSL_READ: OnceLock<SslReadFn> = OnceLock::new();
static REAL_SSL_FREE: OnceLock<SslFreeFn> = OnceLock::new();
static REAL_SSL_WRITE_EX: OnceLock<SslWriteExFn> = OnceLock::new();
static REAL_SSL_READ_EX: OnceLock<SslReadExFn> = OnceLock::new();

/// Versioned-symbol lookup over known OpenSSL version tags. This exists because
/// OpenSSL 3.x on **glibc** exports the SSL_* symbols ONLY as versioned
/// (`SSL_free@OPENSSL_3.0.0`), so unversioned `dlsym` returns NULL and we must
/// fall back to `dlvsym`.
///
/// `dlvsym` is a **GNU (glibc) extension** — it does not exist on musl. On musl,
/// libssl's symbols are unversioned (musl has no ELF symbol versioning), so the
/// plain `dlsym` the caller already tried resolves them; this helper is a no-op
/// there (returns NULL), letting the same code compile + work for both the
/// glibc (`cargo build`) and musl (`cargo zigbuild --target *-musl`, the shipped
/// wheel) targets.
#[cfg(target_env = "gnu")]
unsafe fn dlvsym_versioned(handle: *mut c_void, name: *const libc::c_char) -> *mut c_void {
    // Newest OpenSSL first - if both libraries are somehow in the process, the
    // newer one is what _ssl.so was linked against and what we want.
    const OPENSSL_VERSIONS: &[&[u8]] = &[b"OPENSSL_3.0.0\0", b"OPENSSL_1_1_0\0"];
    for v in OPENSSL_VERSIONS {
        let p = libc::dlvsym(handle, name, v.as_ptr().cast::<libc::c_char>());
        if !p.is_null() {
            return p;
        }
    }
    std::ptr::null_mut()
}

#[cfg(not(target_env = "gnu"))]
unsafe fn dlvsym_versioned(_handle: *mut c_void, _name: *const libc::c_char) -> *mut c_void {
    // musl: no dlvsym, and libssl symbols are unversioned -> plain dlsym suffices.
    std::ptr::null_mut()
}

/// Resolve a symbol via `dlsym(RTLD_NEXT, ...)`, with a versioned-`dlvsym`
/// fallback (glibc only, see `dlvsym_versioned`) for OpenSSL 3.x's versioned-only
/// exports, then an explicit `dlopen("libssl.so.X", RTLD_LAZY|RTLD_NOLOAD)`
/// handle.
///
/// We deliberately DO NOT fall back to `RTLD_DEFAULT`. `RTLD_DEFAULT`
/// searches the whole load order including this shim, which exports the
/// same name as the host's SSL_* — chaining through it would re-enter our
/// own hook and (because `enter_hook()` returns None on the same thread)
/// jump straight back through `real_*` again. The compiler is happy to
/// tail-call that, so the result is a syscall-less, stackless infinite
/// loop at 100% CPU. Seen in the wild against `python:3.10`'s `_ssl.so`,
/// which links libssl.so.3 (versioned-only) — RTLD_NEXT misses, fallback
/// finds us, and `SSL_free` from connection close spins forever during
/// task shutdown.
unsafe fn resolve_or_abort(name: &[u8]) -> *mut c_void {
    debug_assert!(
        name.last() == Some(&0u8),
        "resolve_or_abort needs NUL-terminated name"
    );

    let n = name.as_ptr().cast::<libc::c_char>();

    // 1. Unversioned RTLD_NEXT (OpenSSL 1.0 / un-versioned exports / all of musl).
    let p = libc::dlsym(libc::RTLD_NEXT, n);
    if !p.is_null() {
        return p;
    }

    // 2. Versioned RTLD_NEXT (glibc OpenSSL 3.x versioned-only exports).
    let p = dlvsym_versioned(libc::RTLD_NEXT, n);
    if !p.is_null() {
        return p;
    }

    // 3. Explicit-handle fallback. `RTLD_NOLOAD` means "give me a handle to
    //    an already-loaded library; don't load anything new" - so a NULL
    //    return just means the lib isn't in the process and we move on.
    const LIBSSL_CANDIDATES: &[&[u8]] = &[b"libssl.so.3\0", b"libssl.so.1.1\0"];
    for lib in LIBSSL_CANDIDATES {
        let h = libc::dlopen(
            lib.as_ptr().cast::<libc::c_char>(),
            libc::RTLD_LAZY | libc::RTLD_NOLOAD,
        );
        if h.is_null() {
            continue;
        }
        let p = libc::dlsym(h, n);
        if !p.is_null() {
            return p;
        }
        let p = dlvsym_versioned(h, n);
        if !p.is_null() {
            return p;
        }
    }

    let display = std::str::from_utf8(&name[..name.len().saturating_sub(1)])
        .unwrap_or("<non-utf8 symbol>");
    snug_err!(
        "[snug] FATAL: cannot resolve real {} (RTLD_NEXT unversioned + versioned + libssl handle all failed)",
        display
    );
    libc::abort();
}

fn real_ssl_write() -> SslWriteFn {
    *REAL_SSL_WRITE.get_or_init(|| unsafe {
        std::mem::transmute::<*mut c_void, SslWriteFn>(resolve_or_abort(b"SSL_write\0"))
    })
}

fn real_ssl_read() -> SslReadFn {
    *REAL_SSL_READ.get_or_init(|| unsafe {
        std::mem::transmute::<*mut c_void, SslReadFn>(resolve_or_abort(b"SSL_read\0"))
    })
}

fn real_ssl_free() -> SslFreeFn {
    *REAL_SSL_FREE.get_or_init(|| unsafe {
        std::mem::transmute::<*mut c_void, SslFreeFn>(resolve_or_abort(b"SSL_free\0"))
    })
}

fn real_ssl_write_ex() -> SslWriteExFn {
    *REAL_SSL_WRITE_EX.get_or_init(|| unsafe {
        std::mem::transmute::<*mut c_void, SslWriteExFn>(resolve_or_abort(b"SSL_write_ex\0"))
    })
}

fn real_ssl_read_ex() -> SslReadExFn {
    *REAL_SSL_READ_EX.get_or_init(|| unsafe {
        std::mem::transmute::<*mut c_void, SslReadExFn>(resolve_or_abort(b"SSL_read_ex\0"))
    })
}

/// `int SSL_write(SSL *ssl, const void *buf, int num);`
#[no_mangle]
pub unsafe extern "C" fn SSL_write(ssl: *mut c_void, buf: *const c_void, num: c_int) -> c_int {
    ssl_body::handle_write(real_ssl_write(), ssl, buf, num)
}

/// `int SSL_read(SSL *ssl, void *buf, int num);`
#[no_mangle]
pub unsafe extern "C" fn SSL_read(ssl: *mut c_void, buf: *mut c_void, num: c_int) -> c_int {
    ssl_body::handle_read(real_ssl_read(), ssl, buf, num)
}

/// `void SSL_free(SSL *ssl);`
#[no_mangle]
pub unsafe extern "C" fn SSL_free(ssl: *mut c_void) {
    ssl_body::handle_free(real_ssl_free(), ssl)
}

/// `int SSL_write_ex(SSL *ssl, const void *buf, size_t num, size_t *written);`
///
/// OpenSSL 1.1.1+ API. CPython 3.10+'s `_ssl.so` calls this instead of
/// `SSL_write`, so the header-injection + request-start-observation paths only
/// fire for Python tasks when this hook is present.
#[no_mangle]
pub unsafe extern "C" fn SSL_write_ex(
    ssl: *mut c_void,
    buf: *const c_void,
    num: libc::size_t,
    written: *mut libc::size_t,
) -> c_int {
    ssl_body::handle_write_ex(real_ssl_write_ex(), ssl, buf, num, written)
}

/// `int SSL_read_ex(SSL *ssl, void *buf, size_t num, size_t *readbytes);`
///
/// OpenSSL 1.1.1+ API. Without this hook, Python (3.10+) tasks emit zero
/// `BytesObserved` events on the rx direction.
#[no_mangle]
pub unsafe extern "C" fn SSL_read_ex(
    ssl: *mut c_void,
    buf: *mut c_void,
    num: libc::size_t,
    readbytes: *mut libc::size_t,
) -> c_int {
    ssl_body::handle_read_ex(real_ssl_read_ex(), ssl, buf, num, readbytes)
}
