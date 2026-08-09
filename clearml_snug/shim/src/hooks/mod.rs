//! Hook implementations.
//!
//! * `ssl_body` - the shared SSL_{read,write,free} (+ _ex variants) hook bodies,
//!   parameterized on the resolved real fn. The observability surface.
//! * `openssl`  - **Linux** interception: `#[no_mangle]` `SSL_*` exports bound
//!   via `LD_PRELOAD`, chaining through `dlsym(RTLD_NEXT)`.
//! * `macos`    - **macOS** interception: fishhook-style GOT rebinding via a
//!   dyld add-image callback (no exported symbols).
//! * `exit`     - exit-time flush for connections still in a keep-alive pool
//!   when the process terminates. The drain (`exit_drain`) is shared; it is
//!   invoked by the Linux `#[no_mangle] exit` export and, on macOS, by an
//!   `atexit(3)` handler (a `_exit` GOT rebind misses CPython's
//!   libSystem-internal exit(3); see hooks/exit.rs).

pub mod exit;
pub mod ssl_body;

#[cfg(target_os = "linux")]
pub mod openssl;

#[cfg(target_os = "macos")]
pub mod macos;
