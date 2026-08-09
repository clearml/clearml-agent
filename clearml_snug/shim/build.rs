//! Build script for the shim crate.
//!
//! Empty by design. `exports.map` is NOT applied to the link line:
//! rustc emits its own anonymous version script for cdylibs, and the GNU
//! linker rejects a second, named version script stacked on top of it
//! (rust-lang/rust#123464). It isn't needed anyway — rustc hides Rust
//! std/allocator/panic symbols by default in cdylibs, so only
//! `#[no_mangle] pub extern "C"` functions are exported: exactly
//! `SSL_write`, `SSL_read`, `SSL_free` and nothing else, which is the
//! outcome a version script would enforce.
//!
//! `exports.map` is kept as documentation of the v1 contract. A CI
//! `nm -D` guard enforces the invariant on every build, failing on any
//! global `T` symbol outside the hook set.

fn main() {
    println!("cargo:rerun-if-changed=exports.map");
}
