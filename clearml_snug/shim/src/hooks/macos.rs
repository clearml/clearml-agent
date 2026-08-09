//! macOS OpenSSL interception via fishhook-style GOT rebinding.
//!
//! (The exit-time reporter drain is NOT done here — a `_exit` GOT rewrite never
//! fires for CPython's libSystem-internal exit(3); it runs from an `atexit(3)`
//! handler in hooks/exit.rs instead.)
//!
//! On modern dyld (tested on macOS 26 / dyld4) the simpler mechanisms are all
//! unusable, verified with tiny C tests:
//!   * `__DATA,__interpose` (`DYLD_INTERPOSE`) — the tuple's replacee
//!     `&SSL_read_ex` is an undefined symbol that must bind when OUR dylib
//!     loads, but `DYLD_INSERT_LIBRARIES` loads us BEFORE Python lazily loads
//!     libssl ⇒ `dyld: symbol not found` at load. (Works only for always-present
//!     libs like libSystem.)
//!   * `DYLD_FORCE_FLAT_NAMESPACE=1` + exported `SSL_*` — dyld4 does NOT reroute
//!     `_ssl`'s import to our export. Vestigial.
//!   * `dyld_dynamic_interpose()` — symbol present but a no-op.
//!
//! So we rebind directly: a `_dyld_register_func_for_add_image` callback fires
//! per loaded image (synchronously for every already-loaded image at
//! registration, then for each later one). For each symbol-pointer section
//! (`__got` / `__la_symbol_ptr` in `__DATA` / `__DATA_CONST`) we find the slot
//! for an `SSL_*` symbol, read the real (already-bound libssl) address straight
//! out of the slot — so NO `dlsym`, hence no dyld-lock re-entrancy inside the
//! callback — `mprotect` the page RW, and overwrite the slot with our hook. The
//! hook bodies are the shared `ssl_body::*` (identical to the Linux path); only
//! the call-in + real-resolution differ.
//!
//! This dylib exports ZERO `SSL_*` symbols (the nm guard in
//! snug_validate_macos.sh asserts the inverse of the Linux check).

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::hooks::ssl_body::{
    self, SslFreeFn, SslReadExFn, SslReadFn, SslWriteExFn, SslWriteFn,
};

// Real addresses, captured from each image's bound symbol pointer the first
// time we see one (0 = not captured yet).
static REAL_WRITE: AtomicUsize = AtomicUsize::new(0);
static REAL_READ: AtomicUsize = AtomicUsize::new(0);
static REAL_FREE: AtomicUsize = AtomicUsize::new(0);
static REAL_WRITE_EX: AtomicUsize = AtomicUsize::new(0);
static REAL_READ_EX: AtomicUsize = AtomicUsize::new(0);
// NOTE: we do NOT fishhook `_exit` here. The exit-time reporter drain runs from
// an `atexit(3)` handler instead (hooks/exit.rs::install_atexit_drain) — a
// `_exit` GOT rewrite never fires for CPython's libSystem-internal exit(3)
// (verified empirically: the trailing request was dropped until atexit caught
// it). The raw syscall `_exit(2)` (Mach-O `__exit`) is skipped by design.

// Our replacements: load the captured real and run the shared body. The real is
// stored (Release) before the slot is rewritten, so any later call through the
// rebound slot sees it (Acquire).
unsafe extern "C" fn snug_write(ssl: *mut c_void, buf: *const c_void, num: c_int) -> c_int {
    let real: SslWriteFn = std::mem::transmute(REAL_WRITE.load(Ordering::Acquire));
    ssl_body::handle_write(real, ssl, buf, num)
}
unsafe extern "C" fn snug_read(ssl: *mut c_void, buf: *mut c_void, num: c_int) -> c_int {
    let real: SslReadFn = std::mem::transmute(REAL_READ.load(Ordering::Acquire));
    ssl_body::handle_read(real, ssl, buf, num)
}
unsafe extern "C" fn snug_free(ssl: *mut c_void) {
    let real: SslFreeFn = std::mem::transmute(REAL_FREE.load(Ordering::Acquire));
    ssl_body::handle_free(real, ssl)
}
unsafe extern "C" fn snug_write_ex(
    ssl: *mut c_void,
    buf: *const c_void,
    num: libc::size_t,
    written: *mut libc::size_t,
) -> c_int {
    let real: SslWriteExFn = std::mem::transmute(REAL_WRITE_EX.load(Ordering::Acquire));
    ssl_body::handle_write_ex(real, ssl, buf, num, written)
}
unsafe extern "C" fn snug_read_ex(
    ssl: *mut c_void,
    buf: *mut c_void,
    num: libc::size_t,
    readbytes: *mut libc::size_t,
) -> c_int {
    let real: SslReadExFn = std::mem::transmute(REAL_READ_EX.load(Ordering::Acquire));
    ssl_body::handle_read_ex(real, ssl, buf, num, readbytes)
}

/// Map a mangled symbol name (Mach-O leading `_`) to its (captured-real slot,
/// our replacement address). `None` for anything we don't hook. The replacement
/// is coerced through its fn-pointer type before `as usize` (a direct
/// fn-item-to-integer cast warns).
fn target(name: &[u8]) -> Option<(&'static AtomicUsize, usize)> {
    match name {
        b"_SSL_write" => Some((&REAL_WRITE, snug_write as SslWriteFn as usize)),
        b"_SSL_read" => Some((&REAL_READ, snug_read as SslReadFn as usize)),
        b"_SSL_free" => Some((&REAL_FREE, snug_free as SslFreeFn as usize)),
        b"_SSL_write_ex" => Some((&REAL_WRITE_EX, snug_write_ex as SslWriteExFn as usize)),
        b"_SSL_read_ex" => Some((&REAL_READ_EX, snug_read_ex as SslReadExFn as usize)),
        _ => None,
    }
}

// ---- Minimal Mach-O definitions (mach-o/loader.h, mach-o/nlist.h) ----------
const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const SECTION_TYPE: u32 = 0xff;
const S_NON_LAZY_SYMBOL_POINTERS: u32 = 6;
const S_LAZY_SYMBOL_POINTERS: u32 = 7;
const INDIRECT_SYMBOL_LOCAL: u32 = 0x8000_0000;
const INDIRECT_SYMBOL_ABS: u32 = 0x4000_0000;

#[repr(C)]
struct MachHeader64 {
    magic: u32,
    cputype: i32,
    cpusubtype: i32,
    filetype: u32,
    ncmds: u32,
    sizeofcmds: u32,
    flags: u32,
    reserved: u32,
}
#[repr(C)]
struct LoadCommand {
    cmd: u32,
    cmdsize: u32,
}
#[repr(C)]
struct SegmentCommand64 {
    cmd: u32,
    cmdsize: u32,
    segname: [u8; 16],
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    maxprot: i32,
    initprot: i32,
    nsects: u32,
    flags: u32,
}
#[repr(C)]
struct Section64 {
    sectname: [u8; 16],
    segname: [u8; 16],
    addr: u64,
    size: u64,
    offset: u32,
    align: u32,
    reloff: u32,
    nreloc: u32,
    flags: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
}
#[repr(C)]
struct SymtabCommand {
    cmd: u32,
    cmdsize: u32,
    symoff: u32,
    nsyms: u32,
    stroff: u32,
    strsize: u32,
}
#[repr(C)]
struct DysymtabCommand {
    cmd: u32,
    cmdsize: u32,
    ilocalsym: u32,
    nlocalsym: u32,
    iextdefsym: u32,
    nextdefsym: u32,
    iundefsym: u32,
    nundefsym: u32,
    tocoff: u32,
    ntoc: u32,
    modtaboff: u32,
    nmodtab: u32,
    extrefsymoff: u32,
    nextrefsyms: u32,
    indirectsymoff: u32,
    nindirectsyms: u32,
    extreloff: u32,
    nextrel: u32,
    locreloff: u32,
    nlocrel: u32,
}
#[repr(C)]
struct Nlist64 {
    n_strx: u32,
    n_type: u8,
    n_sect: u8,
    n_desc: u16,
    n_value: u64,
}

fn segname_is(seg: &[u8; 16], want: &[u8]) -> bool {
    let len = seg.iter().position(|&b| b == 0).unwrap_or(16);
    &seg[..len] == want
}

unsafe fn cstr<'a>(p: *const u8) -> &'a [u8] {
    let mut len = 0usize;
    while len < 256 && *p.add(len) != 0 {
        len += 1;
    }
    std::slice::from_raw_parts(p, len)
}

unsafe fn make_writable(addr: usize) -> bool {
    let pagesize = libc::sysconf(libc::_SC_PAGESIZE);
    if pagesize <= 0 {
        return false;
    }
    let pagesize = pagesize as usize;
    let page = addr & !(pagesize - 1);
    // We leave the page writable after the rewrite. These are data pages
    // (`__got` / `__la_symbol_ptr`), not executable code; fishhook historically
    // does the same. Restoring per-page protection would require tracking the
    // original prot per slot for no functional gain here.
    libc::mprotect(
        page as *mut c_void,
        pagesize,
        libc::PROT_READ | libc::PROT_WRITE,
    ) == 0
}

#[allow(clippy::too_many_arguments)]
unsafe fn rewrite_section(
    sect: &Section64,
    slide: isize,
    symtab: *const Nlist64,
    nsyms: usize,
    strtab: *const u8,
    strsize: usize,
    indirect: *const u32,
    nindirect: usize,
) {
    let count = (sect.size / std::mem::size_of::<usize>() as u64) as usize;
    let slots = (slide as usize).wrapping_add(sect.addr as usize) as *mut usize;
    let ind_base = sect.reserved1 as usize;
    for k in 0..count {
        if ind_base + k >= nindirect {
            break;
        }
        let symidx = *indirect.add(ind_base + k);
        if symidx & (INDIRECT_SYMBOL_ABS | INDIRECT_SYMBOL_LOCAL) != 0 {
            continue;
        }
        if symidx as usize >= nsyms {
            continue;
        }
        let strx = (*symtab.add(symidx as usize)).n_strx as usize;
        if strx >= strsize {
            continue;
        }
        let name = cstr(strtab.add(strx));
        if let Some((real_slot, repl)) = target(name) {
            let slot = slots.add(k);
            let cur = *slot;
            if cur == repl || cur == 0 {
                continue; // already ours, or unbound
            }
            // Capture the real (bound libssl/libc) address once.
            let _ = real_slot.compare_exchange(0, cur, Ordering::Release, Ordering::Relaxed);
            if make_writable(slot as usize) {
                std::ptr::write(slot, repl);
            }
        }
    }
}

unsafe fn rebind_image(mh: *const MachHeader64, slide: isize) {
    let ncmds = (*mh).ncmds;
    // Pass 1: locate __LINKEDIT, LC_SYMTAB, LC_DYSYMTAB.
    let mut linkedit: *const SegmentCommand64 = std::ptr::null();
    let mut symc: *const SymtabCommand = std::ptr::null();
    let mut dysc: *const DysymtabCommand = std::ptr::null();
    let mut lc = (mh as *const u8).add(std::mem::size_of::<MachHeader64>()) as *const LoadCommand;
    for _ in 0..ncmds {
        match (*lc).cmd {
            LC_SEGMENT_64 => {
                let seg = lc as *const SegmentCommand64;
                if segname_is(&(*seg).segname, b"__LINKEDIT") {
                    linkedit = seg;
                }
            }
            LC_SYMTAB => symc = lc as *const SymtabCommand,
            LC_DYSYMTAB => dysc = lc as *const DysymtabCommand,
            _ => {}
        }
        lc = (lc as *const u8).add((*lc).cmdsize as usize) as *const LoadCommand;
    }
    if linkedit.is_null() || symc.is_null() || dysc.is_null() {
        return;
    }
    let base = (slide as usize)
        .wrapping_add((*linkedit).vmaddr as usize)
        .wrapping_sub((*linkedit).fileoff as usize);
    let symtab = (base + (*symc).symoff as usize) as *const Nlist64;
    let nsyms = (*symc).nsyms as usize;
    let strtab = (base + (*symc).stroff as usize) as *const u8;
    let strsize = (*symc).strsize as usize;
    let indirect = (base + (*dysc).indirectsymoff as usize) as *const u32;
    let nindirect = (*dysc).nindirectsyms as usize;

    // Pass 2: rewrite symbol-pointer sections in __DATA / __DATA_CONST.
    let mut lc = (mh as *const u8).add(std::mem::size_of::<MachHeader64>()) as *const LoadCommand;
    for _ in 0..ncmds {
        if (*lc).cmd == LC_SEGMENT_64 {
            let seg = lc as *const SegmentCommand64;
            if segname_is(&(*seg).segname, b"__DATA") || segname_is(&(*seg).segname, b"__DATA_CONST")
            {
                let mut sect = (seg as *const u8).add(std::mem::size_of::<SegmentCommand64>())
                    as *const Section64;
                for _ in 0..(*seg).nsects {
                    let typ = (*sect).flags & SECTION_TYPE;
                    if typ == S_LAZY_SYMBOL_POINTERS || typ == S_NON_LAZY_SYMBOL_POINTERS {
                        rewrite_section(
                            &*sect, slide, symtab, nsyms, strtab, strsize, indirect, nindirect,
                        );
                    }
                    sect = (sect as *const u8).add(std::mem::size_of::<Section64>())
                        as *const Section64;
                }
            }
        }
        lc = (lc as *const u8).add((*lc).cmdsize as usize) as *const LoadCommand;
    }
}

extern "C" fn on_add_image(mh: *const MachHeader64, slide: isize) {
    // No dlopen/dlsym here -> no dyld-lock re-entrancy. We only read the image's
    // own memory and mprotect+write the matched slots.
    unsafe { rebind_image(mh, slide) };
}

extern "C" {
    // Calls `func` for every currently-loaded image (synchronously, at
    // registration) and every image loaded afterwards.
    fn _dyld_register_func_for_add_image(func: extern "C" fn(*const MachHeader64, isize));
}

/// Register the add-image callback. Called once from the ctor, EARLY — before
/// Python imports `ssl` — so the interposition is in place by the first TLS call.
pub fn install() {
    unsafe { _dyld_register_func_for_add_image(on_add_image) };
}
