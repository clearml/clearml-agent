//! Reads the agent's handoff descriptor from the inherited fd named by the
//! `CLEARML_SNUG_CRED_FD` env var (a `memfd` on Linux, an unlinked temp file on
//! macOS).
//!
//! Returns the parsed `Descriptor` for the ctor to hand to the reporter, or
//! `None` when there's no descriptor — in which case no reporter is started and
//! the shim falls back to stderr (e.g. an operator running `curl` under the
//! preload var directly, with no agent to hand off creds).
//!
//! ## Linux vs macOS: consume-once vs re-exec-survivable
//!
//!   * **Linux**: the task process loads the shim exactly once, so we
//!     consume-once — read + close the fd and remove the env var so child
//!     processes don't inherit (and mis-read) a now-closed/reused fd number.
//!   * **macOS**: framework Python builds (Homebrew / python.org — the common
//!     case) re-exec themselves once at startup (the `bin/pythonX` stub
//!     `execv`s the real `Python.app` binary), which keeps the SAME pid but
//!     reloads our dylib, running this ctor TWICE. If we consumed the fd on the
//!     first load, the FINAL image (where the task's code + LLM calls run) would
//!     find no descriptor and fall back to `reporter=stderr` — silently losing
//!     all reporting. So on macOS we DON'T consume: we read the inherited fd
//!     without closing it (and keep the env var), so each same-pid re-exec
//!     re-reads it and starts the final, surviving reporter.
//!
//!     To stop a SPAWNED DESCENDANT (a child/worker the task forks+execs, e.g. a
//!     `multiprocessing` "spawn" worker — macOS's default) from each starting
//!     its own reporter for the same task, we stamp `CLEARML_SNUG_CRED_OWNER`
//!     with our pid on the first read. A later load whose pid differs from the
//!     stamp is a descendant (a same-pid re-exec matches) and skips. (Edge: in
//!     the rare case a non-task process loaded the shim with the cred fd set
//!     BEFORE the task, it would claim ownership and the task would skip — but
//!     the venv launch flow spawns no such process between cred-fd creation and
//!     task launch, and the failure mode is just "no reporting", no worse than
//!     having no descriptor at all.)

use std::os::unix::io::RawFd;

use clearml_snug_reporter::Descriptor;

const CRED_FD_ENV: &str = "CLEARML_SNUG_CRED_FD";

/// Base64-encoded descriptor JSON, an alternative to the inheritable fd. An env
/// var survives child spawning that scrubs the inherited cred fd — Chromium's
/// process launcher and `bwrap` both drop non-allowlisted fds but keep the env —
/// so it is the delivery channel for sandboxed hosts (fd-scrubbing sandboxed
/// children of Electron/Chromium/bwrap hosts). Trade-off vs the fd: the secret is
/// then visible in the process environment block, so the agent sets it only for
/// the sandboxed-app launch path, not for ordinary tasks (where the fd works and
/// keeps the secret out of the environment).
const CRED_ENV: &str = "CLEARML_SNUG_CRED";

/// Pid stamp marking the process lineage that owns the cred fd. Same-pid
/// re-execs match it (re-read); spawned descendants don't (skip). macOS only.
#[cfg(target_os = "macos")]
const CRED_OWNER_ENV: &str = "CLEARML_SNUG_CRED_OWNER";

/// True iff `fd` refers to a regular file. The credential handoff fd is ALWAYS a
/// regular file (an anonymous `memfd` on Linux, an unlinked temp file on macOS).
///
/// Guarding on this before reading is critical for robustness: if the fd number
/// does not actually carry our descriptor in this process (e.g. it wasn't passed
/// down and the number got reused for a pipe/socket — observed on some macOS
/// launch paths), a blind `read_to_string` would BLOCK forever waiting for an
/// EOF that never arrives, hanging the host at startup until it's SIGKILLed (a
/// task dying with exit 247 and no `[snug] init` line). A non-regular fd -> skip
/// -> stderr fallback, and we must NOT close it (it isn't ours).
fn is_regular_file(fd: RawFd) -> bool {
    // SAFETY: fstat only writes into our local stat buf and tolerates any fd
    // (returns an error for a bad one).
    unsafe {
        let mut st: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut st) != 0 {
            return false;
        }
        (st.st_mode & libc::S_IFMT) == libc::S_IFREG
    }
}

/// Env-delivered descriptor fallback (see `CRED_ENV`). Base64-decode
/// `CLEARML_SNUG_CRED` and parse the descriptor JSON. `None` when the var is
/// unset or malformed.
fn descriptor_from_cred_env() -> Option<Descriptor> {
    use base64::Engine;
    let raw = std::env::var(CRED_ENV).ok()?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(raw.trim())
        .ok()?;
    let json = String::from_utf8(bytes).ok()?;
    match Descriptor::from_json_str(&json) {
        Ok(d) => Some(d),
        Err(e) => {
            snug_err!("[snug] {} decode failed: {}", CRED_ENV, e);
            None
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn read_and_close() -> Option<Descriptor> {
    // Prefer the inheritable fd (keeps the secret out of the environment); fall
    // back to the env-delivered descriptor for sandboxed processes that lost the
    // fd (fd-scrubbing sandboxed children of Electron/Chromium/bwrap hosts).
    read_from_fd().or_else(descriptor_from_cred_env)
}

#[cfg(not(target_os = "macos"))]
fn read_from_fd() -> Option<Descriptor> {
    let raw = std::env::var(CRED_FD_ENV).ok()?;
    // Don't let children inherit the fd number (the fd is single-use and we
    // close it below). No same-pid re-exec on Linux, so consume-once is safe.
    std::env::remove_var(CRED_FD_ENV);

    let fd: RawFd = match raw.trim().parse() {
        Ok(n) => n,
        Err(_) => {
            snug_err!("[snug] invalid {}={:?}", CRED_FD_ENV, raw);
            return None;
        }
    };

    // The handoff fd MUST be a regular file (see is_regular_file).
    if !is_regular_file(fd) {
        // Expected in scrubbed sandbox children (the fd number wasn't passed
        // down / got reused) — routine, so debug-gated rather than an error.
        snug_log!(
            "[snug] {}={} is not a regular file; skipping (reporter=stderr)",
            CRED_FD_ENV, fd
        );
        return None;
    }

    // `Descriptor::from_fd` takes ownership of the fd and closes it on return.
    match Descriptor::from_fd(fd) {
        Ok(d) => Some(d),
        Err(e) => {
            snug_err!("[snug] descriptor read failed: {}", e);
            None
        }
    }
}

#[cfg(target_os = "macos")]
pub fn read_and_close() -> Option<Descriptor> {
    read_from_fd().or_else(descriptor_from_cred_env)
}

#[cfg(target_os = "macos")]
fn read_from_fd() -> Option<Descriptor> {
    let raw = std::env::var(CRED_FD_ENV).ok()?;

    // SAFETY: getpid is async-signal-safe and infallible.
    let me = unsafe { libc::getpid() };

    // Descendant guard: if a CRED_OWNER stamp exists and it isn't us, this load
    // is a child/worker the task spawned (it inherited the preload var + the fd
    // but has a different pid). Descendants must NOT start their own reporter —
    // skip silently. A same-pid re-exec (the framework stub relaunch) matches
    // the stamp and falls through to re-read.
    if let Ok(owner) = std::env::var(CRED_OWNER_ENV) {
        if owner.trim().parse::<i32>().ok() != Some(me) {
            return None;
        }
    }

    let fd: RawFd = match raw.trim().parse() {
        Ok(n) => n,
        Err(_) => {
            snug_err!("[snug] invalid {}={:?}", CRED_FD_ENV, raw);
            return None;
        }
    };

    // The handoff fd MUST be a regular file (see is_regular_file).
    if !is_regular_file(fd) {
        // Expected in scrubbed sandbox children (the fd number wasn't passed
        // down / got reused) — routine, so debug-gated rather than an error.
        snug_log!(
            "[snug] {}={} is not a regular file; skipping (reporter=stderr)",
            CRED_FD_ENV, fd
        );
        return None;
    }

    // Read WITHOUT consuming the inherited fd: a later same-pid re-exec (the
    // framework Python stub relaunching itself) must re-read this fd to start
    // the FINAL, surviving reporter. We dup it, rewind to the start, and let
    // `Descriptor::from_fd` consume only the dup; the original fd stays open and
    // the env var stays set for the next load.
    let dup = unsafe { libc::dup(fd) };
    if dup < 0 {
        snug_err!("[snug] could not dup {}={}; skipping (reporter=stderr)", CRED_FD_ENV, fd);
        return None;
    }
    // The dup shares the open file description's offset, so rewind to 0 (a prior
    // load read to EOF).
    unsafe { libc::lseek(dup, 0, libc::SEEK_SET) };

    match Descriptor::from_fd(dup) {
        Ok(d) => {
            // Claim ownership so spawned descendants skip; keep CRED_FD set so a
            // same-pid re-exec re-reads. set_var is safe here: the ctor runs
            // single-threaded, before the reporter thread is spawned.
            std::env::set_var(CRED_OWNER_ENV, me.to_string());
            Some(d)
        }
        Err(e) => {
            snug_err!("[snug] descriptor read failed: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn cred_env_roundtrip_and_absent() {
        // No env var -> None.
        std::env::remove_var(CRED_ENV);
        assert!(descriptor_from_cred_env().is_none());

        // Base64 of a minimal descriptor JSON -> parsed Descriptor.
        let json = r#"{"api_server":"https://api.x","task_id":"task-123","report_usage_events":true}"#;
        std::env::set_var(CRED_ENV, base64::engine::general_purpose::STANDARD.encode(json));
        let d = descriptor_from_cred_env().expect("descriptor from CLEARML_SNUG_CRED");
        assert_eq!(d.task_id, "task-123");
        assert_eq!(d.api_server, "https://api.x");
        assert!(d.report_usage_events);
        std::env::remove_var(CRED_ENV);
    }
}
