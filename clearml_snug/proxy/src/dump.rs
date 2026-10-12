//! Developer-only exchange dump.
//!
//! With `CLEARML_SNUG_DUMP_DIR` set, the proxy writes the decrypted request and
//! response bodies of every completed exchange to that directory, for the hosts
//! listed in `CLEARML_SNUG_DUMP_HOSTS` (comma separated, default `claude.ai`).
//! It exists to study a wire format that carries no usage (the claude.ai chat
//! panel) when designing an estimate. The bodies hold the user's conversation
//! content, so it is off unless explicitly enabled and is never meant for a
//! production session.
//!
//! One exchange yields three files sharing a stem: `<stem>.req` (decompressed
//! request body), `<stem>.resp` (decompressed response body) and `<stem>.json`
//! (host, method, path and sizes).

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

struct Dump {
    dir: PathBuf,
    hosts: Vec<String>,
    seq: AtomicU64,
}

static DUMP: OnceLock<Dump> = OnceLock::new();

/// Enable the dump from the environment. A missing/empty dir leaves it off.
pub fn init_from_env() {
    let Some(dir) = std::env::var("CLEARML_SNUG_DUMP_DIR").ok().filter(|s| !s.is_empty()) else {
        return;
    };
    let hosts: Vec<String> = std::env::var("CLEARML_SNUG_DUMP_HOSTS")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "claude.ai".to_string())
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty())
        .collect();
    if let Err(e) = fs::create_dir_all(&dir) {
        crate::log(&format!("WARN: cannot create dump dir {}: {}", dir, e));
        return;
    }
    crate::log(&format!("exchange dump ENABLED -> {} for hosts {:?}", dir, hosts));
    let _ = DUMP.set(Dump { dir: PathBuf::from(dir), hosts, seq: AtomicU64::new(0) });
}

/// Whether exchanges to `host` are dumped.
pub fn wants(host: &str) -> bool {
    match DUMP.get() {
        Some(d) => {
            let h = host.to_ascii_lowercase();
            d.hosts.iter().any(|w| h == *w || h.ends_with(&format!(".{}", w)))
        }
        None => false,
    }
}

/// Write one completed exchange. Best-effort: a write error is logged and dropped.
pub fn write(
    conn_id: u64,
    host: &str,
    method: Option<&str>,
    path: Option<&str>,
    req_plain: &[u8],
    resp_plain: &[u8],
) {
    let Some(d) = DUMP.get() else { return };
    let n = d.seq.fetch_add(1, Ordering::Relaxed);
    let ts = crate::now_ms();
    let stem = d.dir.join(format!("{}_{:04}_c{}", ts, n, conn_id));
    let meta = format!(
        "{{\"ts\":{},\"conn\":{},\"host\":{:?},\"method\":{:?},\"path\":{:?},\"req_bytes\":{},\"resp_bytes\":{}}}\n",
        ts,
        conn_id,
        host,
        method.unwrap_or(""),
        path.unwrap_or(""),
        req_plain.len(),
        resp_plain.len()
    );
    for (ext, data) in [("req", req_plain), ("resp", resp_plain), ("json", meta.as_bytes())] {
        if let Err(e) = fs::write(stem.with_extension(ext), data) {
            crate::log(&format!("WARN: dump write failed for {:?}: {}", stem, e));
            return;
        }
    }
}

/// Append decoded bytes of a stream that never completes (an SSE channel) to
/// `<dir>/stream_<name>.sse`. Best-effort, like [`write`].
pub fn append(name: &str, data: &[u8]) {
    use std::io::Write;
    let Some(d) = DUMP.get() else { return };
    let path = d.dir.join(format!("stream_{}.sse", name));
    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(data))
    {
        crate::log(&format!("WARN: dump append failed for {:?}: {}", path, e));
    }
}
