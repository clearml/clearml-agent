//! Local TLS-terminating forward proxy for SNUG usage metering.
//!
//! Purpose: meter LLM HTTPS traffic from clients the LD_PRELOAD shim can't hook
//! — clients that statically link BoringSSL (e.g. Electron renderers and
//! bun-compiled CLIs, with Claude Desktop as the initial target), so there is no
//! `SSL_write`/`SSL_read` GOT slot to interpose. Such a client is instead pointed
//! at this proxy with `HTTPS_PROXY` and told to trust our CA with
//! `NODE_EXTRA_CA_CERTS`/`SSL_CERT_FILE`; the proxy decrypts the whitelisted
//! provider traffic and runs it through the EXACT same usage parser the shim uses
//! (`body_scan`), producing the same usage/model events.
//!
//! Design: for a known LLM provider host the proxy terminates the client TLS
//! (presenting a leaf minted under a locally generated CA), opens an upstream
//! TLS leg to the real host with the SAME negotiated ALPN, and relays the
//! decrypted bytes verbatim in both directions. Because bytes are relayed
//! verbatim (never re-framed), HTTP/2's end-to-end HPACK dynamic table and flow
//! control stay consistent — the proxy is just a TLS re-encryption hop that also
//! tees a copy of the plaintext into the usage scanner. Non-provider hosts are
//! blind-tunneled (raw TCP relay), so nothing but LLM traffic is decrypted.
//!
//! The parsing modules (`body_scan`, `parser`, `h2`, `decompress`) are included
//! verbatim from the shim source so the parsed usage/model is identical; the
//! reporting path (`start_reporter` + `Descriptor` + `Event`) is the shim's own
//! reporter crate, reused as-is.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, ClientConnection, RootCertStore, ServerConfig, ServerConnection, StreamOwned};
use serde::Serialize;

use clearml_snug_messages::Event;
use clearml_snug_reporter::{start_reporter, Descriptor, ReporterHandle};

// Parsing modules reused verbatim from the shim. `body_scan` refers to
// `crate::parser`, satisfied by declaring `parser` at the crate root here.
#[path = "../../shim/src/parser.rs"]
mod parser;
#[path = "../../shim/src/body_scan.rs"]
mod body_scan;
#[path = "../../shim/src/h2.rs"]
mod h2;
#[path = "../../shim/src/decompress.rs"]
mod decompress;
#[path = "../../shim/src/tokens.rs"]
mod tokens;
// Whitelist matcher reused verbatim from the shim: the proxy consults only
// `load_from_env()`, `matches()`, and `default_action` to gate reporting and
// pick a per-host tokenizer (the path needed by `matches()` is now recovered via
// HPACK decoding). The module's runtime hot-reload machinery is unused here.
#[allow(dead_code)]
#[path = "../../shim/src/whitelist.rs"]
mod whitelist;

// whitelist.rs's (unused) hot-reload machinery references `crate::state` and
// `crate::meter` — the shim's poll/reporter internals. Provide no-op shims so the
// module compiles verbatim without the proxy dragging those in; the proxy never
// calls the machinery that would reach them.
#[allow(dead_code)]
mod state {
    pub fn rearm_whitelist_suppressions() {}
}
#[allow(dead_code)]
mod meter {
    pub fn emit(_event: clearml_snug_messages::Event) {}
}

mod ca;

use body_scan::{
    conversation_fingerprint, looks_like_sse, model_from_request, provider_for_host,
    tool_error_names_in_request, Provider, RespParse, REQ_BODY_CAP,
};

/// Provider resolution for the proxy, config-driven: a whitelist rule may declare
/// a `provider` for a host whose wire format matches a known provider but whose
/// hostname isn't in the shim-shared base `provider_for_host` map. The motivating
/// case is a consumer chat API such as `claude.ai`, which speaks the Anthropic
/// wire (a model in `message_start` but no `usage` object) — a rule
/// `{host:"claude.ai", provider:"anthropic", …}` flows it through the usage/model
/// parser and the byte-ratio estimate fallback (`emit`). With no rule (or no
/// `provider` field) it falls back to the base map, so plain provider detection
/// and shim behavior are unchanged.
fn proxy_provider_for_host(host: &str, whitelist: &whitelist::Whitelist) -> Option<Provider> {
    if let Some(p) = provider_from_name(whitelist.provider_hint(host)) {
        return Some(p);
    }
    provider_for_host(host)
}

/// Map a whitelist rule's `provider` hint to a `Provider`. Empty/unknown -> None
/// (the caller then falls back to the base host map).
fn provider_from_name(name: &str) -> Option<Provider> {
    match name.trim().to_ascii_lowercase().as_str() {
        "anthropic" => Some(Provider::Anthropic),
        "openai" => Some(Provider::OpenAi),
        "gemini" => Some(Provider::Gemini),
        _ => None,
    }
}

/// Tokenizer family for a provider's byte-ratio estimate, used only when the
/// response carried no measured `usage` (the claude.ai chat panel). The served
/// model refines the Claude ratio by generation inside the estimator.
fn tokenizer_for(provider: Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "claude",
        Provider::OpenAi => "cl100k",
        Provider::Gemini => "approx",
    }
}

/// Plaintext length of a (possibly gzip/zstd-compressed) request body, for the
/// input-token estimate. claude.ai's consumer API compresses its request body,
/// and over h2 the `Content-Encoding` is HPACK-hidden, so the encoding is
/// detected from the body's own magic bytes and inflated before measuring; an
/// identity body is measured as-is. Mirrors the request-body handling in the
/// shim's completion path (state.rs).
fn plaintext_len(body: &[u8]) -> u64 {
    let enc = decompress::detect(body);
    if enc.is_compressed() {
        if let Some(plain) = decompress::decompress(enc, body, REQ_DECOMP_CAP) {
            return plain.len() as u64;
        }
    }
    body.len() as u64
}

/// Cap on request-body inflation for the input-token estimate (guards against a
/// decompression bomb). Mirrors the shim's `REQ_DECOMP_CAP`.
const REQ_DECOMP_CAP: usize = 8 * 1024 * 1024;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn log(msg: &str) {
    // stderr, plus stdout for the standalone-run case.
    eprintln!("[snug-proxy] {}", msg);
}

/// Gate for the verbose per-connection / per-request diagnostics. Set once at
/// startup from `CLEARML_SNUG_DEBUG_LOG` (agent config `agent.snug.debug_log`,
/// the same flag the shim honors). Default off keeps the proxy log quiet under
/// load; startup, reporter, and genuine error lines still print unconditionally,
/// while expected client-disconnect churn (`is_benign_disconnect`) is gated
/// behind this flag.
static DEBUG_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn dlog(msg: &str) {
    if DEBUG_LOG.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("[snug-proxy] {}", msg);
    }
}

/// Disconnects that are normal for a browser's connection pool: Chromium opens
/// speculative preconnect sockets and tears down idle keep-alive tunnels,
/// dropping the connection before or during the TLS handshake. These surface as
/// EOF / connection-reset / broken-pipe and are expected churn, not proxy
/// faults, so the accept loop logs them at debug level only. Genuine failures
/// (upstream connect refused/timed out, cert/config errors — kinds like
/// `ConnectionRefused`, `TimedOut`, `Other`) do not match and stay loud.
fn is_benign_disconnect(e: &std::io::Error) -> bool {
    use std::io::ErrorKind::{BrokenPipe, ConnectionReset, UnexpectedEof};
    matches!(e.kind(), UnexpectedEof | ConnectionReset | BrokenPipe)
}

/// Shared process state: the CA, prebuilt upstream configs, the decrypt policy, the
/// capture-all traffic log, the optional reporter channel, and a connection
/// counter.
struct Shared {
    ca: ca::Ca,
    upstream_h2: Arc<ClientConfig>,
    upstream_h1: Arc<ClientConfig>,
    policy: DecryptPolicy,
    traffic: TrafficLog,
    tx: Option<SyncSender<Event>>,
    conn_seq: std::sync::atomic::AtomicU64,
    /// Operator-controlled report gate + per-host tokenizer, loaded once at
    /// startup from `CLEARML_SNUG_WHITELIST`. An empty/unset whitelist is
    /// `default_action="meter"` with no rules, preserving meter-all.
    whitelist: whitelist::Whitelist,
}

/// Which CONNECT targets the proxy decrypts versus blind-tunnels.
///
/// Default (env unset): decrypt only known LLM providers (`provider_for_host`),
/// blind-tunnel everything else — the providers-only metering behavior. With
/// `CLEARML_SNUG_PROXY_DECRYPT_ALL=1`: decrypt EVERY host so all of the client's
/// traffic is captured.
///
/// There is deliberately no host-level opt-out. Every path into this policy
/// either widens coverage or leaves it unchanged, so no configuration can carve a
/// metered provider back out of it. An app that genuinely cannot be decrypted on
/// some host (a pinned updater, say) is a property of that app, so the exemption
/// belongs on its `AppProfile` rather than in operator config.
struct DecryptPolicy {
    decrypt_all: bool,
}

impl DecryptPolicy {
    fn from_env() -> Self {
        Self { decrypt_all: env_flag("CLEARML_SNUG_PROXY_DECRYPT_ALL") }
    }

    /// Whether to decrypt `host`: decrypt-all decrypts everything, and the
    /// default decrypts only known providers (including any host a whitelist rule
    /// maps to a provider, via `proxy_provider_for_host`, so a config-declared
    /// consumer host such as `claude.ai` is metered without requiring
    /// decrypt-all).
    fn should_decrypt(&self, host: &str, whitelist: &whitelist::Whitelist) -> bool {
        self.decrypt_all || proxy_provider_for_host(host, whitelist).is_some()
    }
}

/// Truthy env flag: value in {1, true, yes, on} (case-insensitive).
fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// One completed request/response as recorded in the capture-all traffic log —
/// written for EVERY decrypted host, independent of the provider/usage gating the
/// reporter sinks apply. `method`/`path` come from the request line on HTTP/1.1
/// and from decoding the request's HPACK header block on HTTP/2; either is `null`
/// only when it couldn't be recovered (an h2 connection whose HPACK decoding gave
/// up). `status` is `null` when it couldn't be read (the h2 case — `:status` lives
/// in HPACK, which the response side doesn't decode). One JSON object per line.
#[derive(Serialize)]
struct CaptureRecord<'a> {
    ts: u64,
    host: &'a str,
    method: Option<&'a str>,
    path: Option<&'a str>,
    status: Option<u16>,
    tx: u64,
    rx: u64,
    ms: u64,
}

/// Append-only writer for the capture-all traffic log. The path is
/// `$CLEARML_SNUG_TRAFFIC_LOG`, falling back to `$CLEARML_SNUG_LOG_FILE`; with
/// neither set, capture logging is disabled (the per-request stderr line still
/// prints). One JSON record per completed request is appended and flushed.
struct TrafficLog {
    file: Option<Mutex<File>>,
}

impl TrafficLog {
    fn from_env() -> Self {
        let path = std::env::var("CLEARML_SNUG_TRAFFIC_LOG")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("CLEARML_SNUG_LOG_FILE").ok().filter(|s| !s.is_empty()));
        let file = match path {
            Some(p) => match OpenOptions::new().create(true).append(true).open(&p) {
                Ok(f) => {
                    log(&format!("capture-all traffic log at {}", p));
                    Some(Mutex::new(f))
                }
                Err(e) => {
                    log(&format!("WARN: cannot open traffic log {}: {}", p, e));
                    None
                }
            },
            None => None,
        };
        Self { file }
    }

    /// Append one capture record as a JSON line. Best-effort: a write error is
    /// dropped so capture logging never stalls a relay.
    fn record(&self, rec: &CaptureRecord) {
        let Some(file) = &self.file else { return };
        let Ok(line) = serde_json::to_string(rec) else { return };
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "{}", line);
            let _ = f.flush();
        }
    }
}

/// Derive the CA key path from the cert path: replace a trailing `.pem` with
/// `.key`, else append `.key`.
fn default_key_path(ca_path: &str) -> String {
    match ca_path.strip_suffix(".pem") {
        Some(stem) => format!("{}.key", stem),
        None => format!("{}.key", ca_path),
    }
}

/// Default SPKI-pin file path: `snug_proxy_ca.spki` in the CA cert's directory,
/// the location the launcher reads when `CLEARML_SNUG_PROXY_CA_SPKI_FILE` is
/// unset.
fn default_spki_path(ca_path: &str) -> String {
    let name = "snug_proxy_ca.spki";
    match Path::new(ca_path).parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(name).to_string_lossy().into_owned(),
        _ => name.to_string(),
    }
}

fn build_upstream_config(roots: Arc<RootCertStore>, alpn: &[&[u8]]) -> Arc<ClientConfig> {
    let mut cfg = ClientConfig::builder()
        .with_root_certificates((*roots).clone())
        .with_no_client_auth();
    cfg.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();
    Arc::new(cfg)
}

fn main() {
    // rustls 0.23 needs an installed crypto provider before any builder runs.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Per-connection/per-request diagnostics are opt-in; the agent exports
    // CLEARML_SNUG_DEBUG_LOG from agent.snug.debug_log into the proxy's env.
    DEBUG_LOG.store(
        env_flag("CLEARML_SNUG_DEBUG_LOG"),
        std::sync::atomic::Ordering::Relaxed,
    );

    let port: u16 = std::env::var("CLEARML_SNUG_PROXY_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8888);
    let ca_path =
        std::env::var("CLEARML_SNUG_PROXY_CA").unwrap_or_else(|_| "/tmp/snug_proxy_ca.pem".to_string());
    // Key persisted next to the cert so the CA survives proxy restarts and
    // already-running clients keep trusting it (see ca.rs). Overridable.
    let key_path = std::env::var("CLEARML_SNUG_PROXY_CA_KEY")
        .unwrap_or_else(|_| default_key_path(&ca_path));

    let (ca, freshly_generated) = ca::Ca::load_or_generate(&ca_path, &key_path);
    // Ensure the advertised cert file matches the (possibly reloaded) CA even if
    // it was deleted out from under us since the key was written.
    if let Err(e) = ca.write_pem(&ca_path) {
        log(&format!("FATAL: cannot write CA to {}: {}", ca_path, e));
        std::process::exit(1);
    }
    log(&format!(
        "CA {} at {} (key {}) — trust it via NODE_EXTRA_CA_CERTS / SSL_CERT_FILE",
        if freshly_generated { "generated" } else { "reused" },
        ca_path,
        key_path
    ));

    // Compute the CA's SPKI pin (base64(SHA-256(SubjectPublicKeyInfo))) and hand
    // it off two ways: a file the launcher reads
    // (`$CLEARML_SNUG_PROXY_CA_SPKI_FILE`, default `<ca_dir>/snug_proxy_ca.spki`)
    // and a `SPKI=<value>` line on stdout. A client passes it to
    // `--ignore-certificate-errors-spki-list` to trust the proxy's leaves by
    // pinning the CA key instead of installing the cert.
    let spki = ca.spki_sha256_b64();
    let spki_path = std::env::var("CLEARML_SNUG_PROXY_CA_SPKI_FILE")
        .unwrap_or_else(|_| default_spki_path(&ca_path));
    match std::fs::write(&spki_path, format!("{}\n", spki)) {
        Ok(()) => log(&format!("CA SPKI {} written to {}", spki, spki_path)),
        Err(e) => log(&format!("WARN: cannot write SPKI file {}: {}", spki_path, e)),
    }
    // Machine-readable handoff line for a launcher capturing stdout.
    println!("SPKI={}", spki);

    // Upstream root store (webpki-roots), shared across connections. An optional
    // extra CA PEM (self-hosted gateways / local testing) is added on top.
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    if let Ok(path) = std::env::var("CLEARML_SNUG_PROXY_UPSTREAM_CA") {
        match pem_certs(&path) {
            Ok(certs) => {
                let mut n = 0;
                for c in certs {
                    if roots.add(c).is_ok() {
                        n += 1;
                    }
                }
                log(&format!("added {} upstream CA cert(s) from {}", n, path));
            }
            Err(e) => log(&format!("WARN: upstream CA {} not loaded: {}", path, e)),
        }
    }
    let roots = Arc::new(roots);
    let upstream_h2 = build_upstream_config(roots.clone(), &[b"h2", b"http/1.1"]);
    let upstream_h1 = build_upstream_config(roots.clone(), &[b"http/1.1"]);

    // Optional reporter: only if the agent handed off a descriptor (base64 in
    // CLEARML_SNUG_CRED). Without it the proxy just logs parsed usage lines,
    // which is enough for standalone validation.
    let (tx, handle) = start_optional_reporter();

    let policy = DecryptPolicy::from_env();
    if policy.decrypt_all {
        log("decrypt-all ON: decrypt every host");
    } else {
        log("decrypt-all OFF: decrypt known providers only (blind-tunnel the rest)");
    }
    let traffic = TrafficLog::from_env();

    // Operator-controlled report gate + per-host tokenizer. Empty/unset ->
    // default_action="meter", no rules (meter-all preserved).
    let whitelist = whitelist::Whitelist::load_from_env();
    log(&format!(
        "whitelist: {} rules, default_action={}",
        whitelist.rules.len(),
        whitelist.default_action
    ));

    let shared = Arc::new(Shared {
        ca,
        upstream_h2,
        upstream_h1,
        policy,
        traffic,
        tx,
        conn_seq: std::sync::atomic::AtomicU64::new(1),
        whitelist,
    });

    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            log(&format!("FATAL: cannot bind 127.0.0.1:{}: {}", port, e));
            std::process::exit(1);
        }
    };
    log(&format!("listening on 127.0.0.1:{}", port));

    // Keep the reporter handle alive for the process lifetime so its background
    // thread isn't dropped. (This is a long-lived dev tool; exit-time drain is
    // not wired — the shim owns that for the shipped path.)
    let _handle = handle;

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_conn(s, shared) {
                        if is_benign_disconnect(&e) {
                            dlog(&format!("conn closed early: {}", e));
                        } else {
                            log(&format!("conn error: {}", e));
                        }
                    }
                });
            }
            Err(e) => log(&format!("accept error: {}", e)),
        }
    }
}

/// Read the CONNECT request head (bytes up to and including CRLFCRLF) one byte
/// at a time so we never consume the client's subsequent TLS ClientHello (the
/// client waits for our 200 before sending it).
fn read_connect_head(s: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(256);
    let mut one = [0u8; 1];
    loop {
        let n = s.read(&mut one)?;
        if n == 0 {
            break;
        }
        buf.push(one[0]);
        if buf.ends_with(b"\r\n\r\n") {
            break;
        }
        if buf.len() > 16 * 1024 {
            break;
        }
    }
    Ok(buf)
}

fn handle_conn(mut client: TcpStream, shared: Arc<Shared>) -> std::io::Result<()> {
    client.set_nodelay(true).ok();
    let head = read_connect_head(&mut client)?;
    let line = String::from_utf8_lossy(&head);
    let first = line.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    if method != "CONNECT" {
        let _ = client.write_all(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n");
        return Ok(());
    }
    let host = target.split(':').next().unwrap_or("").to_string();
    let port: u16 = target.split(':').nth(1).and_then(|p| p.parse().ok()).unwrap_or(443);

    client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;

    // Provider detection stays independent of the decrypt decision: it keys the
    // usage/scalars sink. Under decrypt-all a non-provider host is still decrypted
    // (provider `None`) so its traffic is captured, but it feeds only the
    // capture-all log, not the usage sink. A whitelist rule's `provider` field
    // resolves a config-declared host (e.g. a consumer chat wire) to a provider
    // here so it flows through the usage sink too.
    let provider = proxy_provider_for_host(&host, &shared.whitelist);
    if shared.policy.should_decrypt(&host, &shared.whitelist) {
        dlog(&format!("decrypt {}:{} (provider {:?})", host, port, provider));
        intercept(client, shared, host, port, provider)
    } else {
        // Not decrypted: blind-tunnel, decrypt nothing.
        blind_tunnel(client, &host, port)
    }
}

/// Raw TCP relay for non-provider hosts. No TLS termination, no decryption.
fn blind_tunnel(client: TcpStream, host: &str, port: u16) -> std::io::Result<()> {
    let upstream = TcpStream::connect((host, port))?;
    upstream.set_nodelay(true).ok();
    let c2 = client.try_clone()?;
    let u2 = upstream.try_clone()?;
    let t1 = std::thread::spawn(move || copy_raw(c2, upstream));
    let t2 = std::thread::spawn(move || copy_raw(u2, client));
    let _ = t1.join();
    let _ = t2.join();
    Ok(())
}

fn copy_raw(mut from: TcpStream, mut to: TcpStream) {
    let mut buf = [0u8; 65536];
    loop {
        match from.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if to.write_all(&buf[..n]).is_err() {
                    break;
                }
            }
        }
    }
    let _ = to.shutdown(std::net::Shutdown::Write);
}

/// Intercept a provider connection: terminate the client TLS with a minted leaf,
/// open the upstream TLS with matching ALPN, relay decrypted bytes both ways and
/// tee copies through the usage parser.
fn intercept(
    client: TcpStream,
    shared: Arc<Shared>,
    host: String,
    port: u16,
    provider: Option<Provider>,
) -> std::io::Result<()> {
    // Terminate the client's TLS.
    let leaf = shared.ca.leaf_for(&host);
    let mut server_cfg = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(leaf.cert_chain.clone(), leaf.key.clone_key())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("server cfg: {}", e)))?;
    server_cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let server_cfg = Arc::new(server_cfg);

    let mut client_tcp = client;
    let mut sconn = ServerConnection::new(server_cfg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("sconn: {}", e)))?;
    while sconn.is_handshaking() {
        sconn.complete_io(&mut client_tcp)?;
    }
    let neg_alpn = sconn.alpn_protocol().map(|b| b.to_vec());
    let is_h2 = neg_alpn.as_deref() == Some(b"h2");

    // Open the upstream leg with the SAME ALPN so the verbatim byte relay stays
    // protocol-consistent.
    let up_cfg = if is_h2 { shared.upstream_h2.clone() } else { shared.upstream_h1.clone() };
    let up_tcp = TcpStream::connect((host.as_str(), port))?;
    up_tcp.set_nodelay(true).ok();
    let sni = ServerName::try_from(host.clone())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "bad sni"))?;
    let mut cconn = ClientConnection::new(up_cfg, sni)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("cconn: {}", e)))?;
    let mut up_tcp2 = up_tcp;
    while cconn.is_handshaking() {
        cconn.complete_io(&mut up_tcp2)?;
    }

    let conn_id = shared
        .conn_seq
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    dlog(&format!(
        "conn {} TLS up (alpn={}) provider={:?} host={}",
        conn_id,
        if is_h2 { "h2" } else { "http/1.1" },
        provider,
        host
    ));

    if is_h2 {
        relay_h2(conn_id, provider, host, sconn, client_tcp, cconn, up_tcp2, shared)
    } else {
        let server_tls = StreamOwned::new(sconn, client_tcp);
        let upstream_tls = StreamOwned::new(cconn, up_tcp2);
        relay_h1(conn_id, provider, host, server_tls, upstream_tls, shared)
    }
}

type ServerTls = StreamOwned<ServerConnection, TcpStream>;
type UpstreamTls = StreamOwned<ClientConnection, TcpStream>;

/// Emit a completed request/response.
///
/// Two independent outputs: (1) the capture-all traffic log gets one record for
/// EVERY decrypted host (provider or not), and (2) for a known provider only, the
/// reporter usage/scalars sink gets the paired `RequestStarted`/`RequestCompleted`
/// (unchanged from the providers-only path). A non-provider host (decrypt-all)
/// therefore contributes to capture but never to the usage sink.
#[allow(clippy::too_many_arguments)]
fn emit(
    shared: &Shared,
    conn_id: u64,
    provider: Option<Provider>,
    host: &str,
    started_ms: u64,
    status: Option<u16>,
    bytes_tx: u64,
    bytes_rx: u64,
    // Decompressed response body bytes the scanner saw, for the output-token
    // estimate when the provider reported no measured `usage` (the claude.ai
    // chat panel). Ignored when usage was measured.
    resp_text_bytes: u64,
    fin: &body_scan::Finalized,
    req_model: Option<String>,
    req_body: &[u8],
    method: Option<&str>,
    path: Option<&str>,
) {
    let latency_ms = now_ms().saturating_sub(started_ms);

    // Capture-all: one record per completed request for every decrypted host.
    shared.traffic.record(&CaptureRecord {
        ts: now_ms(),
        host,
        method,
        path,
        status,
        tx: bytes_tx,
        rx: bytes_rx,
        ms: latency_ms,
    });

    // The rest (usage log line + reporter sinks) is provider-only.
    let Some(provider) = provider else { return };

    // Operator-controlled report gate: a rule match (needs the HPACK-decoded
    // path) reports and pins the rule's tokenizer; no match falls back to the
    // configured `default_action`. With an empty/unset whitelist that is "meter",
    // so `whitelisted` is always true and meter-all is preserved; only an operator
    // setting default_action="ignore" turns unmatched hosts off.
    let rule = shared.whitelist.matches(host, path.unwrap_or("/"));
    let whitelisted =
        rule.is_some() || shared.whitelist.default_action.eq_ignore_ascii_case("meter");

    let tokens_measured = fin.tokens_in.is_some() || fin.tokens_out.is_some();
    let model = fin.model.clone().or(req_model);
    // A host's UNMEASURED requests are byte-estimated only when its whitelist rule
    // opts in (`estimate_unmeasured`) AND the decoded request line matches the
    // rule's `completion_path` as a `POST` (`WhitelistRule::is_completion`). This
    // is the config-driven form of the consumer-wire gate: a consumer chat host
    // (e.g. claude.ai) carries a model but no `usage`, so its completions must be
    // estimated — but only genuine generations. It matters because such a wire's
    // `GET .../chat_conversations/{id}?...` history-loads return a stored
    // conversation with a model but no usage (indistinguishable from a completion
    // by body alone) and fire on every conversation switch; estimating them would
    // report the whole loaded history as a phantom completion. Real-API hosts
    // leave `estimate_unmeasured` false so their `count_tokens`/telemetry calls
    // are never estimated. A request the proxy couldn't HPACK-decode is NOT a
    // completion (fail closed). For a real completion, input is estimated from the
    // request body (decompressed if the client compressed it); the consumer
    // `/completion` response is only a small ack, so output is usually ~0 (the
    // reply text isn't on the wire — the output-bearing usage is measured on the
    // real API). `output_text_bytes` (the SSE content deltas) is preferred for
    // output when present so response framing doesn't inflate it. Measured
    // completions (any host) and estimated consumer completions are reported;
    // everything else is left at 0 and dropped by the reporter's
    // both-token-counts-zero guard.
    let estimate_this = rule
        .map(|r| r.estimate_unmeasured && r.is_completion(method, path))
        .unwrap_or(false);
    let (tokens_in, tokens_out) = if tokens_measured {
        (fin.tokens_in.unwrap_or(0), fin.tokens_out.unwrap_or(0))
    } else if estimate_this {
        // Per-host tokenizer: a matched rule pins its estimator; fall back to the
        // provider default only if the rule somehow lacks one.
        let tokenizer = rule
            .map(|r| r.tokenizer.as_str())
            .unwrap_or_else(|| tokenizer_for(provider));
        let m = model.as_deref();
        let out_bytes = fin.output_text_bytes.unwrap_or(resp_text_bytes);
        (
            tokens::estimate_tokens(plaintext_len(req_body), tokenizer, m),
            tokens::estimate_tokens(out_bytes, tokenizer, m),
        )
    } else {
        // Not a real completion: a measured-only host with no usage (count_tokens
        // preflight / telemetry), or a consumer-wire non-generation request (a
        // history-load GET, or anything we couldn't HPACK-decode). Leave 0 so the
        // reporter's both-token-counts-zero guard drops it. Surface the drop under
        // debug so a misconfigured consumer wire (metered, model present, but no
        // `estimate_unmeasured`/`completion_path` rule) is diagnosable rather than
        // a silent zero.
        if whitelisted && !tokens_measured && model.is_some() {
            dlog(&format!(
                "conn {} unmeasured model={:?} host={} method={:?} path={:?} left at 0 tokens \
                 (no estimate_unmeasured/completion_path rule match)",
                conn_id, model, host, method, path
            ));
        }
        (0, 0)
    };
    // Anthropic-only prompt-cache breakdown; 0 when absent (non-Anthropic /
    // unmeasured), mirroring the shim.
    let cache_read_tokens = fin.cache_read_tokens.unwrap_or(0);
    let cache_write_tokens = fin.cache_write_tokens.unwrap_or(0);
    let tool_error_names = tool_error_names_in_request(provider, req_body);
    let _chat = conversation_fingerprint(provider, req_body); // session assignment is shim-side; left None here.

    // Break tokens_in (the billable total Anthropic folds cache into) down into
    // fresh + cache_read + cache_write so the split is visible per request, not
    // only on the reporter's scalar series.
    let fresh_in = tokens_in
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    dlog(&format!(
        "conn {} DONE method={:?} path={:?} status={:?} model={:?} tokens_in={} (fresh={} cache_read={} cache_write={}) tokens_out={} measured={} whitelisted={} tool_calls={} tools={:?} bytes_tx={} bytes_rx={} latency_ms={}",
        conn_id, method, path, status, model, tokens_in, fresh_in, cache_read_tokens, cache_write_tokens,
        tokens_out, tokens_measured, whitelisted,
        fin.tool_calls, fin.tool_call_names, bytes_tx, bytes_rx, latency_ms
    ));

    if let Some(tx) = &shared.tx {
        // The reporter's usage/metrics sinks join a `RequestStarted` (host +
        // whitelisted flag) to the `RequestCompleted` by conn_id and only report
        // whitelisted requests. Pass the operator-controlled `whitelisted` flag
        // computed above so the reporter drops requests the whitelist gates out;
        // the proxy funnels only one event per request (the shim emits its own).
        // `inject_headers` stays false — the proxy never injects.
        let _ = tx.try_send(Event::RequestStarted {
            conn_id,
            ts_ms: started_ms,
            host: host.to_string(),
            path: String::new(),
            method: String::new(),
            whitelisted,
            inject_headers: false,
        });
        let ev = Event::RequestCompleted {
            conn_id,
            ts_ms: now_ms(),
            status,
            latency_ms,
            bytes_tx,
            bytes_rx,
            tokens_in,
            tokens_out,
            tokens_measured,
            cache_read_tokens,
            cache_write_tokens,
            tool_calls: fin.tool_calls,
            tool_call_errors: tool_error_names.len() as u64,
            tool_call_names: fin.tool_call_names.clone(),
            tool_call_error_names: tool_error_names,
            chat_id: None,
            model,
        };
        let _ = tx.try_send(ev);
    }
}

/// HTTP/1.1 path: a clean sequential request→response loop (supports
/// keep-alive). curl over this proxy uses this path.
fn relay_h1(
    conn_id: u64,
    provider: Option<Provider>,
    host: String,
    mut server: ServerTls,
    mut upstream: UpstreamTls,
    shared: Arc<Shared>,
) -> std::io::Result<()> {
    loop {
        // --- read one request from the client ---
        let (req_head, leftover) = match read_http_head(&mut server) {
            Ok(Some(v)) => v,
            Ok(None) => break, // client closed
            Err(_) => break,
        };
        let started = now_ms();
        let req_head_str = String::from_utf8_lossy(&req_head);
        let req_line: Vec<&str> = req_head_str
            .lines()
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        let req_method = req_line.first().map(|s| s.to_string());
        let req_path = req_line.get(1).map(|s| s.to_string());
        let req_content_len = header_content_length(&req_head_str);
        let req_chunked = header_is_chunked(&req_head_str);

        // For a provider host, force `Accept-Encoding: identity` so the response
        // comes back uncompressed and the usage scanner can read it (mirrors the
        // shim's `inject::rewrite_headers`); only the header block is edited, the
        // body and its framing are forwarded verbatim. A capture-only host never
        // reads the body, so its request head is relayed verbatim — no point
        // forcing every asset to come back uncompressed.
        let fwd_head = match provider {
            Some(_) => force_identity_request(&req_head),
            None => req_head.clone(),
        };
        upstream.write_all(&fwd_head)?;
        let mut bytes_tx = fwd_head.len() as u64;

        // Stream the WHOLE request body to upstream, retaining only up to
        // REQ_BODY_CAP for the usage scanner. The entire body must reach upstream:
        // truncating it while leaving the original Content-Length in the head
        // would desync the request framing and stall the response, which breaks
        // large requests (e.g. a Code-mode prompt carrying multi-MB of context).
        let mut req_scan: Vec<u8> = Vec::new();
        if !leftover.is_empty() {
            upstream.write_all(&leftover)?;
            bytes_tx += leftover.len() as u64;
            scan_extend(&mut req_scan, &leftover);
        }
        let mut body_fwd = leftover.len();
        if let Some(cl) = req_content_len {
            let mut buf = [0u8; 65536];
            while body_fwd < cl {
                let n = server.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                upstream.write_all(&buf[..n])?;
                bytes_tx += n as u64;
                scan_extend(&mut req_scan, &buf[..n]);
                body_fwd += n;
            }
        } else if req_chunked {
            forward_chunked_body(&mut server, &mut upstream, &mut req_scan, &mut bytes_tx)?;
        }
        upstream.flush()?;
        let req_model =
            provider.and_then(|p| model_from_request(p, &req_scan, req_path.as_deref()));

        // --- read the response, feeding the usage scanner, relaying to client ---
        // The raw response bytes are always relayed to the client verbatim; the
        // scanner keeps its own (possibly decompressed) copy for usage. For a
        // non-provider host (decrypt-all) the scanner runs in capture-only mode:
        // it parses the head (status) and tracks framing for keep-alive, but does
        // no usage parsing or compressed-body buffering.
        let mut scan = match provider {
            Some(p) => H1Scan::new(p),
            None => H1Scan::new_capture(),
        };
        let mut bytes_rx: u64 = 0;
        // First, the head + any body bytes that arrived with it.
        let (resp_head, resp_leftover) = match read_http_head(&mut upstream) {
            Ok(Some(v)) => v,
            _ => {
                // no response; break the connection
                break;
            }
        };
        scan.feed(&resp_head);
        server.write_all(&resp_head)?;
        bytes_rx += resp_head.len() as u64;
        if !resp_leftover.is_empty() {
            scan.feed(&resp_leftover);
            server.write_all(&resp_leftover)?;
            bytes_rx += resp_leftover.len() as u64;
        }
        let resp_head_str = String::from_utf8_lossy(&resp_head);
        let resp_cl = header_content_length(&resp_head_str);
        let resp_chunked = header_is_chunked(&resp_head_str);
        let mut body_seen = resp_leftover.len();

        // Stream the rest of the response.
        loop {
            let complete = if let Some(cl) = resp_cl {
                body_seen >= cl
            } else if resp_chunked {
                scan.is_complete()
            } else {
                false // read until EOF
            };
            if complete {
                break;
            }
            let mut buf = [0u8; 65536];
            let n = match upstream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => break,
            };
            scan.feed(&buf[..n]);
            server.write_all(&buf[..n])?;
            server.flush().ok();
            bytes_rx += n as u64;
            body_seen += n;
        }
        server.flush().ok();

        let status = scan.status();
        let fin = scan.finalize();
        // Read after finalize: a compressed body's plaintext count is only known
        // once finalize inflates it.
        let resp_text_bytes = scan.resp_text_bytes();
        emit(
            &shared, conn_id, provider, &host, started, status, bytes_tx, bytes_rx, resp_text_bytes,
            &fin, req_model, &req_scan, req_method.as_deref(), req_path.as_deref(),
        );

        // Reuse the connection (keep-alive) unless either side asked to close, or
        // the response is close-delimited: no Content-Length AND not chunked, so
        // EOF is the only end-of-body marker. A chunked / SSE response is fully
        // framed and safe to keep alive — forcing a close after every streamed
        // completion would churn reconnects.
        let close_delimited = resp_cl.is_none() && !resp_chunked;
        if header_says_close(&req_head_str) || header_says_close(&resp_head_str) || close_delimited {
            break;
        }
    }
    Ok(())
}

/// One TLS leg (client-facing or upstream-facing): the rustls connection and the
/// write side of its socket, under SEPARATE locks. Reads run on a cloned fd with
/// no lock (see `relay_h2`); the socket held here is used only for writing
/// encrypted output.
///
/// The split is what keeps the full-duplex h2 relay from deadlocking. Both relay
/// directions touch each leg — one to encrypt its own traffic, the other to
/// decrypt the peer's — so the connection is shared and must be locked. But a
/// leg's socket write can BLOCK on backpressure (a slow/greedy peer), and if that
/// block were held under the connection lock the opposite direction could never
/// take the lock to drain its own side, wedging the whole connection (a large
/// request upload stalling the streamed response, or vice-versa). So `conn` is
/// locked only for the in-memory crypto step and dropped before the socket write;
/// `wsock` serializes the writes (preserving TLS record order) and is the only
/// lock held across the blocking network write.
struct Leg {
    conn: Mutex<rustls::Connection>,
    wsock: Mutex<TcpStream>,
}

/// Drain every pending outbound TLS record from `conn` into a buffer, WITHOUT the
/// socket. Encryption assigns each record its implicit sequence number here, so
/// the caller must hold `wsock` across this + the socket write to keep records in
/// order (a peer rejects an out-of-order record with a fatal alert).
fn drain_tls(conn: &mut rustls::Connection) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    while conn.wants_write() {
        if conn.write_tls(&mut out)? == 0 {
            break;
        }
    }
    Ok(out)
}

/// Drain all currently-decrypted plaintext out of `conn` into `out`. rustls caps
/// its received-plaintext buffer, and `process_new_packets` FAILS with "received
/// plaintext buffer full" if a burst of records is decrypted without reading the
/// plaintext out — so this must be called after each `read_tls`, not once at the
/// end, or a large h2 request/response read wedges the connection.
fn drain_plaintext(conn: &mut rustls::Connection, out: &mut Vec<u8>) -> std::io::Result<()> {
    let mut buf = [0u8; 65536];
    loop {
        match conn.reader().read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

/// Feed already-read raw TLS bytes into the leg, advance its state machine, and
/// return the freshly decrypted plaintext. `raw` may be empty to just drain
/// plaintext buffered during the handshake.
///
/// Decryption holds only `conn` (in-memory) and releases it before touching the
/// socket. Any TLS housekeeping the connection wants to send back (alerts / key
/// updates — rare post-handshake) is flushed afterwards, but only if `wsock` is
/// free right now: if the other direction is mid-write on this leg, the queued
/// records are left for it (or the next call) to flush, so decrypting the peer's
/// stream never blocks behind a backpressured write in the other direction.
fn tls_ingest(leg: &Leg, mut raw: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut plaintext = Vec::new();
    let wants_write;
    {
        let mut conn = leg.conn.lock().unwrap();
        while !raw.is_empty() {
            let used = conn.read_tls(&mut raw)?;
            conn.process_new_packets()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            // Drain after every read: rustls bounds its received-plaintext buffer
            // and errors if a multi-record burst is decrypted without reading it
            // out (a large h2 request/response arrives this way).
            drain_plaintext(&mut conn, &mut plaintext)?;
            if used == 0 {
                break;
            }
        }
        conn.process_new_packets()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        drain_plaintext(&mut conn, &mut plaintext)?;
        wants_write = conn.wants_write();
    }
    if wants_write {
        // Best-effort: skip on contention so the decrypt path can't be stalled by
        // the other direction's in-flight write. The queued records stay buffered
        // in `conn` and go out on the next `tls_emit`/`tls_ingest` for this leg.
        if let Ok(mut sock) = leg.wsock.try_lock() {
            let out = { drain_tls(&mut leg.conn.lock().unwrap())? };
            if !out.is_empty() {
                sock.write_all(&out)?;
                sock.flush().ok();
            }
        }
    }
    Ok(plaintext)
}

/// Encrypt `plain` through the leg and send it, forwarding records as they are
/// produced. `wsock` is held for the whole call (serializing writers, preserving
/// record order); `conn` is taken only to encrypt and is dropped before each
/// socket write, so the blocking network write never holds the connection lock.
///
/// rustls caps its outbound plaintext buffer (default 64 KiB) and accepts only a
/// partial write when full, so the plaintext is fed in slices, draining the
/// encrypted records to the socket between slices — this is what lets a large
/// (multi-MB) request body relay cleanly on the h2 leg.
fn tls_emit(leg: &Leg, plain: &[u8]) -> std::io::Result<()> {
    let mut sock = leg.wsock.lock().unwrap();
    let mut off = 0;
    loop {
        let start = off;
        let out = {
            let mut conn = leg.conn.lock().unwrap();
            while off < plain.len() {
                let n = conn.writer().write(&plain[off..])?;
                if n == 0 {
                    break; // rustls' outbound buffer is full; drain then retry
                }
                off += n;
            }
            drain_tls(&mut conn)?
        };
        if !out.is_empty() {
            sock.write_all(&out)?;
        }
        if off >= plain.len() {
            break;
        }
        if off == start && out.is_empty() {
            break; // no forward progress possible; avoid spinning
        }
    }
    sock.flush().ok();
    Ok(())
}

/// Send TLS `close_notify` and shut the leg's socket down, best-effort.
fn tls_shutdown(leg: &Leg) {
    let Ok(mut sock) = leg.wsock.lock() else { return };
    let out = leg
        .conn
        .lock()
        .ok()
        .and_then(|mut conn| {
            conn.send_close_notify();
            drain_tls(&mut conn).ok()
        })
        .unwrap_or_default();
    let _ = sock.write_all(&out);
    let _ = sock.flush();
    let _ = sock.shutdown(std::net::Shutdown::Both);
}

/// HTTP/2 path: full-duplex decrypted-byte relay in both directions, teeing a
/// copy through the h2 frame demux + usage scanner per stream. The plaintext h2
/// bytes are relayed verbatim (never re-framed), keeping the end-to-end HPACK
/// table and flow control consistent; a private, read-only HPACK decoder on the
/// request leg additionally reassembles each request's header block and recovers
/// its `:method`/`:path` for the capture log and DONE line (see t1). Model
/// attribution stays body-only (the JSON body for Anthropic/OpenAI); Gemini's
/// path-based model is not resolved from the recovered path on the h2 path.
///
/// The two directions run on independent threads. Both threads touch each `Leg`
/// (one encrypts its own traffic, the other decrypts the peer's), so a leg's
/// rustls connection is shared under a lock — but that lock is held only for the
/// in-memory crypto step and dropped before the socket write (see `Leg`). A
/// blocking socket write therefore never holds a connection lock, so a
/// backpressured direction (a large request upload, or a client slow to read)
/// can't wedge the other: a long-lived streaming response (many DATA frames over
/// time) is relayed as it arrives, full-duplex, rather than stalling behind the
/// request leg.
#[allow(clippy::too_many_arguments)]
fn relay_h2(
    conn_id: u64,
    provider: Option<Provider>,
    host: String,
    sconn: ServerConnection,
    client_tcp: TcpStream,
    cconn: ClientConnection,
    up_tcp: TcpStream,
    shared: Arc<Shared>,
) -> std::io::Result<()> {
    let started = now_ms();

    // Read fds are cloned so the blocking reads run outside the connection locks;
    // the originals move into the `Leg`s and are used only for writes.
    let mut client_rd = client_tcp.try_clone()?;
    let up_shut = up_tcp.try_clone()?; // A closes upstream (unblocks B) on exit
    let mut up_rd = up_tcp.try_clone()?;
    let client_shut = client_tcp.try_clone()?; // B closes client (unblocks A) on exit

    let sconn = Arc::new(Leg {
        conn: Mutex::new(sconn.into()),
        wsock: Mutex::new(client_tcp),
    });
    let cconn = Arc::new(Leg {
        conn: Mutex::new(cconn.into()),
        wsock: Mutex::new(up_tcp),
    });

    // Per-stream request bodies, shared: written by the request direction, read
    // by the response direction when a stream ends (for model/tool attribution).
    let streams: Arc<Mutex<HashMap<u32, StreamReq>>> = Arc::new(Mutex::new(HashMap::new()));

    // client -> upstream (requests)
    let sconn_a = sconn.clone();
    let cconn_a = cconn.clone();
    let streams_a = streams.clone();
    let t1 = std::thread::spawn(move || {
        let mut fp = h2::FrameParser::new_client();
        // One HPACK decoder for the whole connection: the request dynamic table is
        // cumulative across every header block in arrival order, so each block must
        // reach this decoder exactly once, in order. `new_client` already consumed
        // the connection preface, so the decoder starts in sync with an empty
        // table. `asm` stitches each HEADERS frame together with any following
        // CONTINUATIONs. `hpack_broken` latches on the first decode or reassembly
        // failure — the dynamic table would then be permanently desynced, so we
        // stop decoding (leaving method/path None) rather than surface garbage.
        // None of this affects the verbatim byte relay (`tls_emit`) below.
        let mut decoder = loona_hpack::Decoder::new();
        let mut asm = h2::HeaderBlockAssembler::new();
        let mut hpack_broken = false;
        let mut buf = [0u8; 65536];
        // Relay anything buffered during the handshake, then loop on socket reads.
        let mut first = true;
        loop {
            let plain = if first {
                first = false;
                match tls_ingest(&sconn_a, &[]) {
                    Ok(p) => p,
                    Err(_) => break,
                }
            } else {
                let n = match client_rd.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                match tls_ingest(&sconn_a, &buf[..n]) {
                    Ok(p) => p,
                    Err(_) => break,
                }
            };
            if !plain.is_empty() {
                for f in fp.feed(&plain) {
                    if f.ftype == h2::FRAME_DATA {
                        let mut map = streams_a.lock().unwrap();
                        let e = map.entry(f.stream_id).or_default();
                        if e.body.len() < REQ_BODY_CAP {
                            e.body.extend_from_slice(&f.payload);
                            e.body.truncate(REQ_BODY_CAP);
                        }
                        e.bytes_tx += f.payload.len() as u64;
                    }
                    // Recover `:method`/`:path` by decoding each request header
                    // block. Every block (including ones for streams we don't
                    // otherwise track) must be fed to the decoder exactly once to
                    // keep its dynamic table in sync; on any decode or reassembly
                    // failure the table is unrecoverable, so latch `hpack_broken`
                    // and stop. The byte relay below runs regardless.
                    if !hpack_broken {
                        match asm.feed(&f) {
                            Some(h2::HeaderBlock::Complete(sid, block)) => {
                                match decoder.decode(&block) {
                                    Ok(headers) => {
                                        let mut method = None;
                                        let mut path = None;
                                        for (name, value) in &headers {
                                            if name == b":method" {
                                                method =
                                                    Some(String::from_utf8_lossy(value).into_owned());
                                            } else if name == b":path" {
                                                path =
                                                    Some(String::from_utf8_lossy(value).into_owned());
                                            }
                                        }
                                        // Only overwrite when found so a later
                                        // trailers block (no pseudo-headers) can't
                                        // wipe an earlier method/path.
                                        if method.is_some() || path.is_some() {
                                            let mut map = streams_a.lock().unwrap();
                                            let e = map.entry(sid).or_default();
                                            if method.is_some() {
                                                e.method = method;
                                            }
                                            if path.is_some() {
                                                e.path = path;
                                            }
                                        }
                                    }
                                    Err(_) => hpack_broken = true,
                                }
                            }
                            Some(h2::HeaderBlock::Malformed) => hpack_broken = true,
                            None => {}
                        }
                    }
                }
                if tls_emit(&cconn_a, &plain).is_err() {
                    break;
                }
            }
        }
        // Client done sending: close the upstream write side and unblock B.
        tls_shutdown(&cconn_a);
        let _ = up_shut.shutdown(std::net::Shutdown::Both);
    });

    // upstream -> client (responses)
    let sconn_b = sconn.clone();
    let cconn_b = cconn.clone();
    let streams_b = streams.clone();
    let shared_b = shared.clone();
    let host_b = host.clone();
    let t2 = std::thread::spawn(move || {
        let mut fp = h2::FrameParser::new_server();
        let mut resp: HashMap<u32, RespState> = HashMap::new();
        let mut buf = [0u8; 65536];
        let mut first = true;
        loop {
            let plain = if first {
                first = false;
                match tls_ingest(&cconn_b, &[]) {
                    Ok(p) => p,
                    Err(_) => break,
                }
            } else {
                let n = match up_rd.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                match tls_ingest(&cconn_b, &buf[..n]) {
                    Ok(p) => p,
                    Err(_) => break,
                }
            };
            if plain.is_empty() {
                continue;
            }
            if tls_emit(&sconn_b, &plain).is_err() {
                break;
            }
            for f in fp.feed(&plain) {
                if f.ftype != h2::FRAME_DATA {
                    continue;
                }
                let st = resp.entry(f.stream_id).or_insert_with(|| match provider {
                    Some(p) => RespState::new(p),
                    None => RespState::new_capture(),
                });
                st.feed(&f.payload);
                st.bytes_rx += f.payload.len() as u64;
                if f.end_stream() {
                    let fin = st.finish();
                    // The provider path keeps `new_h2_body`'s 200 default (the
                    // usage sink's contract); capture-only reports `None`, since
                    // the real h2 `:status` lives in HPACK we never decode.
                    let status = if provider.is_some() { st.parser.status } else { None };
                    let bytes_rx = st.bytes_rx;
                    // Decompressed response text bytes for the output estimate
                    // (populated by feed + finish); read before `st` is removed.
                    let resp_text_bytes = st.resp_text_bytes();
                    let (req_body, bytes_tx, req_method, req_path) = {
                        let map = streams_b.lock().unwrap();
                        map.get(&f.stream_id)
                            .map(|r| {
                                (r.body.clone(), r.bytes_tx, r.method.clone(), r.path.clone())
                            })
                            .unwrap_or_default()
                    };
                    // The request `:method`/`:path` were recovered by decoding this
                    // stream's HPACK header block on the request leg (t1); either is
                    // None if HPACK decoding gave up for the connection. Model
                    // attribution stays body-only here — the recovered path feeds
                    // only the capture log and the DONE line, not model resolution.
                    let req_model = provider.and_then(|p| model_from_request(p, &req_body, None));
                    emit(
                        &shared_b, conn_id, provider, &host_b, started, status, bytes_tx, bytes_rx,
                        resp_text_bytes, &fin, req_model, &req_body,
                        req_method.as_deref(), req_path.as_deref(),
                    );
                    resp.remove(&f.stream_id);
                    streams_b.lock().unwrap().remove(&f.stream_id);
                }
            }
        }
        // Upstream done: close the client write side and unblock A.
        tls_shutdown(&sconn_b);
        let _ = client_shut.shutdown(std::net::Shutdown::Both);
    });

    let _ = t1.join();
    let _ = t2.join();
    Ok(())
}

#[derive(Default)]
struct StreamReq {
    body: Vec<u8>,
    bytes_tx: u64,
    /// Request `:method`/`:path`, decoded from this stream's HPACK header block on
    /// the client→upstream leg (t1) and read by the response leg (t2) when the
    /// stream ends. `None` until decoded, or if HPACK decoding gave up for the
    /// connection.
    method: Option<String>,
    path: Option<String>,
}

/// Cap on a compressed h2 response body buffered for whole-stream inflation.
const COMP_BODY_CAP: usize = 16 * 1024 * 1024;

struct RespState {
    parser: RespParse,
    started: bool,
    provider: Provider,
    /// False for a non-provider host under decrypt-all: only bytes/timing are
    /// captured, so `feed`/`finish` skip all body parsing and buffering.
    usage: bool,
    bytes_rx: u64,
    /// Decompressed response body bytes fed to the parser, for the output-token
    /// estimate when the response carried no measured `usage` (the claude.ai
    /// chat panel). Counts the inflated plaintext, not the compressed wire bytes.
    plain_out: u64,
    /// When the h2 body is gzip/zstd compressed we can't feed it incrementally
    /// (the scanner wants plaintext), so we buffer the compressed bytes and
    /// inflate the whole thing at END_STREAM. `None` for identity bodies, which
    /// stream straight into the parser.
    comp: Option<(decompress::Encoding, Vec<u8>)>,
}

impl RespState {
    fn new(provider: Provider) -> Self {
        Self {
            parser: RespParse::new_h2_body(provider, false),
            started: false,
            provider,
            usage: true,
            bytes_rx: 0,
            plain_out: 0,
            comp: None,
        }
    }
    /// Capture-only state for a non-provider h2 stream: tracks nothing but the
    /// byte count (kept by the caller), so no usage parsing runs. `parser.status`
    /// stays `None` — h2 response status lives in HPACK, which isn't decoded.
    fn new_capture() -> Self {
        let mut s = Self::new(Provider::Anthropic);
        s.usage = false;
        s
    }
    fn feed(&mut self, data: &[u8]) {
        if !self.usage {
            return;
        }
        if !self.started {
            // Sniff SSE vs JSON and gzip/zstd from the first DATA bytes.
            let enc = decompress::detect(data);
            let sniff: Vec<u8> = if enc.is_compressed() {
                decompress::decompress(enc, data, 4096).unwrap_or_default()
            } else {
                data.to_vec()
            };
            let sse = looks_like_sse(&sniff);
            self.parser = RespParse::new_h2_body(self.provider, sse);
            // The proxy isn't the app's hot path (this runs after the bytes are
            // already forwarded to the client), so count the generated-text bytes
            // for the output estimate; the whole SSE envelope would overcount.
            self.parser.enable_output_text_count();
            self.started = true;
            if enc.is_compressed() {
                self.comp = Some((enc, Vec::new()));
            }
        }
        match &mut self.comp {
            Some((_enc, buf)) => {
                if buf.len() < COMP_BODY_CAP {
                    buf.extend_from_slice(data);
                }
            }
            None => {
                self.parser.feed(data);
                self.plain_out += data.len() as u64;
            }
        }
    }
    /// Inflate any buffered compressed body, feed it, and finalize.
    fn finish(&mut self) -> body_scan::Finalized {
        if !self.usage {
            return body_scan::Finalized::default();
        }
        if let Some((enc, buf)) = self.comp.take() {
            if let Some(plain) = decompress::decompress(enc, &buf, COMP_BODY_CAP) {
                self.plain_out += plain.len() as u64;
                self.parser.feed(&plain);
            }
        }
        self.parser.finalize()
    }
    /// Decompressed response body bytes seen, for the output-token estimate.
    fn resp_text_bytes(&self) -> u64 {
        self.plain_out
    }
}

// --- HTTP/1.1 response usage scanner (with decompression) ---------------------

/// How the scanner handles an HTTP/1.1 response body once the head is parsed.
enum H1Body {
    /// Identity body: raw bytes stream straight into `RespParse` (which parses
    /// the head, de-chunks, and reads usage) — the existing zero-copy path.
    Stream(RespParse),
    /// gzip/zstd body: the raw (still chunk-framed) body is buffered and, at
    /// completion, de-chunked then inflated before being fed to a body-mode
    /// parser as plaintext. Mirrors the h2 leg's `RespState`.
    Compressed {
        parser: RespParse,
        enc: decompress::Encoding,
        chunked: bool,
        raw: Vec<u8>,
    },
    /// Capture-only body (non-provider host under decrypt-all): no usage parsing
    /// and no buffering — the body bytes are relayed and discarded. `RespParse`
    /// can't be reused here because it gives up on non-LLM content types and so
    /// never de-chunks, leaving a chunked keep-alive body's boundary undetected;
    /// a chunked body carries its own lightweight framing scanner instead.
    /// `None` for a Content-Length/close-delimited body (the relay loop frames
    /// those without help).
    Capture(Option<ChunkScan>),
}

/// Streaming detector for the end of an HTTP/1.1 chunked body. Parses only the
/// chunk framing (sizes + the terminal 0-size chunk), retaining no payload, so a
/// capture-only relay finds the keep-alive boundary without buffering the body.
#[derive(Default)]
struct ChunkScan {
    state: ChunkState,
    /// Bytes remaining in the current chunk's data section.
    size: usize,
    /// Accumulator for a size line / CRLF / trailer line (small, cleared often).
    line: Vec<u8>,
    done: bool,
}

#[derive(Default, PartialEq)]
enum ChunkState {
    /// Reading the hex chunk-size line up to its CRLF.
    #[default]
    Size,
    /// Consuming `size` data bytes.
    Data,
    /// Consuming the CRLF that follows a chunk's data.
    DataCrlf,
    /// After the 0-size chunk: consuming trailers until the blank line.
    Trailer,
}

impl ChunkScan {
    fn feed(&mut self, mut data: &[u8]) {
        while !data.is_empty() && !self.done {
            match self.state {
                ChunkState::Size => {
                    self.line.push(data[0]);
                    data = &data[1..];
                    if self.line.ends_with(b"\r\n") {
                        let size = parse_hex_prefix(&self.line);
                        self.line.clear();
                        if size == 0 {
                            self.state = ChunkState::Trailer;
                        } else {
                            self.size = size;
                            self.state = ChunkState::Data;
                        }
                    }
                }
                ChunkState::Data => {
                    let take = self.size.min(data.len());
                    self.size -= take;
                    data = &data[take..];
                    if self.size == 0 {
                        self.state = ChunkState::DataCrlf;
                    }
                }
                ChunkState::DataCrlf => {
                    self.line.push(data[0]);
                    data = &data[1..];
                    if self.line.ends_with(b"\r\n") || self.line.len() >= 2 {
                        self.line.clear();
                        self.state = ChunkState::Size;
                    }
                }
                ChunkState::Trailer => {
                    self.line.push(data[0]);
                    data = &data[1..];
                    if self.line.ends_with(b"\r\n") {
                        // A blank line (bare CRLF) terminates the trailer section.
                        if self.line == b"\r\n" {
                            self.done = true;
                        }
                        self.line.clear();
                    }
                }
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.done
    }
}

/// Parse-side scanner for an HTTP/1.1 response. Extracts usage/model from the
/// body while the raw response bytes are relayed to the client verbatim
/// (elsewhere). Compression is handled here so the shared `body_scan` parser
/// only ever sees plaintext, exactly as on the h2 leg:
///   * Identity bodies stream straight into `RespParse` (body not copied).
///   * gzip/zstd bodies — flagged by the response `Content-Encoding`, or by a
///     gzip/zstd magic-byte sniff of the first body bytes as a fallback for a
///     server that ignores our forced `Accept-Encoding: identity` — are
///     de-chunked, buffered and inflated at completion, then fed as plaintext.
struct H1Scan {
    provider: Provider,
    /// False for a non-provider host under decrypt-all: parse the head (for
    /// status) and de-chunk for keep-alive framing, but do no usage parsing and
    /// no compressed-body buffering (every body stays on the identity stream
    /// path, so nothing is inflated or held).
    usage: bool,
    head_done: bool,
    head_buf: Vec<u8>,
    status: Option<u16>,
    sse: bool,
    chunked: bool,
    /// Content-Encoding declared by the head (`Identity` until the head says
    /// otherwise, or if it named an encoding we can't inflate — br/deflate —
    /// which then stays on the identity stream path, no worse than before).
    head_enc: decompress::Encoding,
    body_started: bool,
    body: Option<H1Body>,
    /// Decompressed response body bytes seen, for the output-token estimate when
    /// the response carried no measured `usage` (the claude.ai chat panel).
    plain_out: u64,
}

impl H1Scan {
    fn new(provider: Provider) -> Self {
        Self {
            provider,
            usage: true,
            head_done: false,
            head_buf: Vec::new(),
            status: None,
            sse: false,
            chunked: false,
            head_enc: decompress::Encoding::Identity,
            body_started: false,
            body: None,
            plain_out: 0,
        }
    }

    /// Capture-only scanner (non-provider host): status + framing, no usage.
    fn new_capture() -> Self {
        let mut s = Self::new(Provider::Anthropic);
        s.usage = false;
        s
    }

    /// Feed raw response bytes (head first, then body chunks) as they stream.
    fn feed(&mut self, data: &[u8]) {
        if !self.head_done {
            self.head_buf.extend_from_slice(data);
            let pos = match find_subslice(&self.head_buf, b"\r\n\r\n") {
                Some(p) => p,
                None => {
                    // Bound head accumulation; the caller's read loop also caps.
                    if self.head_buf.len() > 256 * 1024 {
                        self.head_done = true;
                    }
                    return;
                }
            };
            let head = self.head_buf[..pos + 4].to_vec();
            let remainder = self.head_buf.split_off(pos + 4);
            self.head_buf = Vec::new();
            self.parse_head(&head);
            self.head_done = true;
            if !remainder.is_empty() {
                self.feed_body(&remainder);
            }
            return;
        }
        self.feed_body(data);
    }

    fn parse_head(&mut self, head: &[u8]) {
        let hs = String::from_utf8_lossy(head);
        let parsed = parser::parse_response_head(head);
        self.status = parsed.as_ref().map(|h| h.status);
        self.sse = parsed.as_ref().map(|h| h.is_event_stream).unwrap_or(false);
        self.chunked = parsed
            .as_ref()
            .map(|h| h.chunked)
            .unwrap_or_else(|| header_is_chunked(&hs));
        self.head_enc = encoding_from_header(header_value(&hs, "content-encoding"));

        if !self.usage {
            // Capture-only: no usage parsing, no buffering. Only a chunked body
            // needs framing help (to find the keep-alive boundary); Content-Length
            // and close-delimited bodies are framed by the relay loop itself.
            self.body = Some(H1Body::Capture(self.chunked.then(ChunkScan::default)));
        } else if self.head_enc.is_compressed() {
            self.body = Some(H1Body::Compressed {
                parser: self.new_body_parser(),
                enc: self.head_enc,
                chunked: self.chunked,
                raw: Vec::new(),
            });
        } else {
            // Identity path: hand the parser the full head so it parses status /
            // content-type / transfer-encoding and runs its own de-chunker.
            let mut rp = RespParse::new(self.provider);
            rp.feed(head);
            self.body = Some(H1Body::Stream(rp));
        }
    }

    /// A body-mode parser (no HTTP head), carrying the real status from the head.
    fn new_body_parser(&self) -> RespParse {
        let mut p = RespParse::new_h2_body(self.provider, self.sse);
        if let Some(s) = self.status {
            p.status = Some(s);
        }
        p
    }

    fn feed_body(&mut self, data: &[u8]) {
        if !self.body_started {
            self.body_started = true;
            // Fallback for a server that ignored `Accept-Encoding: identity` and
            // sent a compressed body with NO `Content-Encoding` header: sniff the
            // first body bytes for the gzip/zstd magic and promote to the
            // buffered path before any byte is consumed as plaintext. Chunked
            // bodies open with the chunk-size line (magic isn't at offset 0), so
            // this only fires for non-chunked bodies; chunked-compressed is
            // caught by the Content-Encoding header instead.
            if self.usage && matches!(self.body, Some(H1Body::Stream(_))) && !self.chunked {
                let enc = decompress::detect(data);
                if enc.is_compressed() {
                    self.body = Some(H1Body::Compressed {
                        parser: self.new_body_parser(),
                        enc,
                        chunked: false,
                        raw: Vec::new(),
                    });
                }
            }
        }
        match self.body.as_mut() {
            Some(H1Body::Stream(rp)) => {
                rp.feed(data);
                // Identity body streams as plaintext (chunk framing overhead is
                // negligible for the estimate); count it for the output estimate.
                self.plain_out += data.len() as u64;
            }
            Some(H1Body::Compressed { raw, .. }) => {
                let room = COMP_BODY_CAP.saturating_sub(raw.len());
                if room > 0 {
                    let take = data.len().min(room);
                    raw.extend_from_slice(&data[..take]);
                }
            }
            Some(H1Body::Capture(Some(cs))) => cs.feed(data),
            Some(H1Body::Capture(None)) | None => {}
        }
    }

    /// True once the body is fully received. Only meaningful for chunked
    /// framing (the caller uses Content-Length / EOF otherwise): identity
    /// delegates to the parser's de-chunker, compressed checks the raw buffer
    /// for the terminal 0-size chunk.
    fn is_complete(&self) -> bool {
        match &self.body {
            Some(H1Body::Stream(rp)) => rp.is_complete(),
            Some(H1Body::Compressed { chunked, raw, .. }) => *chunked && raw_chunk_terminated(raw),
            Some(H1Body::Capture(Some(cs))) => cs.is_complete(),
            Some(H1Body::Capture(None)) | None => false,
        }
    }

    fn status(&self) -> Option<u16> {
        self.status
    }

    /// Inflate any buffered compressed body and return the final usage.
    fn finalize(&mut self) -> body_scan::Finalized {
        match self.body.as_mut() {
            Some(H1Body::Stream(rp)) => rp.finalize(),
            Some(H1Body::Compressed {
                parser,
                enc,
                chunked,
                raw,
            }) => {
                let entity = if *chunked {
                    dechunk_all(raw)
                } else {
                    std::mem::take(raw)
                };
                match decompress::decompress(*enc, &entity, COMP_BODY_CAP) {
                    Some(plain) => {
                        self.plain_out += plain.len() as u64;
                        parser.feed(&plain);
                    }
                    // Inflate failed (truncated/corrupt): feed the entity as-is
                    // as a best effort — it may have been plaintext after all.
                    None => {
                        self.plain_out += entity.len() as u64;
                        parser.feed(&entity);
                    }
                }
                parser.finalize()
            }
            Some(H1Body::Capture(_)) | None => body_scan::Finalized::default(),
        }
    }

    /// Decompressed response body bytes seen, for the output-token estimate.
    /// For a compressed body this is populated by `finalize` (which inflates),
    /// so call it after `finalize`.
    fn resp_text_bytes(&self) -> u64 {
        self.plain_out
    }
}

/// Map a `Content-Encoding` header value to an encoding we can inflate. Only
/// gzip and zstd are supported (matching `decompress`); anything else
/// (identity, br, deflate, absent) maps to `Identity` so the body stays on the
/// plaintext stream path.
fn encoding_from_header(v: Option<&str>) -> decompress::Encoding {
    match v {
        Some(v) => {
            let v = v.to_ascii_lowercase();
            if v.contains("gzip") || v.contains("x-gzip") {
                decompress::Encoding::Gzip
            } else if v.contains("zstd") {
                decompress::Encoding::Zstd
            } else {
                decompress::Encoding::Identity
            }
        }
        None => decompress::Encoding::Identity,
    }
}

/// De-chunk a COMPLETE HTTP/1.1 chunked body (all bytes buffered), returning
/// the concatenated chunk payloads up to the terminal 0-size chunk. A truncated
/// tail keeps whatever payload was decoded (the usage prelude often survives).
fn dechunk_all(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let nl = match find_subslice(&data[i..], b"\r\n") {
            Some(p) => i + p,
            None => break,
        };
        let size = parse_hex_prefix(&data[i..nl]);
        i = nl + 2;
        if size == 0 {
            break;
        }
        if i + size > data.len() {
            out.extend_from_slice(&data[i..]); // truncated final chunk
            break;
        }
        out.extend_from_slice(&data[i..i + size]);
        i += size;
        if data[i..].starts_with(b"\r\n") {
            i += 2; // trailing CRLF after the chunk body
        }
    }
    out
}

/// Parse leading hex digits as a chunk size, stopping at the first non-hex byte
/// (e.g. a `;` chunk extension). Mirrors `body_scan`'s de-chunker.
fn parse_hex_prefix(s: &[u8]) -> usize {
    let mut n: usize = 0;
    for &b in s {
        let d = match b {
            b'0'..=b'9' => (b - b'0') as usize,
            b'a'..=b'f' => (b - b'a' + 10) as usize,
            b'A'..=b'F' => (b - b'A' + 10) as usize,
            _ => break,
        };
        n = n.saturating_mul(16).saturating_add(d);
    }
    n
}

/// True if a buffered chunked body carries the terminal 0-size chunk.
fn raw_chunk_terminated(raw: &[u8]) -> bool {
    raw.ends_with(b"0\r\n\r\n")
        || raw.starts_with(b"0\r\n\r\n")
        || find_subslice(raw, b"\r\n0\r\n\r\n").is_some()
}

/// Rewrite a relayed HTTP/1.1 request head so the upstream returns an
/// uncompressed body: drop any client `Accept-Encoding` and set
/// `accept-encoding: identity`. A duplicate would be comma-combined by the
/// server (`gzip, identity`) and gzip could still win, so the original is
/// removed rather than appended to (mirrors the shim's `inject`). Only the
/// header block is touched; the request body and its framing are untouched.
fn force_identity_request(head: &[u8]) -> Vec<u8> {
    // Header terminator: the first `\r\n\r\n`. `insert_at` is the start of the
    // empty-line terminator (just past the last header's CRLF).
    let eoh = match find_subslice(head, b"\r\n\r\n") {
        Some(p) => p,
        None => return head.to_vec(), // partial head: forward verbatim
    };
    let insert_at = eoh + 2;
    let mut out = Vec::with_capacity(head.len() + 32);
    let mut start = 0;
    while start < insert_at {
        let end = match find_subslice(&head[start..insert_at], b"\r\n") {
            Some(p) => start + p,
            None => insert_at,
        };
        let line = &head[start..end];
        if !is_accept_encoding_line(line) {
            out.extend_from_slice(line);
            out.extend_from_slice(b"\r\n");
        }
        start = (end + 2).min(insert_at);
    }
    out.extend_from_slice(b"accept-encoding: identity\r\n");
    out.extend_from_slice(&head[insert_at..]); // empty-line terminator
    out
}

/// True if `line` is an `Accept-Encoding` header (name compared
/// case-insensitively). The request line has no matching name before its first
/// `:`, so it's never stripped.
fn is_accept_encoding_line(line: &[u8]) -> bool {
    match line.iter().position(|&b| b == b':') {
        Some(colon) => line[..colon].eq_ignore_ascii_case(b"accept-encoding"),
        None => false,
    }
}

// --- HTTP/1 helpers -----------------------------------------------------------

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Read an HTTP/1 head (up to and including CRLFCRLF). Returns the head plus any
/// body bytes that arrived in the same read. `None` on a clean EOF with no data.
fn read_http_head<R: Read>(r: &mut R) -> std::io::Result<Option<(Vec<u8>, Vec<u8>)>> {
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut tmp = [0u8; 8192];
    loop {
        let n = r.read(&mut tmp)?;
        if n == 0 {
            if buf.is_empty() {
                return Ok(None);
            }
            return Ok(Some((buf, Vec::new())));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            let head = buf[..pos + 4].to_vec();
            let leftover = buf[pos + 4..].to_vec();
            return Ok(Some((head, leftover)));
        }
        if buf.len() > 256 * 1024 {
            return Ok(Some((buf, Vec::new())));
        }
    }
}

fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    for line in head.lines().skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case(name) {
                return Some(v.trim());
            }
        }
    }
    None
}

fn header_content_length(head: &str) -> Option<usize> {
    header_value(head, "content-length").and_then(|v| v.parse().ok())
}

fn header_is_chunked(head: &str) -> bool {
    header_value(head, "transfer-encoding")
        .map(|v| v.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false)
}

fn header_says_close(head: &str) -> bool {
    header_value(head, "connection")
        .map(|v| v.eq_ignore_ascii_case("close"))
        .unwrap_or(false)
}

/// Append `chunk` to a capped usage-scan buffer (retains at most `REQ_BODY_CAP`
/// bytes; excess is dropped from the scan copy only — never from what is
/// forwarded).
fn scan_extend(scan: &mut Vec<u8>, chunk: &[u8]) {
    if scan.len() < REQ_BODY_CAP {
        let take = (REQ_BODY_CAP - scan.len()).min(chunk.len());
        scan.extend_from_slice(&chunk[..take]);
    }
}

/// Forward an HTTP/1.1 chunked request body from `from` to `to`, verbatim
/// (framing preserved), until the terminal zero-size chunk. The complete body is
/// always forwarded; `scan` accumulates the decoded chunk payloads capped at
/// REQ_BODY_CAP for usage parsing. `bytes_tx` is advanced by every forwarded
/// byte. Chunk-size lines are read a byte at a time (they are short); chunk data
/// is bulk-read.
fn forward_chunked_body<R: Read, W: Write>(
    from: &mut R,
    to: &mut W,
    scan: &mut Vec<u8>,
    bytes_tx: &mut u64,
) -> std::io::Result<()> {
    let mut one = [0u8; 1];
    loop {
        // Chunk-size line, up to and including CRLF, forwarded as read.
        let mut line = Vec::with_capacity(16);
        loop {
            let n = from.read(&mut one)?;
            if n == 0 {
                return Ok(()); // truncated stream; stop
            }
            to.write_all(&one)?;
            *bytes_tx += 1;
            line.push(one[0]);
            if line.ends_with(b"\r\n") || line.len() > 64 {
                break;
            }
        }
        let size = parse_hex_prefix(&line);
        if size == 0 {
            // Terminal chunk: forward any trailer bytes up to the final blank line.
            let mut end = Vec::with_capacity(4);
            loop {
                let n = from.read(&mut one)?;
                if n == 0 {
                    break;
                }
                to.write_all(&one)?;
                *bytes_tx += 1;
                end.push(one[0]);
                if end.ends_with(b"\r\n") || end.len() > 4096 {
                    break;
                }
            }
            return Ok(());
        }
        // Chunk data (exactly `size` bytes) then its trailing CRLF, all forwarded.
        let mut remaining = size;
        let mut buf = [0u8; 65536];
        while remaining > 0 {
            let want = remaining.min(buf.len());
            let n = from.read(&mut buf[..want])?;
            if n == 0 {
                return Ok(());
            }
            to.write_all(&buf[..n])?;
            *bytes_tx += n as u64;
            scan_extend(scan, &buf[..n]);
            remaining -= n;
        }
        let mut got = 0;
        let mut crlf = [0u8; 2];
        while got < 2 {
            let n = from.read(&mut crlf[got..])?;
            if n == 0 {
                break;
            }
            to.write_all(&crlf[got..got + n])?;
            *bytes_tx += n as u64;
            got += n;
        }
    }
}

// --- optional reporter --------------------------------------------------------

/// Start the in-process reporter iff the agent handed off a descriptor via
/// `CLEARML_SNUG_CRED` (base64 JSON). Returns the event sender + handle, or
/// `(None, None)` when no descriptor is present (standalone/dev runs, which just
/// log parsed usage lines).
fn start_optional_reporter() -> (Option<SyncSender<Event>>, Option<ReporterHandle>) {
    use base64::Engine;
    let cred = match std::env::var("CLEARML_SNUG_CRED") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            log("no CLEARML_SNUG_CRED: running in log-only mode (no backend reporting)");
            return (None, None);
        }
    };
    let json = match base64::engine::general_purpose::STANDARD.decode(cred.trim()) {
        Ok(b) => b,
        Err(e) => {
            log(&format!("CLEARML_SNUG_CRED not valid base64: {} (log-only mode)", e));
            return (None, None);
        }
    };
    let json = String::from_utf8_lossy(&json).to_string();
    let desc = match Descriptor::from_json_str(&json) {
        Ok(d) => d,
        Err(e) => {
            log(&format!("descriptor parse failed: {} (log-only mode)", e));
            return (None, None);
        }
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<Event>(8192);
    let handle = start_reporter(desc, rx, None);
    log("reporter started (backend usage reporting enabled)");
    (Some(tx), Some(handle))
}

/// Load PEM CA certs from a file into rustls cert types.
fn pem_certs(path: &str) -> std::io::Result<Vec<rustls::pki_types::CertificateDer<'static>>> {
    let data = std::fs::read(path)?;
    let mut rd = std::io::BufReader::new(&data[..]);
    let mut out = Vec::new();
    for c in rustls_pemfile::certs(&mut rd) {
        out.push(c?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    fn chunk_encode(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for piece in body.chunks(29) {
            out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
            out.extend_from_slice(piece);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }

    const ANTHROPIC_JSON: &[u8] = br#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-20250514","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1234,"output_tokens":567}}"#;

    // --- force_identity_request ---------------------------------------------

    #[test]
    fn identity_replaces_existing_accept_encoding() {
        let head = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\nAccept-Encoding: gzip, br\r\nContent-Length: 5\r\n\r\n";
        let out = force_identity_request(head);
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("accept-encoding: identity\r\n"));
        assert!(!s.to_ascii_lowercase().contains("gzip"), "old value dropped");
        assert!(s.contains("Content-Length: 5\r\n"), "other headers preserved");
        assert!(s.ends_with("\r\n\r\n"), "terminator intact");
        // Exactly one Accept-Encoding line.
        assert_eq!(s.to_ascii_lowercase().matches("accept-encoding:").count(), 1);
    }

    #[test]
    fn identity_added_when_absent() {
        let head = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n";
        let s = String::from_utf8(force_identity_request(head)).unwrap();
        assert!(s.contains("accept-encoding: identity\r\n"));
        assert!(s.starts_with("POST /v1/messages HTTP/1.1\r\n"));
        assert!(s.ends_with("\r\n\r\n"));
    }

    // --- H1Scan decompression ------------------------------------------------

    fn feed_bytewise(scan: &mut H1Scan, data: &[u8]) {
        for b in data {
            scan.feed(std::slice::from_ref(b));
        }
    }

    #[test]
    fn h1scan_gzip_content_length_json() {
        let gz = gzip(ANTHROPIC_JSON);
        let mut resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\n\r\n",
            gz.len()
        )
        .into_bytes();
        resp.extend_from_slice(&gz);
        let mut scan = H1Scan::new(Provider::Anthropic);
        scan.feed(&resp);
        let fin = scan.finalize();
        assert_eq!(scan.status(), Some(200));
        assert_eq!(fin.tokens_in, Some(1234));
        assert_eq!(fin.tokens_out, Some(567));
        assert_eq!(fin.model.as_deref(), Some("claude-opus-4-20250514"));
    }

    #[test]
    fn h1scan_gzip_chunked_json_split_reads() {
        let gz = gzip(ANTHROPIC_JSON);
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        resp.extend_from_slice(&chunk_encode(&gz));
        let mut scan = H1Scan::new(Provider::Anthropic);
        feed_bytewise(&mut scan, &resp);
        assert!(scan.is_complete(), "terminal chunk seen");
        let fin = scan.finalize();
        assert_eq!(fin.tokens_in, Some(1234));
        assert_eq!(fin.tokens_out, Some(567));
    }

    #[test]
    fn h1scan_gzip_without_content_encoding_header_is_sniffed() {
        // Server ignored identity and sent gzip with NO Content-Encoding header
        // (non-chunked): the magic-byte sniff must still inflate it.
        let gz = gzip(ANTHROPIC_JSON);
        let mut resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            gz.len()
        )
        .into_bytes();
        resp.extend_from_slice(&gz);
        let mut scan = H1Scan::new(Provider::Anthropic);
        scan.feed(&resp);
        let fin = scan.finalize();
        assert_eq!(fin.tokens_in, Some(1234));
        assert_eq!(fin.tokens_out, Some(567));
    }

    #[test]
    fn h1scan_identity_json_unchanged_path() {
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_JSON);
        let mut scan = H1Scan::new(Provider::Anthropic);
        scan.feed(&resp);
        let fin = scan.finalize();
        assert_eq!(fin.tokens_in, Some(1234));
        assert_eq!(fin.tokens_out, Some(567));
    }

    #[test]
    fn h1scan_resp_text_bytes_counts_decompressed_output() {
        // The output-token estimate feeds on the DECOMPRESSED body length, not
        // the compressed wire bytes: an identity body counts its body length,
        // and a gzip body counts the inflated length (> the gzip'd size).
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        resp.extend_from_slice(ANTHROPIC_JSON);
        let mut id = H1Scan::new(Provider::Anthropic);
        id.feed(&resp);
        let _ = id.finalize();
        assert_eq!(id.resp_text_bytes(), ANTHROPIC_JSON.len() as u64);

        let gz = gzip(ANTHROPIC_JSON);
        let mut resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\n\r\n",
            gz.len()
        )
        .into_bytes();
        resp.extend_from_slice(&gz);
        let mut c = H1Scan::new(Provider::Anthropic);
        c.feed(&resp);
        let _ = c.finalize();
        assert_eq!(
            c.resp_text_bytes(),
            ANTHROPIC_JSON.len() as u64,
            "gzip body counts inflated plaintext length"
        );
    }

    #[test]
    fn h1scan_gzip_sse_stream() {
        let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":736,\"output_tokens\":1}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":21}}\n\n";
        let gz = gzip(sse);
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        resp.extend_from_slice(&chunk_encode(&gz));
        let mut scan = H1Scan::new(Provider::Anthropic);
        scan.feed(&resp);
        let fin = scan.finalize();
        assert_eq!(fin.tokens_in, Some(736));
        assert_eq!(fin.tokens_out, Some(21));
        assert_eq!(fin.model.as_deref(), Some("claude-haiku-4-5"));
    }

    #[test]
    fn h1scan_sse_cache_read_dominant_input() {
        // The real Claude Code code-path shape over the proxy: a chunked
        // `text/event-stream` completion whose `message_start` reports a tiny
        // fresh `input_tokens` with the prompt bulk in `cache_read_input_tokens`.
        // The scanner must measure the full prompt (2 + 45000) and the model, so
        // a cache-heavy turn is no longer logged as `tokens_in=2 measured=false`.
        let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":2,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":45000,\"output_tokens\":1}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":13}}\n\n";
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        resp.extend_from_slice(&chunk_encode(sse));
        let mut scan = H1Scan::new(Provider::Anthropic);
        feed_bytewise(&mut scan, &resp);
        assert!(scan.is_complete(), "terminal chunk seen");
        let fin = scan.finalize();
        assert_eq!(fin.tokens_in, Some(45002));
        assert_eq!(fin.tokens_out, Some(13));
        assert_eq!(fin.model.as_deref(), Some("claude-sonnet-5"));
    }

    #[test]
    fn h1scan_gzip_sse_cache_read_dominant_input() {
        // Same cache-heavy SSE completion, but gzip-compressed (a server that
        // ignored our forced `Accept-Encoding: identity`): inflate then sum.
        let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":2,\"cache_creation_input_tokens\":11,\"cache_read_input_tokens\":45000,\"output_tokens\":1}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":13}}\n\n";
        let gz = gzip(sse);
        let mut resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        resp.extend_from_slice(&chunk_encode(&gz));
        let mut scan = H1Scan::new(Provider::Anthropic);
        scan.feed(&resp);
        let fin = scan.finalize();
        assert_eq!(fin.tokens_in, Some(45013));
        assert_eq!(fin.tokens_out, Some(13));
    }

    #[test]
    fn dechunk_all_roundtrip() {
        let body = b"the quick brown fox jumps over the lazy dog, twice for good measure";
        assert_eq!(dechunk_all(&chunk_encode(body)), body);
    }

    #[test]
    fn forward_chunked_body_forwards_all_bytes_verbatim() {
        // A body larger than one read buffer, chunk-encoded, must be forwarded
        // byte-for-byte (framing preserved) while the scan copy stays capped.
        let payload = vec![b'z'; 200 * 1024];
        let encoded = chunk_encode(&payload);
        let mut src = std::io::Cursor::new(encoded.clone());
        let mut dst: Vec<u8> = Vec::new();
        let mut scan: Vec<u8> = Vec::new();
        let mut bytes_tx: u64 = 0;
        forward_chunked_body(&mut src, &mut dst, &mut scan, &mut bytes_tx).unwrap();
        assert_eq!(dst, encoded, "full chunked body forwarded verbatim");
        assert_eq!(bytes_tx as usize, encoded.len(), "bytes_tx counts every byte");
        assert_eq!(dechunk_all(&dst), payload, "decodes back to the original body");
        assert_eq!(scan, payload, "scan holds the decoded payload (under the cap)");
    }

    #[test]
    fn scan_extend_caps_at_req_body_cap() {
        let mut scan = Vec::new();
        scan_extend(&mut scan, &vec![1u8; REQ_BODY_CAP + 4096]);
        assert_eq!(scan.len(), REQ_BODY_CAP, "scan copy capped");
        scan_extend(&mut scan, &[2u8; 16]);
        assert_eq!(scan.len(), REQ_BODY_CAP, "no growth past the cap");
    }

    #[test]
    fn encoding_from_header_maps_known() {
        assert_eq!(encoding_from_header(Some("gzip")), decompress::Encoding::Gzip);
        assert_eq!(encoding_from_header(Some("zstd")), decompress::Encoding::Zstd);
        assert_eq!(encoding_from_header(Some("br")), decompress::Encoding::Identity);
        assert_eq!(encoding_from_header(None), decompress::Encoding::Identity);
    }

    // --- decrypt policy: providers-only vs decrypt-all -----------------------

    fn policy(decrypt_all: bool) -> DecryptPolicy {
        DecryptPolicy { decrypt_all }
    }

    #[test]
    fn policy_default_decrypts_providers_only() {
        // env unset == providers-only: the shim-shared base-map API hosts are
        // decrypted; unrelated hosts are not. A consumer host like claude.ai is
        // decrypted only when a whitelist rule maps it to a provider (next test).
        let p = policy(false);
        let empty = whitelist::Whitelist::empty();
        assert!(p.should_decrypt("api.anthropic.com", &empty));
        assert!(p.should_decrypt("api.openai.com", &empty));
        assert!(p.should_decrypt("generativelanguage.googleapis.com", &empty));
        assert!(!p.should_decrypt("claude.ai", &empty), "no rule -> not a base-map provider");
        assert!(!p.should_decrypt("example.com", &empty));
    }

    #[test]
    fn policy_decrypt_all_decrypts_every_host() {
        let p = policy(true);
        let empty = whitelist::Whitelist::empty();
        assert!(p.should_decrypt("api.anthropic.com", &empty), "providers still decrypted");
        assert!(p.should_decrypt("claude.ai", &empty));
        assert!(p.should_decrypt("browser-intake-datadoghq.com", &empty));
        assert!(p.should_decrypt("anything.example", &empty));
    }

    #[test]
    fn policy_has_no_provider_opt_out() {
        // A metered provider is decrypted under BOTH modes, and the policy carries
        // no state that could exempt one: a host cannot be configured out of
        // metering here. Guards the no-opt-out invariant on `DecryptPolicy`.
        let empty = whitelist::Whitelist::empty();
        for host in ["api.anthropic.com", "api.openai.com", "generativelanguage.googleapis.com"] {
            assert!(policy(false).should_decrypt(host, &empty), "{} under providers-only", host);
            assert!(policy(true).should_decrypt(host, &empty), "{} under decrypt-all", host);
        }
        // A whitelist-declared consumer host is likewise always decrypted.
        let cwl = wl("meter", vec![wl_rule_est("claude.ai", "claude")]);
        assert!(policy(false).should_decrypt("claude.ai", &cwl));
        assert!(policy(true).should_decrypt("claude.ai", &cwl));
    }

    // --- A1: a whitelist rule resolves a consumer host to a provider --------

    #[test]
    fn proxy_provider_maps_claude_ai_to_anthropic() {
        // Config-driven: a rule with provider="anthropic" resolves claude.ai (the
        // Electron chat panel) to Anthropic so it flows through the usage/model
        // parser + estimate fallback. The shim-shared provider_for_host is
        // unchanged and still doesn't know claude.ai.
        let cwl = wl("meter", vec![wl_rule_est("claude.ai", "claude")]);
        assert_eq!(proxy_provider_for_host("claude.ai", &cwl), Some(Provider::Anthropic));
        assert_eq!(proxy_provider_for_host("Claude.AI", &cwl), Some(Provider::Anthropic));
        // Base-map providers resolve regardless of the whitelist.
        assert_eq!(
            proxy_provider_for_host("api.anthropic.com", &cwl),
            Some(Provider::Anthropic)
        );
        assert_eq!(proxy_provider_for_host("example.com", &cwl), None);
        // Without a provider rule, claude.ai is unknown (base map doesn't know it).
        let empty = whitelist::Whitelist::empty();
        assert_eq!(proxy_provider_for_host("claude.ai", &empty), None);
        // The shared map (compiled into the shim too) still doesn't know claude.ai.
        assert_eq!(provider_for_host("claude.ai"), None);
    }

    #[test]
    fn policy_decrypts_claude_ai_with_provider_rule_without_decrypt_all() {
        // With a whitelist rule mapping claude.ai to a provider, it is decrypted
        // without decrypt-all so the chat panel is metered. Without such a rule,
        // claude.ai is not auto-decrypted.
        let cwl = wl("meter", vec![wl_rule_est("claude.ai", "claude")]);
        assert!(policy(false).should_decrypt("claude.ai", &cwl));
        assert!(!policy(false).should_decrypt("claude.ai", &whitelist::Whitelist::empty()));
    }

    // --- A1 + B1: emit reports estimated (chat) and measured (code) usage ----

    /// A minimal `Shared` wired to a channel and a specific `Whitelist`, so
    /// `emit`'s report gate + per-host tokenizer can be exercised. No traffic log;
    /// a real CA + upstream configs satisfy the struct.
    fn test_shared_wl(tx: SyncSender<Event>, wl: whitelist::Whitelist) -> Arc<Shared> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let roots = Arc::new(RootCertStore::empty());
        Arc::new(Shared {
            ca: ca::Ca::generate(),
            upstream_h2: build_upstream_config(roots.clone(), &[b"h2"]),
            upstream_h1: build_upstream_config(roots, &[b"http/1.1"]),
            policy: DecryptPolicy { decrypt_all: false },
            traffic: TrafficLog { file: None },
            tx: Some(tx),
            conn_seq: std::sync::atomic::AtomicU64::new(1),
            whitelist: wl,
        })
    }

    /// The meter-all default (empty whitelist): `default_action="meter"`, no
    /// rules, so every provider request is whitelisted — the pre-whitelist proxy
    /// behavior these existing tests assert.
    fn test_shared(tx: SyncSender<Event>) -> Arc<Shared> {
        test_shared_wl(tx, whitelist::Whitelist::empty())
    }

    #[test]
    fn emit_estimates_tokens_for_claude_ai_when_usage_absent() {
        // The consumer-chat completion case: a `POST .../completion` on claude.ai
        // with NO measured usage. emit must pair RequestStarted + RequestCompleted
        // (so the sinks report) with nonzero, model-aware ESTIMATES and
        // tokens_measured=false. Model comes from the request body (the ack
        // response carries none).
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, claude_est_wl());
        let model = "claude-opus-4-8";
        let fin = body_scan::Finalized::default(); // tokens_in/out None -> usage absent
        let req_body =
            br#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hello there, how are you today?"}]}"#;
        let resp_text_bytes = 512u64;
        let req_model = model_from_request(Provider::Anthropic, req_body, None);
        emit(
            &shared, 7, Some(Provider::Anthropic), "claude.ai", now_ms(), Some(200),
            600, 1500, resp_text_bytes, &fin, req_model, req_body,
            Some("POST"), Some("/api/organizations/o/chat_conversations/c/completion"),
        );

        match rx.try_recv().expect("RequestStarted emitted") {
            Event::RequestStarted { host, whitelisted, .. } => {
                assert_eq!(host, "claude.ai");
                assert!(whitelisted, "decrypted provider request is whitelisted");
            }
            e => panic!("expected RequestStarted, got {e:?}"),
        }
        match rx.try_recv().expect("RequestCompleted emitted") {
            Event::RequestCompleted {
                tokens_in, tokens_out, tokens_measured, model: m, ..
            } => {
                assert!(!tokens_measured, "chat panel is estimated, not measured");
                assert_eq!(m.as_deref(), Some(model));
                assert_eq!(
                    tokens_in,
                    tokens::estimate_tokens(req_body.len() as u64, "claude", Some(model))
                );
                assert_eq!(
                    tokens_out,
                    tokens::estimate_tokens(resp_text_bytes, "claude", Some(model))
                );
                assert!(
                    tokens_in > 0 && tokens_out > 0,
                    "chat panel still reports nonzero estimates (not gated out)"
                );
            }
            e => panic!("expected RequestCompleted, got {e:?}"),
        }
    }

    #[test]
    fn emit_prefers_measured_usage_when_present() {
        // The code/API path: provider-reported usage is used verbatim and marked
        // measured; the response byte count does NOT drive the estimate.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared(tx);
        let fin = body_scan::Finalized {
            tokens_in: Some(1234),
            tokens_out: Some(56),
            model: Some("claude-sonnet-5".to_string()),
            ..Default::default()
        };
        emit(
            &shared, 8, Some(Provider::Anthropic), "api.anthropic.com", now_ms(), Some(200),
            999, 999, 9_999_999, &fin, None, b"{}", None, None,
        );
        let _ = rx.try_recv().expect("RequestStarted emitted");
        match rx.try_recv().expect("RequestCompleted emitted") {
            Event::RequestCompleted { tokens_in, tokens_out, tokens_measured, .. } => {
                assert!(tokens_measured);
                assert_eq!(tokens_in, 1234, "measured used verbatim, not estimated");
                assert_eq!(tokens_out, 56);
            }
            e => panic!("expected RequestCompleted, got {e:?}"),
        }
    }

    #[test]
    fn emit_does_not_estimate_api_host_without_usage() {
        // The count_tokens / telemetry case: a real-API host with a model but NO
        // usage is NOT a completion. emit must NOT estimate — tokens stay 0 so the
        // reporter's both-token-counts-zero guard drops it (no report flood).
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared(tx);
        let fin = body_scan::Finalized {
            model: Some("claude-sonnet-5".to_string()),
            ..Default::default() // no measured usage
        };
        let req_body =
            br#"{"model":"claude-sonnet-5","messages":[{"role":"user","content":"count me"}]}"#;
        emit(
            &shared, 9, Some(Provider::Anthropic), "api.anthropic.com", now_ms(), Some(200),
            2000, 506, 400, &fin, None, req_body, Some("POST"), Some("/v1/messages/count_tokens"),
        );
        let _ = rx.try_recv().expect("RequestStarted emitted");
        match rx.try_recv().expect("RequestCompleted emitted") {
            Event::RequestCompleted { tokens_in, tokens_out, tokens_measured, .. } => {
                assert!(!tokens_measured);
                assert_eq!(tokens_in, 0, "api host without usage is not estimated");
                assert_eq!(tokens_out, 0, "api host without usage is not estimated");
            }
            e => panic!("expected RequestCompleted, got {e:?}"),
        }
    }

    #[test]
    fn emit_drops_claude_ai_history_load_get() {
        // The phantom-report scenario: switching conversations issues a
        // `GET .../chat_conversations/{id}?...` that returns the whole stored
        // conversation as JSON — a model, no usage, and a large body. It looks
        // exactly like a completion by body alone, so ONLY the decoded method/path
        // tells them apart. A GET must NOT be estimated even with a big response;
        // it stays 0 and is dropped by the both-token-counts-zero guard.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, claude_est_wl());
        let fin = body_scan::Finalized {
            model: Some("claude-haiku-4-5".to_string()),
            ..Default::default() // stored conversation carries a model, no usage
        };
        emit(
            &shared, 10, Some(Provider::Anthropic), "claude.ai", now_ms(), Some(200),
            0, 4000, 12000, &fin, None, b"", Some("GET"),
            Some("/api/organizations/o/chat_conversations/c?tree=True&rendering_mode=messages"),
        );
        let _ = rx.try_recv().expect("RequestStarted emitted");
        match rx.try_recv().expect("RequestCompleted emitted") {
            Event::RequestCompleted { tokens_in, tokens_out, tokens_measured, .. } => {
                assert!(!tokens_measured);
                assert_eq!(tokens_in, 0, "history-load GET is not a completion");
                assert_eq!(tokens_out, 0, "history-load GET must not be byte-estimated");
            }
            e => panic!("expected RequestCompleted, got {e:?}"),
        }
    }

    #[test]
    fn emit_output_estimate_uses_generated_text_bytes_not_envelope() {
        // The chat panel streams SSE: the whole decompressed body (resp_text_bytes)
        // is dominated by `event:`/`data:`/JSON framing and overcounts output ~10x.
        // When the scanner summed the generated-text bytes (output_text_bytes),
        // emit must estimate output from THAT, not from the envelope.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, claude_est_wl());
        let model = "claude-haiku-4-5";
        let generated_text_bytes = 120u64; // ~a short reply
        let whole_envelope_bytes = 100_000u64; // SSE framing bloat
        let fin = body_scan::Finalized {
            model: Some(model.to_string()),
            output_text_bytes: Some(generated_text_bytes),
            ..Default::default() // usage absent -> estimated
        };
        emit(
            &shared, 11, Some(Provider::Anthropic), "claude.ai", now_ms(), Some(200),
            0, 3000, whole_envelope_bytes, &fin, None, b"", Some("POST"),
            Some("/api/organizations/o/chat_conversations/c/completion"),
        );
        let _ = rx.try_recv().expect("RequestStarted emitted");
        match rx.try_recv().expect("RequestCompleted emitted") {
            Event::RequestCompleted { tokens_out, tokens_measured, .. } => {
                assert!(!tokens_measured);
                assert_eq!(
                    tokens_out,
                    tokens::estimate_tokens(generated_text_bytes, "claude", Some(model)),
                    "output estimated from generated text, not the SSE envelope"
                );
                assert_ne!(
                    tokens_out,
                    tokens::estimate_tokens(whole_envelope_bytes, "claude", Some(model)),
                    "must not use the whole-envelope byte count"
                );
            }
            e => panic!("expected RequestCompleted, got {e:?}"),
        }
    }

    // --- E2: operator-controlled report gate + per-host tokenizer ------------

    /// Build a `Whitelist` with a `default_action` and a single-rule set.
    fn wl(default_action: &str, rules: Vec<whitelist::WhitelistRule>) -> whitelist::Whitelist {
        whitelist::Whitelist { version: 1, default_action: default_action.into(), rules }
    }

    fn wl_rule(host: &str, tokenizer: &str) -> whitelist::WhitelistRule {
        whitelist::WhitelistRule {
            host: host.into(),
            path_prefix: "/".into(),
            inject_headers: false,
            tokenizer: tokenizer.into(),
            estimate_unmeasured: false,
            completion_path: String::new(),
            provider: String::new(),
        }
    }

    /// A consumer-wire rule that opts in to byte-estimation of unmeasured
    /// completions (a `POST` to `*/completion`), mirroring the claude_desktop app
    /// profile's whitelist_contribution.
    fn wl_rule_est(host: &str, tokenizer: &str) -> whitelist::WhitelistRule {
        whitelist::WhitelistRule {
            host: host.into(),
            path_prefix: "/".into(),
            inject_headers: false,
            tokenizer: tokenizer.into(),
            estimate_unmeasured: true,
            completion_path: "*/completion".into(),
            provider: "anthropic".into(),
        }
    }

    /// The claude.ai estimate whitelist the app profile ships (meter-all base +
    /// the consumer-wire estimate rule), for the estimate-path emit tests.
    fn claude_est_wl() -> whitelist::Whitelist {
        wl("meter", vec![wl_rule_est("claude.ai", "claude")])
    }

    #[test]
    fn emit_whitelist_ignore_gates_matched_and_unmatched_hosts() {
        // default_action="ignore" + a rule for claude.ai only: the matched host
        // reports (whitelisted=true); an unmatched host is flagged NOT whitelisted
        // so the reporter's whitelist filter drops it.
        let fin = body_scan::Finalized {
            tokens_in: Some(10),
            tokens_out: Some(20),
            model: Some("claude-opus-4-8".into()),
            ..Default::default()
        };

        // Matched host -> whitelisted.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, wl("ignore", vec![wl_rule("claude.ai", "claude")]));
        emit(
            &shared, 20, Some(Provider::Anthropic), "claude.ai", now_ms(), Some(200),
            1, 1, 0, &fin, None, b"{}", Some("POST"),
            Some("/api/organizations/o/chat_conversations/c/completion"),
        );
        match rx.try_recv().expect("RequestStarted emitted") {
            Event::RequestStarted { whitelisted, .. } => {
                assert!(whitelisted, "matched rule under default_action=ignore is whitelisted")
            }
            e => panic!("expected RequestStarted, got {e:?}"),
        }

        // Unmatched host under default_action="ignore" -> NOT whitelisted.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, wl("ignore", vec![wl_rule("claude.ai", "claude")]));
        emit(
            &shared, 21, Some(Provider::Anthropic), "api.anthropic.com", now_ms(), Some(200),
            1, 1, 0, &fin, None, b"{}", Some("POST"), Some("/v1/messages"),
        );
        match rx.try_recv().expect("RequestStarted emitted") {
            Event::RequestStarted { whitelisted, .. } => {
                assert!(!whitelisted, "unmatched host under default_action=ignore is dropped")
            }
            e => panic!("expected RequestStarted, got {e:?}"),
        }
    }

    #[test]
    fn emit_whitelist_meter_default_reports_unmatched_host() {
        // Meter-all preserved: default_action="meter" with no matching rule still
        // reports (whitelisted=true) — the empty/unset-whitelist contract that
        // never silently stops metering.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, wl("meter", vec![wl_rule("example.com", "approx")]));
        let fin = body_scan::Finalized {
            tokens_in: Some(5),
            tokens_out: Some(7),
            ..Default::default()
        };
        emit(
            &shared, 22, Some(Provider::Anthropic), "api.anthropic.com", now_ms(), Some(200),
            1, 1, 0, &fin, None, b"{}", Some("POST"), Some("/v1/messages"),
        );
        match rx.try_recv().expect("RequestStarted emitted") {
            Event::RequestStarted { whitelisted, .. } => {
                assert!(whitelisted, "meter default reports the unmatched host")
            }
            e => panic!("expected RequestStarted, got {e:?}"),
        }
    }

    #[test]
    fn emit_uses_per_host_tokenizer_from_matched_rule() {
        // A matched rule's tokenizer overrides the provider default for the
        // byte-ratio estimate. claude.ai's provider default is "claude" (2.7
        // bytes/token for opus-4-8); a rule pinning "cl100k" (4.0) must make BOTH
        // the input and output estimates use cl100k instead.
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let shared = test_shared_wl(tx, wl("ignore", vec![wl_rule_est("claude.ai", "cl100k")]));
        let model = "claude-opus-4-8";
        let out_bytes = 400u64;
        let fin = body_scan::Finalized {
            model: Some(model.into()),
            output_text_bytes: Some(out_bytes),
            ..Default::default() // no measured usage -> estimated
        };
        let req_body =
            br#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hello there, how are you?"}]}"#;
        emit(
            &shared, 23, Some(Provider::Anthropic), "claude.ai", now_ms(), Some(200),
            1, 1, 0, &fin, None, req_body, Some("POST"),
            Some("/api/organizations/o/chat_conversations/c/completion"),
        );
        let _ = rx.try_recv().expect("RequestStarted emitted");
        match rx.try_recv().expect("RequestCompleted emitted") {
            Event::RequestCompleted { tokens_in, tokens_out, tokens_measured, .. } => {
                assert!(!tokens_measured);
                let m = Some(model);
                assert_eq!(
                    tokens_in,
                    tokens::estimate_tokens(plaintext_len(req_body), "cl100k", m),
                    "input estimated with the rule's cl100k tokenizer"
                );
                assert_eq!(
                    tokens_out,
                    tokens::estimate_tokens(out_bytes, "cl100k", m),
                    "output estimated with the rule's cl100k tokenizer"
                );
                assert_ne!(
                    tokens_out,
                    tokens::estimate_tokens(out_bytes, "claude", m),
                    "not the claude provider default tokenizer"
                );
            }
            e => panic!("expected RequestCompleted, got {e:?}"),
        }
    }

    #[test]
    fn env_flag_truthiness() {
        let key = "CLEARML_SNUG_TEST_FLAG_XYZ";
        for v in ["1", "true", "TRUE", "Yes", "on"] {
            std::env::set_var(key, v);
            assert!(env_flag(key), "{v} is truthy");
        }
        for v in ["0", "false", "no", "off", ""] {
            std::env::set_var(key, v);
            assert!(!env_flag(key), "{v} is falsy");
        }
        std::env::remove_var(key);
        assert!(!env_flag(key), "unset is falsy");
    }

    #[test]
    fn benign_disconnect_gating() {
        use std::io::{Error, ErrorKind};
        // The three peer-disconnect kinds observed as browser connection-pool
        // churn (preconnect cancel / keep-alive teardown): quieted to debug.
        for k in [
            ErrorKind::UnexpectedEof,
            ErrorKind::ConnectionReset,
            ErrorKind::BrokenPipe,
        ] {
            assert!(is_benign_disconnect(&Error::from(k)), "{k:?} is expected client churn");
        }
        // Everything else stays loud: connect failures that signal a broken
        // upstream, the `Other` kind used for the proxy's cert/config errors, and
        // disconnect kinds we have never observed on this path.
        for k in [
            ErrorKind::ConnectionRefused,
            ErrorKind::TimedOut,
            ErrorKind::PermissionDenied,
            ErrorKind::Other,
            ErrorKind::ConnectionAborted,
            ErrorKind::NotConnected,
        ] {
            assert!(!is_benign_disconnect(&Error::from(k)), "{k:?} is a genuine error");
        }
    }

    #[test]
    fn default_spki_path_sits_next_to_the_ca() {
        assert_eq!(
            default_spki_path("/tmp/certs/snug_proxy_ca.pem"),
            "/tmp/certs/snug_proxy_ca.spki"
        );
        // No directory component -> bare filename.
        assert_eq!(default_spki_path("ca.pem"), "snug_proxy_ca.spki");
    }

    // --- capture-all record construction / serialization --------------------

    #[test]
    fn capture_record_serializes_all_keys() {
        let rec = CaptureRecord {
            ts: 1_700_000_000_000,
            host: "api.anthropic.com",
            method: Some("POST"),
            path: Some("/v1/messages"),
            status: Some(200),
            tx: 1234,
            rx: 5678,
            ms: 42,
        };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(v["ts"], 1_700_000_000_000u64);
        assert_eq!(v["host"], "api.anthropic.com");
        assert_eq!(v["method"], "POST");
        assert_eq!(v["path"], "/v1/messages");
        assert_eq!(v["status"], 200);
        assert_eq!(v["tx"], 1234);
        assert_eq!(v["rx"], 5678);
        assert_eq!(v["ms"], 42);
    }

    #[test]
    fn capture_record_h2_unknowns_are_null() {
        // Over h2 `status` is unavailable (and method/path too, if HPACK decoding
        // gave up); absent values stay present as null so every log line has a
        // stable shape for the Python consumer.
        let rec = CaptureRecord {
            ts: 1,
            host: "claude.ai",
            method: None,
            path: None,
            status: None,
            tx: 10,
            rx: 20,
            ms: 5,
        };
        let s = serde_json::to_string(&rec).unwrap();
        assert!(s.contains("\"method\":null"));
        assert!(s.contains("\"path\":null"));
        assert!(s.contains("\"status\":null"));
        assert!(s.contains("\"host\":\"claude.ai\""));
    }

    #[test]
    fn traffic_log_appends_json_lines_per_request() {
        let dir = std::env::temp_dir().join(format!("snug_traffic_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("traffic.log");
        let log = TrafficLog {
            file: Some(Mutex::new(
                OpenOptions::new().create(true).append(true).open(&path).unwrap(),
            )),
        };
        log.record(&CaptureRecord {
            ts: 1, host: "a.com", method: Some("GET"), path: Some("/"),
            status: Some(204), tx: 1, rx: 2, ms: 3,
        });
        log.record(&CaptureRecord {
            ts: 2, host: "b.com", method: None, path: None, status: None, tx: 4, rx: 5, ms: 6,
        });
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "one JSON line per completed request");
        let l0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(l0["host"], "a.com");
        assert_eq!(l0["status"], 204);
        let l1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(l1["host"], "b.com");
        assert!(l1["method"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn traffic_log_disabled_is_a_noop() {
        // No file configured (neither env var set): recording must not panic.
        let log = TrafficLog { file: None };
        log.record(&CaptureRecord {
            ts: 1, host: "a.com", method: None, path: None, status: None, tx: 0, rx: 0, ms: 0,
        });
    }

    // --- H1Scan capture-only mode -------------------------------------------

    #[test]
    fn h1scan_capture_reads_status_without_usage_or_inflation() {
        // A gzipped, provider-shaped JSON body on the capture-only path is NOT
        // inflated or usage-parsed (the whole point of the light path), but the
        // status is still read from the head.
        let gz = gzip(ANTHROPIC_JSON);
        let mut resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\n\r\n",
            gz.len()
        )
        .into_bytes();
        resp.extend_from_slice(&gz);
        let mut scan = H1Scan::new_capture();
        scan.feed(&resp);
        let fin = scan.finalize();
        assert_eq!(scan.status(), Some(200), "status parsed in capture mode");
        assert_eq!(fin.tokens_in, None, "no usage parsed in capture mode");
        assert_eq!(fin.tokens_out, None);
    }

    #[test]
    fn chunk_scan_detects_terminal_chunk_bytewise() {
        // Fed one byte at a time (the worst case for a streaming framer), the
        // scanner must flip to complete exactly at the terminal 0-size chunk and
        // not before.
        let body = vec![b'x'; 100];
        let encoded = chunk_encode(&body); // pieces of 29 bytes + `0\r\n\r\n`
        let mut cs = ChunkScan::default();
        for (i, b) in encoded.iter().enumerate() {
            assert!(!cs.is_complete(), "not complete before the end (byte {i})");
            cs.feed(std::slice::from_ref(b));
        }
        assert!(cs.is_complete(), "complete after the terminal chunk");
    }

    #[test]
    fn chunk_scan_handles_trailers() {
        // A chunked body ending with a trailer header before the blank line.
        let mut data = Vec::new();
        data.extend_from_slice(b"4\r\nabcd\r\n");
        data.extend_from_slice(b"0\r\nX-Trailer: v\r\n\r\n");
        let mut cs = ChunkScan::default();
        cs.feed(&data);
        assert!(cs.is_complete(), "trailer then blank line completes the body");
    }

    #[test]
    fn h1scan_capture_detects_chunked_completion() {
        // Non-provider (e.g. text/html) chunked body: capture mode must still see
        // the terminal chunk so the keep-alive loop advances to the next request.
        let body = b"<html>hello world, this body spans a couple of chunks</html>";
        let mut resp =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        resp.extend_from_slice(&chunk_encode(body));
        let mut scan = H1Scan::new_capture();
        feed_bytewise(&mut scan, &resp);
        assert!(scan.is_complete(), "terminal chunk detected in capture mode");
        assert_eq!(scan.status(), Some(200));
    }

    // --- relay_h2 full-duplex streaming (no deadlock) -----------------------

    /// A connected loopback TCP pair (both ends `TCP_NODELAY`).
    fn tcp_pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let a = TcpStream::connect(addr).unwrap();
        let (b, _) = l.accept().unwrap();
        a.set_nodelay(true).unwrap();
        b.set_nodelay(true).unwrap();
        (a, b)
    }

    /// The streaming regression guard: a decrypted h2 connection must relay the
    /// response as it arrives, full-duplex, even while the request upload is
    /// backpressured. `relay_h2` shares each TLS leg's rustls connection between
    /// the two relay directions; if a blocking socket write is held under that
    /// connection lock (as it was when the response was withheld until the
    /// request drained), the response direction can never take the lock to
    /// forward the stream and the client hangs. Here the fake upstream NEVER
    /// reads the request — so the proxy's upstream write side blocks — yet a
    /// small streamed response chunk must still reach the client. On the buggy
    /// (lock-held-across-write) relay this deadlocks and the assert fails; the
    /// split conn/write locks keep it flowing.
    #[test]
    fn relay_h2_streams_response_while_request_upload_backpressured() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};

        let _ = rustls::crypto::ring::default_provider().install_default();

        // One CA; a "localhost" leaf reused by the proxy's client-facing side and
        // the fake upstream. Both TLS clients trust the CA.
        let ca = ca::Ca::generate();
        let leaf = ca.leaf_for("localhost");
        let ca_der = leaf.cert_chain[1].clone();

        let server_cfg = {
            let mut c = ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(leaf.cert_chain.clone(), leaf.key.clone_key())
                .unwrap();
            c.alpn_protocols = vec![b"h2".to_vec()];
            Arc::new(c)
        };
        let client_cfg = {
            let mut roots = RootCertStore::empty();
            roots.add(ca_der).unwrap();
            let mut c = ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth();
            c.alpn_protocols = vec![b"h2".to_vec()];
            Arc::new(c)
        };
        let sni = || ServerName::try_from("localhost".to_string()).unwrap();

        // Handshake the client leg: real client <-> proxy's ServerConnection.
        let (mut rc_tcp, ps_tcp) = tcp_pair();
        let scfg = server_cfg.clone();
        let hs_server = std::thread::spawn(move || {
            let mut ps_tcp = ps_tcp;
            let mut sconn = ServerConnection::new(scfg).unwrap();
            while sconn.is_handshaking() {
                sconn.complete_io(&mut ps_tcp).unwrap();
            }
            (sconn, ps_tcp)
        });
        let mut real_client = ClientConnection::new(client_cfg.clone(), sni()).unwrap();
        while real_client.is_handshaking() {
            real_client.complete_io(&mut rc_tcp).unwrap();
        }
        let (sconn, ps_tcp) = hs_server.join().unwrap();

        // Handshake the upstream leg: proxy's ClientConnection <-> fake upstream.
        let (pc_tcp, fu_tcp) = tcp_pair();
        let scfg2 = server_cfg.clone();
        let hs_up = std::thread::spawn(move || {
            let mut fu_tcp = fu_tcp;
            let mut fu = ServerConnection::new(scfg2).unwrap();
            while fu.is_handshaking() {
                fu.complete_io(&mut fu_tcp).unwrap();
            }
            (fu, fu_tcp)
        });
        let mut cconn = ClientConnection::new(client_cfg.clone(), sni()).unwrap();
        let mut pc_tcp = pc_tcp;
        while cconn.is_handshaking() {
            cconn.complete_io(&mut pc_tcp).unwrap();
        }
        let (mut fu, mut fu_tcp) = hs_up.join().unwrap();

        // Minimal Shared on the non-provider capture path: no reporter, no log.
        let roots = Arc::new(RootCertStore::empty());
        let shared = Arc::new(Shared {
            ca,
            upstream_h2: build_upstream_config(roots.clone(), &[b"h2"]),
            upstream_h1: build_upstream_config(roots, &[b"http/1.1"]),
            policy: DecryptPolicy { decrypt_all: true },
            traffic: TrafficLog { file: None },
            tx: None,
            conn_seq: std::sync::atomic::AtomicU64::new(1),
            whitelist: whitelist::Whitelist::empty(),
        });

        let relay = std::thread::spawn(move || {
            let _ = relay_h2(1, None, "localhost".to_string(), sconn, ps_tcp, cconn, pc_tcp, shared);
        });

        let done = Arc::new(AtomicBool::new(false));
        const MARKER: &[u8] = b"HELLO_STREAM_CHUNK";

        // Fake upstream: NEVER reads the request (so the proxy's upstream write
        // side backpressures and blocks), then — once the upload has wedged the
        // request direction — streams a small chunk that must still reach the
        // client. Holds the connection open until the test signals done.
        let done_up = done.clone();
        let upstream = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            fu.writer().write_all(MARKER).unwrap();
            while fu.wants_write() {
                if fu.write_tls(&mut fu_tcp).unwrap() == 0 {
                    break;
                }
            }
            fu_tcp.flush().ok();
            while !done_up.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
            }
        });

        // Real client, full-duplex on one thread via non-blocking I/O: keep
        // pushing a large request (to backpressure the upstream leg) while
        // reading the response.
        rc_tcp.set_nonblocking(true).unwrap();
        let request = vec![b'q'; 8 * 1024 * 1024];
        let mut req_off = 0usize;
        let mut received: Vec<u8> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut got = false;
        while Instant::now() < deadline && !got {
            if req_off < request.len() {
                let end = (req_off + 256 * 1024).min(request.len());
                if let Ok(n) = real_client.writer().write(&request[req_off..end]) {
                    req_off += n;
                }
            }
            while real_client.wants_write() {
                match real_client.write_tls(&mut rc_tcp) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
            match real_client.read_tls(&mut rc_tcp) {
                Ok(0) => {}
                Ok(_) => {
                    if real_client.process_new_packets().is_err() {
                        break;
                    }
                    let mut buf = [0u8; 65536];
                    loop {
                        match real_client.reader().read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => received.extend_from_slice(&buf[..n]),
                            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(_) => break,
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => {}
            }
            got = received.windows(MARKER.len()).any(|w| w == MARKER);
            std::thread::sleep(Duration::from_millis(5));
        }

        // Signal shutdown and drop the client socket so the relay unwinds.
        done.store(true, Ordering::Relaxed);
        drop(rc_tcp);

        // Assert BEFORE joining: on the buggy (deadlocking) relay the join would
        // hang, so fail fast instead.
        assert!(
            got,
            "streamed upstream response reached the client while the request upload was backpressured (no full-duplex deadlock)"
        );

        // Clean shutdown: joining the upstream drops its socket, unblocking the
        // proxy's wedged upstream write so the relay threads exit.
        let _ = upstream.join();
        let _ = relay.join();
    }

    // --- HPACK request header decode (loona-hpack) --------------------------

    /// Find a pseudo-header value (lossy String) in a decoded header list.
    fn find_header(headers: &[(Vec<u8>, Vec<u8>)], name: &[u8]) -> Option<String> {
        headers
            .iter()
            .find(|(n, _)| n.as_slice() == name)
            .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
    }

    fn h2_frame_bytes(ftype: u8, flags: u8, stream_id: u32, payload: &[u8]) -> Vec<u8> {
        let len = payload.len();
        let mut f = vec![(len >> 16) as u8, (len >> 8) as u8, len as u8, ftype, flags];
        f.extend_from_slice(&stream_id.to_be_bytes());
        f.extend_from_slice(payload);
        f
    }

    fn h2_headers_frame(stream_id: u32, block: &[u8], end_headers: bool, end_stream: bool) -> Vec<u8> {
        let mut flags = 0u8;
        if end_headers {
            flags |= 0x4; // END_HEADERS
        }
        if end_stream {
            flags |= 0x1; // END_STREAM
        }
        h2_frame_bytes(h2::FRAME_HEADERS, flags, stream_id, block)
    }

    fn h2_continuation_frame(stream_id: u32, block: &[u8], end_headers: bool) -> Vec<u8> {
        let flags = if end_headers { 0x4 } else { 0 };
        h2_frame_bytes(h2::FRAME_CONTINUATION, flags, stream_id, block)
    }

    /// Drive the request-side HPACK pipeline exactly as t1 does
    /// (FrameParser -> HeaderBlockAssembler -> Decoder) and return the last
    /// stream's recovered (method, path).
    fn run_hpack_pipeline(wire: &[u8]) -> (Option<String>, Option<String>) {
        let mut fp = h2::FrameParser::new_client();
        let mut asm = h2::HeaderBlockAssembler::new();
        let mut decoder = loona_hpack::Decoder::new();
        let mut method = None;
        let mut path = None;
        for f in fp.feed(wire) {
            if let Some(h2::HeaderBlock::Complete(_sid, block)) = asm.feed(&f) {
                let headers = decoder.decode(&block).expect("decode");
                for (name, value) in &headers {
                    if name == b":method" {
                        method = Some(String::from_utf8_lossy(value).into_owned());
                    } else if name == b":path" {
                        path = Some(String::from_utf8_lossy(value).into_owned());
                    }
                }
            }
        }
        (method, path)
    }

    #[test]
    fn hpack_decodes_method_and_path_rfc_c31() {
        // RFC 7541 C.3.1: 82 86 84 41 0f "www.example.com" decodes to
        // :method GET, :scheme http, :path /, :authority www.example.com.
        let mut block = vec![0x82, 0x86, 0x84, 0x41, 0x0f];
        block.extend_from_slice(b"www.example.com");
        let mut d = loona_hpack::Decoder::new();
        let headers = d.decode(&block).expect("decode");
        assert_eq!(find_header(&headers, b":method").as_deref(), Some("GET"));
        assert_eq!(find_header(&headers, b":path").as_deref(), Some("/"));
        assert_eq!(
            find_header(&headers, b":authority").as_deref(),
            Some("www.example.com")
        );
    }

    #[test]
    fn hpack_pipeline_single_headers_frame() {
        // Encode a realistic request header set, wrap it in one HEADERS frame with
        // END_HEADERS, and run it through the same pieces t1 uses.
        let mut enc = loona_hpack::Encoder::new();
        let block = enc.encode(vec![
            (b":method".as_slice(), b"POST".as_slice()),
            (b":scheme", b"https"),
            (b":authority", b"api.anthropic.com"),
            (b":path", b"/v1/messages"),
        ]);
        let mut wire = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
        wire.extend_from_slice(&h2_headers_frame(1, &block, true, true));
        let (method, path) = run_hpack_pipeline(&wire);
        assert_eq!(method.as_deref(), Some("POST"));
        assert_eq!(path.as_deref(), Some("/v1/messages"));
    }

    #[test]
    fn hpack_pipeline_across_continuation() {
        // The block is split across a HEADERS frame (no END_HEADERS) and a
        // CONTINUATION frame (END_HEADERS) — the reassembly path that most risks a
        // dynamic-table desync. HPACK decodes only the reassembled whole.
        let mut enc = loona_hpack::Encoder::new();
        let block = enc.encode(vec![
            (b":method".as_slice(), b"GET".as_slice()),
            (b":scheme", b"https"),
            (b":authority", b"example.com"),
            (b":path", b"/a/b/c"),
        ]);
        let mid = block.len() / 2;
        let mut wire = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
        wire.extend_from_slice(&h2_headers_frame(1, &block[..mid], false, false));
        wire.extend_from_slice(&h2_continuation_frame(1, &block[mid..], true));
        let (method, path) = run_hpack_pipeline(&wire);
        assert_eq!(method.as_deref(), Some("GET"));
        assert_eq!(path.as_deref(), Some("/a/b/c"));
    }

    #[test]
    fn hpack_dynamic_table_stays_synced_across_requests() {
        // Two sequential request header blocks on ONE connection share one
        // decoder's cumulative dynamic table; the second must still decode
        // correctly, proving blocks are fed in order exactly once.
        let mut enc = loona_hpack::Encoder::new();
        let mut wire = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
        let b1 = enc.encode(vec![
            (b":method".as_slice(), b"POST".as_slice()),
            (b":path", b"/v1/messages"),
            (b"x-custom", b"first"),
        ]);
        wire.extend_from_slice(&h2_headers_frame(1, &b1, true, true));
        let b2 = enc.encode(vec![
            (b":method".as_slice(), b"GET".as_slice()),
            (b":path", b"/v1/models"),
            (b"x-custom", b"second"),
        ]);
        wire.extend_from_slice(&h2_headers_frame(3, &b2, true, true));
        let (method, path) = run_hpack_pipeline(&wire);
        // run_hpack_pipeline returns the last stream's values.
        assert_eq!(method.as_deref(), Some("GET"));
        assert_eq!(path.as_deref(), Some("/v1/models"));
    }
}
