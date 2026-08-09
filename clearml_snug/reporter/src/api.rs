//! Minimal ClearML backend API client for the in-process reporter.
//!
//! Covers the slice of `clearml_agent.backend_api.session.Session` the reporter
//! needs: `auth.login` → JWT (HTTP Basic when access/secret are present, else
//! Bearer-renewal with the current token — the token-only path, matching the
//! agent's own `_do_refresh_token`), lazy + proactive token refresh, and the
//! authenticated `events.add_batch` POST. TLS is rustls + ring: the default
//! agent trusts the webpki-roots bundle; a custom CA bundle and
//! `verify_certificate=false` are honored via an explicit rustls ClientConfig.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde_json::Value;

use crate::descriptor::Descriptor;

/// Fallback API version used for the URL path until the JWT tells us the
/// server's negotiated version (the Python client defaults to "2.1").
const DEFAULT_API_VERSION: &str = "2.1";

/// Refresh the token this many seconds before its `exp`. Mirrors the Python
/// client's `token_expiration_threshold_sec` default (12h).
const TOKEN_REFRESH_THRESHOLD_SEC: i64 = 12 * 60 * 60;

/// Max retries (beyond the first attempt) in **bounded** mode — control-plane
/// polling/writes and auth login. Kept small so the total backoff stays well
/// under the docker drain budget.
const HTTP_MAX_RETRIES: u32 = 4;

/// Backoff cap (ms) for bounded retries: small, so a bounded sequence finishes
/// quickly and well within the exit drain budget.
const BOUNDED_BACKOFF_MAX_MS: u64 = 2000;

/// Backoff cap (ms) for **forever** retries (data plane during a backend
/// outage). Mirrors the agent's `api.http.retries.backoff_max` (120s): a long
/// outage is probed gently rather than hammered.
const FOREVER_BACKOFF_MAX_MS: u64 = 120_000;

/// Granularity of the interruptible backoff sleep, so a forever-backoff (up to
/// 120s) still notices the exit drain signal within this bound.
const BACKOFF_CHUNK: Duration = Duration::from_millis(250);

/// How hard a request retries on a transient failure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RetryMode {
    /// Up to `HTTP_MAX_RETRIES`, then give up — control-plane polling/writes
    /// and auth login.
    Bounded,
    /// Retry until success — data-plane reporting during a backend outage, so
    /// events are never dropped. Always interruptible by the abort flag, so
    /// process exit stays bounded by the drain budget.
    Forever,
}

/// Floor on how early to proactively refresh: never let the renewal window
/// shrink below 5 minutes, so a transient outage near expiry can't lock us out
/// (token-only deployments can't recover from a fully-expired token).
const REFRESH_MIN_MARGIN_SEC: i64 = 300;

/// For an opaque token whose expiry we can't read (and the agent didn't supply
/// one in the descriptor), refresh on this fixed cadence as a best effort.
const OPAQUE_REFRESH_INTERVAL_SEC: i64 = 900;

pub struct ClearmlClient {
    agent: ureq::Agent,
    host: String, // api_server with any trailing slash trimmed
    access_key: String,
    secret_key: String,
    worker_id: String,
    api_version: String,
    token: Option<String>,
    token_exp: i64, // unix seconds; 0 = unknown (e.g. a pre-issued opaque token)
    /// Lifetime (seconds) of the current token, observed at issue/refresh; 0 if
    /// the expiry is unknown. Drives the "refresh at ~50% of TTL" schedule.
    token_ttl: i64,
    /// Unix seconds of the last successful token issue/refresh.
    last_refresh: i64,
    /// The reporter's drain/exit signal, checked between retries so a
    /// forever-retry bails promptly at shutdown. Defaults to a never-set flag;
    /// `start_reporter` replaces it with the real `drain` via `set_abort_signal`.
    abort: Arc<AtomicBool>,
}

impl ClearmlClient {
    pub fn from_descriptor(d: &Descriptor) -> Self {
        let mut builder = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(30))
            .timeout_write(Duration::from_secs(30));
        // Default (verify=true, no custom CA) uses ureq's built-in rustls +
        // webpki-roots. Build an explicit rustls config only for the custom
        // modes: verify_certificate=false or a custom CA bundle.
        if !d.verify_certificate || d.ca_cert_path.is_some() {
            match build_tls_config(d) {
                Ok(cfg) => builder = builder.tls_config(cfg),
                Err(e) => eprintln!(
                    "WARNING: SNUG reporter custom TLS config failed ({}); falling back to default verification",
                    e
                ),
            }
        }
        let agent = builder.build();
        let now = Self::now();
        // Prefer the JWT's own `exp`; fall back to an agent-supplied expiry for
        // opaque (non-JWT) tokens; else unknown (0).
        let token_exp = d
            .auth_token
            .as_deref()
            .and_then(jwt_exp)
            .or(d.token_expiry_sec)
            .unwrap_or(0);
        ClearmlClient {
            agent,
            host: d.api_server.trim_end_matches('/').to_string(),
            access_key: d.access_key.clone(),
            secret_key: d.secret_key.clone(),
            worker_id: d.worker_id.clone(),
            api_version: DEFAULT_API_VERSION.to_string(),
            token: d.auth_token.clone(),
            token_exp,
            token_ttl: if token_exp != 0 { (token_exp - now).max(0) } else { 0 },
            last_refresh: now,
            abort: Arc::new(AtomicBool::new(false)),
        }
    }

    fn now() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    /// Install the reporter's drain/exit flag so forever-retries and backoff
    /// sleeps bail promptly at shutdown. Called once from `start_reporter`.
    pub fn set_abort_signal(&mut self, flag: Arc<AtomicBool>) {
        self.abort = flag;
    }

    fn aborting(&self) -> bool {
        self.abort.load(Ordering::SeqCst)
    }

    /// Sleep up to `total`, waking every `BACKOFF_CHUNK` to check the abort
    /// flag, so a long forever-backoff still reacts to the exit drain signal
    /// within ~`BACKOFF_CHUNK` instead of blocking the bounded shutdown budget.
    fn sleep_interruptible(&self, total: Duration) {
        let mut slept = Duration::ZERO;
        while slept < total {
            if self.aborting() {
                return;
            }
            let step = BACKOFF_CHUNK.min(total - slept);
            std::thread::sleep(step);
            slept += step;
        }
    }

    /// `POST {host}/auth.login` → a fresh JWT (no version prefix on the auth
    /// endpoint, matching the Python client). Uses HTTP Basic (access:secret)
    /// when available, else **Bearer-renewal** with the current token (the
    /// token-only path most deployments use) — mirroring the agent's own
    /// `_do_refresh_token`. Bearer-renewal only works while the current token is
    /// still valid; a fully-expired token can't be renewed without a secret,
    /// which is why the reporter refreshes proactively and early.
    pub fn login(&mut self) -> Result<(), String> {
        let auth_header = if !self.access_key.is_empty() && !self.secret_key.is_empty() {
            let basic = base64::engine::general_purpose::STANDARD
                .encode(format!("{}:{}", self.access_key, self.secret_key));
            format!("Basic {}", basic)
        } else if let Some(tok) = &self.token {
            format!("Bearer {}", tok)
        } else {
            return Err("cannot refresh token: no access_key/secret_key and no token".to_string());
        };
        let url = format!("{}/auth.login", self.host);
        let resp = self
            .execute_with_retry(
                || {
                    self.agent
                        .post(&url)
                        .set("Authorization", &auth_header)
                        .call()
                },
                // Auth never retries forever: a refused login is more likely a
                // credential problem than a transient outage.
                RetryMode::Bounded,
            )
            .map_err(|e| format!("auth.login failed: {}", describe_err(e)))?;
        let txt = resp
            .into_string()
            .map_err(|e| format!("auth.login read failed: {}", e))?;
        let body: Value =
            serde_json::from_str(&txt).map_err(|e| format!("auth.login bad json: {}", e))?;
        let token = body
            .get("data")
            .and_then(|d| d.get("token"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| "auth.login: no data.token in response".to_string())?
            .to_string();
        let now = Self::now();
        self.token_exp = jwt_exp(&token).unwrap_or(0);
        self.token_ttl = if self.token_exp != 0 {
            (self.token_exp - now).max(0)
        } else {
            0
        };
        self.last_refresh = now;
        if let Some(v) = jwt_api_version(&token) {
            self.api_version = v;
        }
        self.token = Some(token);
        Ok(())
    }

    /// Return a valid bearer token, refreshing if missing or near expiry.
    fn ensure_token(&mut self) -> Result<String, String> {
        let need = match self.token {
            None => true,
            // Only refresh on a known expiry; an opaque token (exp unknown) is
            // used as-is until it 401s (handled by callers if needed).
            Some(_) => self.token_exp != 0 && (self.token_exp - Self::now() - TOKEN_REFRESH_THRESHOLD_SEC) <= 0,
        };
        if need {
            self.login()?;
        }
        Ok(self.token.clone().unwrap_or_default())
    }

    /// Proactively refresh the token if it's approaching expiry. Called on a
    /// timer by the reporter thread, independent of request traffic, so a
    /// long-running task with sparse LLM calls never lets its token lapse.
    /// Best-effort; logs and drops failures (the next tick retries).
    pub fn maybe_refresh_token(&mut self) {
        let due = refresh_due(
            self.token.is_some(),
            self.token_exp,
            self.token_ttl,
            self.last_refresh,
            Self::now(),
        );
        if due {
            if let Err(e) = self.login() {
                eprintln!("WARNING: SNUG proactive token refresh failed: {}", e);
            }
        }
    }

    /// `events.add_batch`: NDJSON body, `Content-Type: application/json-lines`.
    pub fn events_add_batch(&mut self, events: &[Value]) -> Result<Value, String> {
        if events.is_empty() {
            return Ok(Value::Null);
        }
        let token = self.ensure_token()?;
        let url = format!("{}/v{}/events.add_batch", self.host, self.api_version);
        let mut body = String::new();
        for ev in events {
            body.push_str(&serde_json::to_string(ev).map_err(|e| e.to_string())?);
            body.push('\n');
        }
        self.send(&url, &token, "application/json-lines", body.into_bytes(), RetryMode::Forever)
    }

    /// `routers.report_llm_usage`: a batch of external LLM usage events wrapped
    /// in the `{usage:[...]}` envelope. JSON body (not NDJSON). Data-plane
    /// durability: transient failures retry forever (interruptible by the drain
    /// signal), so usage is never silently dropped during a backend outage.
    pub fn report_llm_usage(&mut self, events: &[Value]) -> Result<Value, String> {
        if events.is_empty() {
            return Ok(Value::Null);
        }
        let token = self.ensure_token()?;
        let url = format!("{}/v{}/routers.report_llm_usage", self.host, self.api_version);
        let body = serde_json::to_vec(&serde_json::json!({ "usage": events }))
            .map_err(|e| e.to_string())?;
        self.send(&url, &token, "application/json", body, RetryMode::Forever)
    }

    /// Forward a batch of events to an external aggregator URL as NDJSON
    /// (`application/json-lines`, one event per line — matching the agent's
    /// `send_packet`). Plain POST, no auth, short 5s timeout, SINGLE attempt
    /// (deliberately not retried): one attempt keeps a slow/broken
    /// aggregator from monopolizing the reporter thread, preserving the original
    /// best-effort contract. Returns `Err` so the caller can log it; the sink
    /// drops the batch on failure.
    pub fn aggregator_post(&self, url: &str, body: &[u8]) -> Result<(), String> {
        let resp = self
            .agent
            .post(url)
            .timeout(Duration::from_secs(5))
            .set("Content-Type", "application/json-lines")
            .send_bytes(body);
        match resp {
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(code, _)) => Err(format!("HTTP {} from {}", code, url)),
            Err(e) => Err(format!("request to {} failed: {}", url, e)),
        }
    }

    /// Read the task's User Properties (hyperparams section `properties`) via
    /// `tasks.get_by_id`, flattened to a `name -> value` map. This is the field
    /// the ClearML UI exposes as editable WHILE THE TASK RUNS — the live control
    /// surface for the call-history mode (`_snug_call_history`) — written via
    /// `tasks.edit_hyper_params(section="properties")`.
    pub fn get_task_user_properties(
        &mut self,
        task_id: &str,
    ) -> Result<serde_json::Map<String, Value>, String> {
        let token = self.ensure_token()?;
        let url = format!("{}/v{}/tasks.get_by_id", self.host, self.api_version);
        let body = serde_json::to_vec(&serde_json::json!({
            "task": task_id,
            "only_fields": ["hyperparams.properties"],
        }))
        .map_err(|e| e.to_string())?;
        // Control plane: bounded. The poll loop retries on its own cadence.
        let resp = self.send(&url, &token, "application/json", body, RetryMode::Bounded)?;
        // Shape: data.task.hyperparams.properties.<name> = {section,name,value,type}.
        let mut out = serde_json::Map::new();
        if let Some(props) = resp
            .get("data")
            .and_then(|d| d.get("task"))
            .and_then(|t| t.get("hyperparams"))
            .and_then(|h| h.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (name, entry) in props {
                if let Some(v) = entry.get("value") {
                    out.insert(name.clone(), v.clone());
                }
            }
        }
        Ok(out)
    }

    /// Set one task User Property (hyperparams section `properties`) via
    /// `tasks.edit_hyper_params`. Used to auto-revert `_snug_call_history` from
    /// `dump` back to `collect` after a dump, so the operator doesn't have to
    /// switch back and the next `dump` is a fresh edge.
    pub fn set_task_user_property(
        &mut self,
        task_id: &str,
        name: &str,
        value: &str,
    ) -> Result<(), String> {
        let token = self.ensure_token()?;
        let url = format!("{}/v{}/tasks.edit_hyper_params", self.host, self.api_version);
        let body = serde_json::to_vec(&serde_json::json!({
            "task": task_id,
            "hyperparams": [{ "section": "properties", "name": name, "value": value }],
        }))
        .map_err(|e| e.to_string())?;
        // Control plane: bounded retry.
        self.send(&url, &token, "application/json", body, RetryMode::Bounded)?;
        Ok(())
    }

    /// Run a request builder, retrying transient failures (retryable HTTP
    /// statuses + transport errors) with bounded exponential backoff. The
    /// builder is re-invoked per attempt so it produces a fresh request.
    fn execute_with_retry<F>(&self, build: F, mode: RetryMode) -> Result<ureq::Response, ureq::Error>
    where
        F: Fn() -> Result<ureq::Response, ureq::Error>,
    {
        let mut attempt = 0u32;
        loop {
            match build() {
                Ok(r) => return Ok(r),
                Err(e) => {
                    let retryable = match &e {
                        ureq::Error::Status(code, _) => is_retryable_status(*code),
                        // Transport errors (connect refused, reset, timeout) are
                        // transient by nature.
                        ureq::Error::Transport(_) => true,
                    };
                    // `aborting()`: at shutdown we make only the single attempt
                    // just performed and stop — the shared exit drain budget
                    // must cover every still-queued event, so one event must not
                    // spend it all retrying.
                    if !should_retry(retryable, self.aborting(), mode, attempt) {
                        return Err(e);
                    }
                    attempt += 1;
                    let cap = match mode {
                        RetryMode::Forever => FOREVER_BACKOFF_MAX_MS,
                        RetryMode::Bounded => BOUNDED_BACKOFF_MAX_MS,
                    };
                    // Interruptible: a forever-backoff still bails within
                    // ~BACKOFF_CHUNK when the exit drain signal flips.
                    self.sleep_interruptible(retry_backoff(attempt, cap));
                }
            }
        }
    }

    /// Authenticated POST with one **401-recovery** retry: a rejected token
    /// (expired or revoked) forces a refresh + a single re-send. With key/secret
    /// this always recovers; token-only recovers only while the token is still
    /// Bearer-renewable (hence the proactive early refresh above).
    fn send(&mut self, url: &str, token: &str, content_type: &str, body: Vec<u8>, mode: RetryMode) -> Result<Value, String> {
        match self.send_inner(url, token, content_type, &body, mode) {
            SendOutcome::Ok(v) => Ok(v),
            SendOutcome::Unauthorized => {
                self.login()?;
                let fresh = self.token.clone().unwrap_or_default();
                match self.send_inner(url, &fresh, content_type, &body, mode) {
                    SendOutcome::Ok(v) => Ok(v),
                    SendOutcome::Unauthorized => {
                        Err(format!("HTTP 401 from {} even after token refresh", url))
                    }
                    SendOutcome::Err(e) => Err(e),
                }
            }
            SendOutcome::Err(e) => Err(e),
        }
    }

    /// One authenticated POST attempt. Distinguishes a 401 (token rejected →
    /// recoverable by refresh) from other outcomes so `send` can retry.
    fn send_inner(&self, url: &str, token: &str, content_type: &str, body: &[u8], mode: RetryMode) -> SendOutcome {
        let worker = if self.worker_id.is_empty() {
            "clearml-snug"
        } else {
            &self.worker_id
        };
        let resp = self.execute_with_retry(
            || {
                self.agent
                    .post(url)
                    .set("Authorization", &format!("Bearer {}", token))
                    .set("Content-Type", content_type)
                    .set("X-ClearML-Worker", worker)
                    .set("X-ClearML-Agent", "clearml-snug")
                    .send_bytes(body)
            },
            mode,
        );
        match resp {
            Ok(r) => {
                let txt = match r.into_string() {
                    Ok(t) => t,
                    Err(e) => return SendOutcome::Err(format!("response read failed: {}", e)),
                };
                if txt.is_empty() {
                    return SendOutcome::Ok(Value::Null);
                }
                match serde_json::from_str::<Value>(&txt) {
                    Ok(v) => SendOutcome::Ok(v),
                    Err(e) => {
                        SendOutcome::Err(format!("response not json ({}): {}", e, truncate(&txt, 200)))
                    }
                }
            }
            Err(ureq::Error::Status(401, _)) => SendOutcome::Unauthorized,
            Err(ureq::Error::Status(code, r)) => {
                let txt = r.into_string().unwrap_or_default();
                SendOutcome::Err(format!("HTTP {} from {}: {}", code, url, truncate(&txt, 300)))
            }
            Err(e) => SendOutcome::Err(format!("request to {} failed: {}", url, e)),
        }
    }
}

/// Outcome of a single authenticated POST, distinguishing a 401 (recoverable by
/// a token refresh) from other errors.
enum SendOutcome {
    Ok(Value),
    Unauthorized,
    Err(String),
}

fn describe_err(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, r) => {
            let txt = r.into_string().unwrap_or_default();
            format!("HTTP {}: {}", code, truncate(&txt, 300))
        }
        other => other.to_string(),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

/// Whether the token should be proactively refreshed now. Pure (no `self`) so
/// the scheduling logic is unit-testable without a live client. Refresh at the
/// halfway point of the token's lifetime with a 5-minute floor; for an opaque
/// token (unknown expiry) fall back to a fixed cadence since the last refresh.
fn refresh_due(has_token: bool, token_exp: i64, token_ttl: i64, last_refresh: i64, now: i64) -> bool {
    if !has_token {
        return true;
    }
    if token_exp != 0 {
        let margin = (token_ttl / 2).max(REFRESH_MIN_MARGIN_SEC);
        (token_exp - now) <= margin
    } else {
        (now - last_refresh) >= OPAQUE_REFRESH_INTERVAL_SEC
    }
}

/// Transient HTTP statuses worth retrying (matches the Python client's
/// retry-on-status set: rate-limit / header-too-large / gateway errors). A 500
/// is intentionally excluded — it can be a deterministic server error.
fn is_retryable_status(code: u16) -> bool {
    matches!(code, 429 | 431 | 502 | 503 | 504)
}

/// Exponential backoff: 200ms, 400, 800, … doubling per attempt, capped at
/// `cap_ms` (2s bounded / 120s forever). The shift is clamped to avoid overflow
/// at the high attempt counts a forever-retry can reach. Bounded so a
/// bounded-mode sequence stays comfortably under the drain budget; the forever
/// cap mirrors the agent's `backoff_max`.
fn retry_backoff(attempt: u32, cap_ms: u64) -> Duration {
    let shift = (attempt.saturating_sub(1)).min(20);
    let ms = 200u64.saturating_mul(1u64 << shift).min(cap_ms);
    Duration::from_millis(ms)
}

/// Whether to retry after a failed attempt. Pure (no `self`) so the policy is
/// unit-testable without ureq types or a live server. A non-retryable status or
/// a set abort flag stops immediately; otherwise `Forever` always retries while
/// `Bounded` retries only while under `HTTP_MAX_RETRIES`. `attempt` is the
/// number of retries already performed (0 on the first failure).
fn should_retry(retryable: bool, aborting: bool, mode: RetryMode, attempt: u32) -> bool {
    if !retryable || aborting {
        return false;
    }
    match mode {
        RetryMode::Forever => true,
        RetryMode::Bounded => attempt < HTTP_MAX_RETRIES,
    }
}

/// Decode a JWT's `exp` claim (seconds since epoch). No signature verification,
/// matching the Python client which decodes with `verify_signature=False`.
fn jwt_exp(token: &str) -> Option<i64> {
    jwt_claims(token)?.get("exp").and_then(|v| v.as_i64())
}

/// Decode a JWT's `api_version` claim, used to pin the URL version after login.
fn jwt_api_version(token: &str) -> Option<String> {
    jwt_claims(token)?
        .get("api_version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Build an explicit rustls 0.23 ClientConfig for the custom TLS modes:
/// `verify_certificate=false` (accept any cert) or a custom CA bundle (PEM).
fn build_tls_config(d: &Descriptor) -> Result<Arc<rustls::ClientConfig>, String> {
    // ClientConfig::builder() needs a process-default crypto provider; ureq
    // doesn't install one, so do it here (idempotent; ring matches ureq's backend).
    let _ = rustls::crypto::ring::default_provider().install_default();

    if !d.verify_certificate {
        let config = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertVerification))
            .with_no_client_auth();
        return Ok(Arc::new(config));
    }

    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    if let Some(ca_path) = &d.ca_cert_path {
        let f = std::fs::File::open(ca_path).map_err(|e| format!("open CA {}: {}", ca_path, e))?;
        let mut reader = std::io::BufReader::new(f);
        let mut added = 0usize;
        for cert in rustls_pemfile::certs(&mut reader) {
            let cert = cert.map_err(|e| format!("parse CA {}: {}", ca_path, e))?;
            roots
                .add(cert)
                .map_err(|e| format!("add CA {}: {}", ca_path, e))?;
            added += 1;
        }
        if added == 0 {
            return Err(format!("no certificates found in CA bundle {}", ca_path));
        }
    }
    let config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(config))
}

/// Accept-any server-cert verifier backing `verify_certificate=false`.
#[derive(Debug)]
struct NoCertVerification;

impl rustls::client::danger::ServerCertVerifier for NoCertVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        use rustls::SignatureScheme::*;
        vec![
            RSA_PKCS1_SHA256,
            RSA_PKCS1_SHA384,
            RSA_PKCS1_SHA512,
            ECDSA_NISTP256_SHA256,
            ECDSA_NISTP384_SHA384,
            ECDSA_NISTP521_SHA512,
            RSA_PSS_SHA256,
            RSA_PSS_SHA384,
            RSA_PSS_SHA512,
            ED25519,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jwt_exp_and_version_decode() {
        // header.payload.signature ; payload = {"exp":1700000000,"api_version":"2.31"}
        // base64url(no pad) of that payload:
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"exp":1700000000,"api_version":"2.31"}"#);
        let token = format!("aaa.{}.bbb", payload);
        assert_eq!(jwt_exp(&token), Some(1700000000));
        assert_eq!(jwt_api_version(&token).as_deref(), Some("2.31"));
    }

    #[test]
    fn jwt_garbage_is_none() {
        assert_eq!(jwt_exp("not-a-jwt"), None);
        assert_eq!(jwt_exp(""), None);
    }

    fn sample(verify: bool, ca: Option<&str>) -> Descriptor {
        let ca_field = ca
            .map(|c| format!(r#","ca_cert_path":"{}""#, c))
            .unwrap_or_default();
        let s = format!(
            r#"{{"api_server":"https://h/","access_key":"K","secret_key":"S","task_id":"t","socket_path":"/tmp/s.sock","verify_certificate":{}{}}}"#,
            verify, ca_field
        );
        Descriptor::from_json_str(&s).unwrap()
    }

    #[test]
    fn tls_config_verify_off_builds() {
        assert!(build_tls_config(&sample(false, None)).is_ok());
    }

    #[test]
    fn retryable_statuses() {
        for c in [429u16, 431, 502, 503, 504] {
            assert!(is_retryable_status(c), "{} should retry", c);
        }
        for c in [200u16, 400, 401, 404, 409, 500] {
            assert!(!is_retryable_status(c), "{} should not retry", c);
        }
    }

    #[test]
    fn backoff_grows_then_caps() {
        let b = BOUNDED_BACKOFF_MAX_MS;
        assert!(retry_backoff(1, b) < retry_backoff(2, b));
        assert!(retry_backoff(2, b) < retry_backoff(3, b));
        assert_eq!(retry_backoff(1, b), Duration::from_millis(200));
        // Bounded cap holds even at high attempts.
        assert!(retry_backoff(20, b) <= Duration::from_millis(BOUNDED_BACKOFF_MAX_MS));
        // Forever cap is higher and is reached/held at high attempts.
        assert_eq!(
            retry_backoff(50, FOREVER_BACKOFF_MAX_MS),
            Duration::from_millis(FOREVER_BACKOFF_MAX_MS)
        );
    }

    #[test]
    fn should_retry_policy() {
        // Non-retryable never retries, regardless of mode.
        assert!(!should_retry(false, false, RetryMode::Forever, 0));
        assert!(!should_retry(false, false, RetryMode::Bounded, 0));
        // Aborting stops even a forever retry (one-shot at shutdown).
        assert!(!should_retry(true, true, RetryMode::Forever, 0));
        // Forever retries indefinitely while not aborting.
        assert!(should_retry(true, false, RetryMode::Forever, 0));
        assert!(should_retry(true, false, RetryMode::Forever, 10_000));
        // Bounded retries only while under the cap.
        assert!(should_retry(true, false, RetryMode::Bounded, HTTP_MAX_RETRIES - 1));
        assert!(!should_retry(true, false, RetryMode::Bounded, HTTP_MAX_RETRIES));
    }

    #[test]
    fn sleep_interruptible_returns_early_on_abort() {
        let mut c = ClearmlClient::from_descriptor(&sample(true, None));
        c.set_abort_signal(Arc::new(AtomicBool::new(true))); // already aborting
        let start = std::time::Instant::now();
        c.sleep_interruptible(Duration::from_secs(60));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "interruptible sleep must bail promptly when aborting"
        );
    }

    #[test]
    fn refresh_due_schedules_at_halfway_with_floor() {
        let now = 1_000_000i64;
        // No token at all -> always due.
        assert!(refresh_due(false, 0, 0, 0, now));
        // 1h TTL, 40 min remaining: margin = max(30min, 5min) = 30min -> not due.
        assert!(!refresh_due(true, now + 2400, 3600, now - 1200, now));
        // 1h TTL, 20 min remaining -> past halfway -> due.
        assert!(refresh_due(true, now + 1200, 3600, now - 2400, now));
        // Short 8 min TTL: margin floors at 5 min; 4 min remaining -> due.
        assert!(refresh_due(true, now + 240, 480, now - 240, now));
        // Opaque token (exp unknown): recent refresh -> not due; stale -> due.
        assert!(!refresh_due(true, 0, 0, now - 100, now));
        assert!(refresh_due(true, 0, 0, now - 1000, now));
    }

    #[test]
    fn tls_config_missing_ca_errors() {
        assert!(build_tls_config(&sample(true, Some("/nonexistent/ca-bundle.pem"))).is_err());
    }
}
