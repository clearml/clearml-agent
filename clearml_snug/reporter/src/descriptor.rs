//! The agent → shim handoff descriptor.
//!
//! A self-contained JSON blob the agent writes into an anonymous `memfd` and
//! whose fd number it passes to the shim via `CLEARML_SNUG_CRED_FD`. The shim
//! reads it once at `#[ctor]` time and closes the fd. It carries everything the
//! in-process reporter needs to reach the ClearML backend and identify the task
//! — without a live Python `Session`.
//!
//! Credentials are token-primary: in most deployments the agent has only a
//! token (no access/secret), so `access_key`/`secret_key` default to empty and
//! the reporter refreshes the token via Bearer-renewal (see `api.rs`). When
//! key/secret ARE present they're used for full re-login robustness.

use std::io::Read;
use std::os::unix::io::{FromRawFd, RawFd};

use serde::Deserialize;

fn default_true() -> bool {
    true
}

fn default_poll_interval() -> f64 {
    10.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Descriptor {
    // Backend connection.
    pub api_server: String,
    /// HTTP Basic credentials. Often absent (token-only deployments) — default
    /// empty, in which case the reporter relies on `auth_token` + Bearer-renewal.
    #[serde(default)]
    pub access_key: String,
    #[serde(default)]
    pub secret_key: String,
    /// Pre-issued token. The primary credential in token-only deployments;
    /// refreshed in-process via Bearer-renewal before expiry.
    #[serde(default)]
    pub auth_token: Option<String>,
    /// Absolute expiry (unix seconds) of `auth_token`, supplied by the agent for
    /// opaque (non-JWT) tokens whose `exp` the reporter can't read itself. Lets
    /// the proactive refresh schedule correctly even for opaque tokens.
    #[serde(default)]
    pub token_expiry_sec: Option<i64>,
    /// Honor `api.verify_certificate` (default true).
    #[serde(default = "default_true")]
    pub verify_certificate: bool,
    /// Custom CA bundle (PEM) for servers with a private CA.
    #[serde(default)]
    pub ca_cert_path: Option<String>,

    // Task identity.
    pub task_id: String,
    #[serde(default)]
    pub worker_id: String,
    /// The task's owning user id, for usage attribution on `report_llm_usage`.
    /// Empty when the agent didn't supply it; the backend then derives the user
    /// from the task itself.
    #[serde(default)]
    pub user: String,
    /// The task's project id, for usage attribution on `report_llm_usage`. Empty
    /// when unsupplied; the backend derives it from the task.
    #[serde(default)]
    pub project: String,
    /// How often the control plane polls the task's runtime properties (seconds).
    #[serde(default = "default_poll_interval")]
    pub poll_interval_sec: f64,

    // Reporting sinks.
    /// Emit per-request LLM usage to the backend `report_llm_usage` endpoint.
    #[serde(default)]
    pub report_usage_events: bool,
    /// Emit per-request usage scalars to the task's own SCALARS tab.
    #[serde(default)]
    pub report_task_metrics: bool,
    /// Which task-metric fields to report, resolved by the agent (env override,
    /// else the configured list). Empty means the sink reports all known fields.
    #[serde(default)]
    pub task_metrics_fields: Vec<String>,
    /// Forward every `RequestCompleted` event (verbatim) to this URL. None
    /// disables it. Independent of the whitelist gate the other sinks apply.
    #[serde(default)]
    pub aggregator_url: Option<String>,

    /// Hostnames of the ClearML backend this task reports to (api / files / web
    /// servers), resolved by the agent. The shim suppresses metering of the
    /// task's own ClearML SDK traffic to these hosts so it isn't billed as LLM
    /// usage. Carried here purely for the shim's use; the reporter itself uses
    /// rustls and is already invisible to the hooks. Empty = nothing excluded.
    #[serde(default)]
    pub self_hosts: Vec<String>,
}

impl Descriptor {
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("invalid descriptor JSON: {}", e))
    }

    /// Read the descriptor from an inherited fd (the agent's `memfd`), consuming
    /// it to EOF and closing it on return. The agent `lseek`s the memfd to 0
    /// before exec, so the read starts at the beginning.
    pub fn from_fd(fd: RawFd) -> Result<Self, String> {
        // SAFETY: `fd` is a memfd the agent created, populated, and marked
        // inheritable; we take ownership and the `File` closes it on drop.
        let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
        let mut data = String::new();
        f.read_to_string(&mut data)
            .map_err(|e| format!("cannot read descriptor fd {}: {}", fd, e))?;
        Self::from_json_str(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_descriptor_with_defaults() {
        // Token-only handoff: just api_server + token + task_id.
        let s = r#"{
            "api_server": "https://api.example.com/",
            "auth_token": "tok",
            "task_id": "t1"
        }"#;
        let d = Descriptor::from_json_str(s).expect("parse");
        assert_eq!(d.task_id, "t1");
        assert!(d.verify_certificate, "verify_certificate defaults true");
        assert_eq!(d.auth_token.as_deref(), Some("tok"));
        // Creds optional (token-only) and worker empty by default.
        assert_eq!(d.access_key, "");
        assert_eq!(d.secret_key, "");
        assert_eq!(d.worker_id, "");
        assert!(d.user.is_empty() && d.project.is_empty());
        assert!(d.token_expiry_sec.is_none());
        // Sinks default off / empty.
        assert!(!d.report_usage_events);
        assert!(!d.report_task_metrics);
        assert!(d.task_metrics_fields.is_empty());
        // self-hosts default empty (old agents / stderr fallback) — nothing excluded.
        assert!(d.self_hosts.is_empty());
    }

    #[test]
    fn parses_self_hosts() {
        let s = r#"{"api_server":"https://api.clear.ml/","task_id":"t",
            "self_hosts":["api.clear.ml","files.clear.ml","app.clear.ml"]}"#;
        let d = Descriptor::from_json_str(s).expect("parse");
        assert_eq!(d.self_hosts, vec!["api.clear.ml", "files.clear.ml", "app.clear.ml"]);
    }

    #[test]
    fn parses_user_and_project() {
        let s = r#"{"api_server":"https://h/","task_id":"t","user":"u-1","project":"p-2"}"#;
        let d = Descriptor::from_json_str(s).expect("parse");
        assert_eq!(d.user, "u-1");
        assert_eq!(d.project, "p-2");
    }

    #[test]
    fn parses_key_secret_and_sink_fields() {
        let s = r#"{
            "api_server": "https://h/", "access_key": "K", "secret_key": "S",
            "task_id": "t",
            "report_usage_events": true, "report_task_metrics": true,
            "task_metrics_fields": ["tokens_in", "requests"]
        }"#;
        let d = Descriptor::from_json_str(s).expect("parse");
        assert_eq!(d.access_key, "K");
        assert_eq!(d.secret_key, "S");
        assert!(d.report_usage_events);
        assert!(d.report_task_metrics);
        assert_eq!(d.task_metrics_fields, vec!["tokens_in", "requests"]);
        assert!(d.aggregator_url.is_none());
    }

    #[test]
    fn parses_token_expiry_and_aggregator() {
        let s = r#"{"api_server":"https://h/","auth_token":"opaque","token_expiry_sec":1700000000,
            "task_id":"t","aggregator_url":"https://agg.example/ingest"}"#;
        let d = Descriptor::from_json_str(s).expect("parse");
        assert_eq!(d.token_expiry_sec, Some(1700000000));
        assert_eq!(d.aggregator_url.as_deref(), Some("https://agg.example/ingest"));
    }

    #[test]
    fn rejects_missing_required_field() {
        // api_server + task_id are the only required fields; creds are optional.
        let s = r#"{"api_server":"h","auth_token":"tok"}"#;
        assert!(Descriptor::from_json_str(s).is_err());
    }
}
