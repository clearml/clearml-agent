//! IPC wire schema for shim <-> executioner communication.
//!
//! This crate is the single source of truth for the v1 wire format: both
//! the Rust shim and the (eventual) Rust executioner depend on it so they
//! cannot drift. The Python parent reads the same NDJSON over the unix
//! socket and parses it via TypedDict / dataclass with the same field
//! names and types.
//!
//! Discriminators:
//!   - shim -> parent: `Event`, tagged by `"kind"`.
//!   - parent -> shim: `Control`, tagged by `"action"`.
//!
//! Both `Serialize` and `Deserialize` are derived so this same crate covers
//! both directions: the shim serializes `Event`s (and deserializes
//! `Control`s); the parent does the opposite.

#![allow(clippy::module_name_repetitions)]

use serde::{Deserialize, Serialize};

/// Version of the event/control wire format shared by the shim and the
/// in-process reporter. Bump on any breaking change to a locked field.
pub const SCHEMA_VERSION: u32 = 1;

/// Byte-flow direction relative to the user task: `tx` = task -> server,
/// `rx` = server -> task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Tx,
    Rx,
}

/// Call-history capture mode, switched live from the task's User Properties
/// (`_snug_call_history`). Independent of metering, which always runs when a
/// sink is enabled regardless of this mode:
///   * `Off` — capture nothing.
///   * `Collect` — keep the last N request/response pairs in a ring buffer,
///     but don't print them.
///   * `Dump` — print the buffered backlog once (edge-triggered on entering
///     the mode), then keep collecting (sliding window continues).
///   * `Continuous` — print each request/response pair as it completes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CallHistoryMode {
    Off,
    Collect,
    Dump,
    Continuous,
}

/// Shim -> parent events. Wire-tagged by `"kind"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Event {
    /// First successful parse of an HTTP/1.x request on a connection.
    /// Emitted exactly once per request.
    RequestStarted {
        conn_id: u64,
        ts_ms: u64,
        host: String,
        path: String,
        method: String,
        whitelisted: bool,
        inject_headers: bool,
    },

    /// Per-write byte report. `tokens_est` is an approximate token count.
    BytesObserved {
        conn_id: u64,
        ts_ms: u64,
        direction: Direction,
        bytes: u64,
        tokens_est: u64,
    },

    /// Emitted when a request completes: at the next request boundary on a
    /// keep-alive connection, or on `SSL_free` / process exit for the final
    /// request. Carries the final byte counts and timing for the connection.
    RequestCompleted {
        conn_id: u64,
        ts_ms: u64,
        /// Response status code if we parsed it; otherwise None.
        status: Option<u16>,
        latency_ms: u64,
        bytes_tx: u64,
        bytes_rx: u64,
        tokens_in: u64,
        tokens_out: u64,
        /// True when `tokens_in`/`tokens_out` came from the provider's
        /// reported `usage` (parsed from the response body) rather than the
        /// byte-ratio estimate. `#[serde(default)]` keeps the wire
        /// backward-compatible with shims that predate body parsing.
        #[serde(default)]
        tokens_measured: bool,
        /// Prompt-cache breakdown of the input that `tokens_in` already folds in,
        /// for providers that report one: tokens served from the prompt cache
        /// (e.g. `cache_read_input_tokens`) and tokens written to it
        /// (e.g. `cache_creation_input_tokens`). Surfaced as their own SCALARS
        /// series so the dashboard separates fresh / cache-read / cache-write
        /// input, while `tokens_in` stays the summed billable total the usage sink
        /// reports. 0 when a provider reports no breakdown (a cache-inclusive
        /// prompt total) and for non-usage requests. `#[serde(default)]` for wire
        /// back-compat.
        #[serde(default)]
        cache_read_tokens: u64,
        #[serde(default)]
        cache_write_tokens: u64,
        /// Number of tool calls the model requested in this response
        /// (Anthropic `tool_use` / OpenAI `tool_calls` / Gemini
        /// `functionCall`). `#[serde(default)]` for wire back-compat.
        #[serde(default)]
        tool_calls: u64,
        /// Number of failed tool results submitted in this request's freshest
        /// turn (`is_error` etc.). `#[serde(default)]` for wire back-compat.
        #[serde(default)]
        tool_call_errors: u64,
        /// Names of the tools the model requested in this response, powering the
        /// per-tool-type calls graph (e.g. `["get_weather","search"]`; length
        /// equals `tool_calls`). Empty where only counts are available (OpenAI/
        /// Gemini streaming). `#[serde(default)]` for wire back-compat.
        #[serde(default)]
        tool_call_names: Vec<String>,
        /// Names of the tools whose freshest-turn results failed (`is_error`),
        /// resolved from `tool_use_id` → name within the request body. Powers
        /// the per-tool error overlay (the "(err)" variant on the "LLM Tool
        /// Calls by Tool" graph); its length equals `tool_call_errors`. Populated
        /// only for providers that mark tool errors structurally (others don't).
        /// `#[serde(default)]` for wire back-compat.
        #[serde(default)]
        tool_call_error_names: Vec<String>,
        /// Stable per-conversation id the shim derives by fingerprinting the
        /// request and matching it to the chat it continues (`crate::session`),
        /// NOT a client header. Robust to appended, retried, edited, and
        /// sliding-window-trimmed histories, so every turn of one chat shares
        /// the id while distinct chats differ. Lets the task-metrics sink split
        /// its scalar series per chat, so one task's N conversations plot as N
        /// lines instead of one. `None` (and omitted on the wire) for non-LLM
        /// traffic or an unparseable/oversized body;
        /// `#[serde(default, skip_serializing_if)]` keeps the wire
        /// backward-compatible with shims that predate it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chat_id: Option<String>,
        /// The LLM model this request used (e.g. `claude-opus-4-20250514`,
        /// `gpt-4o`, `gemini-1.5-pro`), for per-model usage attribution — the
        /// "coset" the usage aggregator groups token usage by, alongside the
        /// provider. The shim reads it from the provider's response (the served
        /// model, parsed from the same body as the usage), falling back to the
        /// model named in the request. `None` (and omitted on the wire) for
        /// non-LLM traffic or when no model could be determined;
        /// `#[serde(default, skip_serializing_if)]` keeps the wire
        /// backward-compatible with shims that predate it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },

    /// One captured request/response pair surfaced to the task console (in the
    /// `Collect`-then-dump or `Continuous` call-history modes). One event per
    /// pair so the reporter renders the request and response as one contiguous
    /// console block. `request_b64`/`response_b64` carry the (redacted, capped)
    /// raw HTTP bytes — request line + headers + body, and status line + headers
    /// + body — base64-encoded for binary safety; the reporter decodes them into
    /// human-readable console lines.
    CallHistoryEntry {
        conn_id: u64,
        ts_ms: u64,
        /// Monotonic per-task capture sequence (ring-buffer order).
        seq: u64,
        host: String,
        path: String,
        method: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<u16>,
        /// Base64 of the redacted request bytes. Empty if nothing was captured.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        request_b64: String,
        /// Base64 of the response bytes. Empty if no response was captured.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        response_b64: String,
        /// True if the captured request exceeded the per-direction cap.
        #[serde(default)]
        request_truncated: bool,
        /// True if the captured response exceeded the per-direction cap.
        #[serde(default)]
        response_truncated: bool,
        /// True when the response carried a `Content-Encoding` other than
        /// identity (the body bytes are compressed/opaque in the dump).
        #[serde(default)]
        response_compressed: bool,
        /// Per-conversation chat id — the task's running chat ordinal
        /// ("1", "2", …, assigned per metered call incl. uncaptured ones), so
        /// the console header shows the SAME number as the SCALARS series and
        /// reflects the call's true position in the task (not the capture order).
        /// `None` (omitted) when usage parsing is off / non-LLM / unparseable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chat_id: Option<String>,
    },

    /// A one-line call-history NOTICE (a mode switch or a dump summary),
    /// rendered verbatim after the `[SNUG-CALL]` prefix on the task console.
    /// Emitted by the shim's mode setter so each off/collect/dump/continuous
    /// transition is visible/auditable — the flips are otherwise silent.
    CallHistoryNotice { ts_ms: u64, text: String },

    /// Internal health signal. `conn_id`, `dropped_events`, and `host` are
    /// optional - present only when relevant to the specific diagnostic.
    ShimDiagnostic {
        ts_ms: u64,
        /// e.g. `"http2_unsupported"`, `"dlsym_fallback_rtld_default"`,
        /// `"dropped_events"`, `"whitelist_reload_failed"`,
        /// `"unknown_control_action"`.
        kind_detail: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        conn_id: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        dropped_events: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        host: Option<String>,
    },
}

/// Parent -> shim control messages. Wire-tagged by `"action"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Control {
    /// Flip the shim's call-history capture mode mid-task. New requests follow
    /// the new mode; in-flight requests finish under their starting mode.
    SetCallHistoryMode { mode: CallHistoryMode },

    /// Re-read the whitelist and atomically swap it in. Unused since the
    /// reporting pivot (control is in-process callbacks, not this wire enum),
    /// and now also a structural no-op: the whitelist arrives as immutable
    /// base64 env content (`CLEARML_SNUG_WHITELIST`). A future hot-reload
    /// would carry the new whitelist as a payload here instead.
    ReloadWhitelist,

    /// Update the shim's internal diagnostic-flush cadence.
    SetPollInterval { seconds: f64 },
}

impl Event {
    /// Current UNIX time in milliseconds. Convenience for callers
    /// populating `ts_ms` at emission time. Returns 0 if the system clock
    /// is before the UNIX epoch (impossible in practice).
    pub fn now_ts_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The shape of these JSON snapshots is the *contract* against the
    // Python parent. Changing them breaks the wire format; bump
    // SCHEMA_VERSION and keep the Python parser in sync if you really must.

    #[test]
    fn request_started_tagged_by_kind() {
        let e = Event::RequestStarted {
            conn_id: 42,
            ts_ms: 1700000000000,
            host: "api.anthropic.com".into(),
            path: "/v1/messages".into(),
            method: "POST".into(),
            whitelisted: true,
            inject_headers: true,
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"kind\":\"RequestStarted\""));
        assert!(s.contains("\"host\":\"api.anthropic.com\""));
        assert!(s.contains("\"method\":\"POST\""));
    }

    #[test]
    fn bytes_observed_round_trip() {
        let e = Event::BytesObserved {
            conn_id: 7,
            ts_ms: 200,
            direction: Direction::Tx,
            bytes: 1000,
            tokens_est: 0,
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"direction\":\"tx\""));
        let back: Event = serde_json::from_str(&s).unwrap();
        match back {
            Event::BytesObserved { direction, bytes, .. } => {
                assert_eq!(direction, Direction::Tx);
                assert_eq!(bytes, 1000);
            }
            _ => panic!("round-trip changed variant"),
        }
    }

    #[test]
    fn shim_diagnostic_omits_none_optionals() {
        let e = Event::ShimDiagnostic {
            ts_ms: 1,
            kind_detail: "http2_unsupported".into(),
            conn_id: Some(99),
            dropped_events: None,
            host: None,
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"kind\":\"ShimDiagnostic\""));
        assert!(s.contains("\"kind_detail\":\"http2_unsupported\""));
        assert!(s.contains("\"conn_id\":99"));
        // Nones are skipped, not serialized as null.
        assert!(!s.contains("\"dropped_events\""));
        assert!(!s.contains("\"host\""));
    }

    #[test]
    fn control_tagged_by_action_snake_case() {
        let c = Control::SetCallHistoryMode {
            mode: CallHistoryMode::Dump,
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains("\"action\":\"set_call_history_mode\""));
        assert!(s.contains("\"mode\":\"dump\""));
    }

    #[test]
    fn call_history_mode_serializes_lowercase() {
        // The exact wire strings the operator types into the User Property.
        assert_eq!(serde_json::to_string(&CallHistoryMode::Off).unwrap(), "\"off\"");
        assert_eq!(
            serde_json::to_string(&CallHistoryMode::Collect).unwrap(),
            "\"collect\""
        );
        assert_eq!(serde_json::to_string(&CallHistoryMode::Dump).unwrap(), "\"dump\"");
        assert_eq!(
            serde_json::to_string(&CallHistoryMode::Continuous).unwrap(),
            "\"continuous\""
        );
        let back: CallHistoryMode = serde_json::from_str("\"continuous\"").unwrap();
        assert_eq!(back, CallHistoryMode::Continuous);
    }

    #[test]
    fn call_history_entry_tagged_by_kind_and_round_trips() {
        let e = Event::CallHistoryEntry {
            conn_id: 9,
            ts_ms: 1700000000000,
            seq: 3,
            host: "api.anthropic.com".into(),
            path: "/v1/messages".into(),
            method: "POST".into(),
            status: Some(200),
            request_b64: "cmVx".into(),
            response_b64: "cmVzcA==".into(),
            request_truncated: false,
            response_truncated: true,
            response_compressed: false,
            chat_id: Some("8".into()),
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"kind\":\"CallHistoryEntry\""));
        assert!(s.contains("\"seq\":3"));
        assert!(s.contains("\"request_b64\":\"cmVx\""));
        assert!(s.contains("\"chat_id\":\"8\""));
        match serde_json::from_str::<Event>(&s).unwrap() {
            Event::CallHistoryEntry {
                seq,
                status,
                response_truncated,
                chat_id,
                ..
            } => {
                assert_eq!(seq, 3);
                assert_eq!(status, Some(200));
                assert!(response_truncated);
                assert_eq!(chat_id.as_deref(), Some("8"));
            }
            _ => panic!("round-trip changed variant"),
        }
    }

    #[test]
    fn call_history_entry_defaults_for_minimal_wire() {
        // A minimal entry (only the required fields) deserializes with the
        // optional fields defaulted: no truncation/compression flags, empty
        // payloads, no status, no chat id.
        let minimal = r#"{"kind":"CallHistoryEntry","conn_id":1,"ts_ms":2,"seq":0,"host":"h","path":"/","method":"GET"}"#;
        match serde_json::from_str::<Event>(minimal).unwrap() {
            Event::CallHistoryEntry {
                status,
                request_b64,
                response_b64,
                request_truncated,
                response_truncated,
                response_compressed,
                chat_id,
                ..
            } => {
                assert!(status.is_none());
                assert!(request_b64.is_empty() && response_b64.is_empty());
                assert!(!request_truncated && !response_truncated && !response_compressed);
                assert!(chat_id.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn call_history_notice_round_trips() {
        let e = Event::CallHistoryNotice {
            ts_ms: 5,
            text: "mode -> off (was continuous)".into(),
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"kind\":\"CallHistoryNotice\""));
        match serde_json::from_str::<Event>(&s).unwrap() {
            Event::CallHistoryNotice { text, .. } => {
                assert_eq!(text, "mode -> off (was continuous)")
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn control_reload_whitelist_just_the_tag() {
        let c = Control::ReloadWhitelist;
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(s, r#"{"action":"reload_whitelist"}"#);
    }

    #[test]
    fn control_round_trip_set_poll_interval() {
        let c = Control::SetPollInterval { seconds: 12.5 };
        let s = serde_json::to_string(&c).unwrap();
        let back: Control = serde_json::from_str(&s).unwrap();
        match back {
            Control::SetPollInterval { seconds } => assert_eq!(seconds, 12.5),
            _ => panic!("round-trip changed variant"),
        }
    }

    #[test]
    fn now_ts_ms_nonzero() {
        // Sanity: time has moved past 1970.
        assert!(Event::now_ts_ms() > 1_500_000_000_000);
    }

    #[test]
    fn request_completed_tokens_measured_defaults_false_on_old_wire() {
        // A RequestCompleted serialized by a shim that predates body
        // parsing has no `tokens_measured` field; it must deserialize as
        // false rather than failing (wire back-compat).
        let old = r#"{"kind":"RequestCompleted","conn_id":1,"ts_ms":2,"status":200,"latency_ms":3,"bytes_tx":4,"bytes_rx":5,"tokens_in":6,"tokens_out":7}"#;
        let back: Event = serde_json::from_str(old).unwrap();
        match back {
            Event::RequestCompleted {
                tokens_measured,
                tokens_in,
                status,
                ..
            } => {
                assert!(!tokens_measured);
                assert_eq!(tokens_in, 6);
                assert_eq!(status, Some(200));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_completed_round_trips_tokens_measured_true() {
        let e = Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 2,
            status: Some(200),
            latency_ms: 3,
            bytes_tx: 4,
            bytes_rx: 5,
            tokens_in: 14,
            tokens_out: 495,
            tokens_measured: true,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: None,
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"tokens_measured\":true"));
        let back: Event = serde_json::from_str(&s).unwrap();
        match back {
            Event::RequestCompleted {
                tokens_measured,
                tokens_out,
                ..
            } => {
                assert!(tokens_measured);
                assert_eq!(tokens_out, 495);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_completed_tool_fields_default_and_round_trip() {
        // Old wire (no tool fields) -> default 0.
        let old = r#"{"kind":"RequestCompleted","conn_id":1,"ts_ms":2,"status":200,"latency_ms":3,"bytes_tx":4,"bytes_rx":5,"tokens_in":6,"tokens_out":7}"#;
        match serde_json::from_str::<Event>(old).unwrap() {
            Event::RequestCompleted {
                tool_calls,
                tool_call_errors,
                ..
            } => {
                assert_eq!(tool_calls, 0);
                assert_eq!(tool_call_errors, 0);
            }
            _ => panic!("wrong variant"),
        }
        // New wire round-trips the counts.
        let e = Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 2,
            status: Some(200),
            latency_ms: 3,
            bytes_tx: 4,
            bytes_rx: 5,
            tokens_in: 14,
            tokens_out: 495,
            tokens_measured: true,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 3,
            tool_call_errors: 1,
            tool_call_names: vec!["get_weather".into(), "search".into(), "get_time".into()],
            tool_call_error_names: vec!["search".into()],
            chat_id: Some("3f9a1c2b7d4e5f60".into()),
            model: Some("claude-haiku-4-5".into()),
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"tool_calls\":3"));
        assert!(s.contains("\"chat_id\":\"3f9a1c2b7d4e5f60\""));
        assert!(s.contains("\"tool_call_errors\":1"));
        assert!(s.contains("\"tool_call_error_names\":[\"search\"]"));
        match serde_json::from_str::<Event>(&s).unwrap() {
            Event::RequestCompleted {
                tool_calls,
                tool_call_errors,
                tool_call_names,
                tool_call_error_names,
                ..
            } => {
                assert_eq!(tool_calls, 3);
                assert_eq!(tool_call_errors, 1);
                assert_eq!(tool_call_names, vec!["get_weather", "search", "get_time"]);
                assert_eq!(tool_call_error_names, vec!["search"]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_completed_chat_id_default_and_round_trip() {
        // Old wire (no chat_id) -> None, and untagged events omit the field.
        let old = r#"{"kind":"RequestCompleted","conn_id":1,"ts_ms":2,"status":200,"latency_ms":3,"bytes_tx":4,"bytes_rx":5,"tokens_in":6,"tokens_out":7}"#;
        match serde_json::from_str::<Event>(old).unwrap() {
            Event::RequestCompleted { chat_id, .. } => assert!(chat_id.is_none()),
            _ => panic!("wrong variant"),
        }
        let e = Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 2,
            status: Some(200),
            latency_ms: 3,
            bytes_tx: 4,
            bytes_rx: 5,
            tokens_in: 6,
            tokens_out: 7,
            tokens_measured: true,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: None,
        };
        // None is skipped, not serialized as null.
        assert!(!serde_json::to_string(&e).unwrap().contains("chat_id"));

        let tagged = Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 2,
            status: Some(200),
            latency_ms: 3,
            bytes_tx: 4,
            bytes_rx: 5,
            tokens_in: 6,
            tokens_out: 7,
            tokens_measured: true,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: Some("abc123".into()),
            model: None,
        };
        let s = serde_json::to_string(&tagged).unwrap();
        match serde_json::from_str::<Event>(&s).unwrap() {
            Event::RequestCompleted { chat_id, .. } => {
                assert_eq!(chat_id.as_deref(), Some("abc123"))
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_completed_model_default_and_round_trip() {
        // Old wire (no model) -> None (back-compat with pre-model shims).
        let old = r#"{"kind":"RequestCompleted","conn_id":1,"ts_ms":2,"status":200,"latency_ms":3,"bytes_tx":4,"bytes_rx":5,"tokens_in":6,"tokens_out":7}"#;
        match serde_json::from_str::<Event>(old).unwrap() {
            Event::RequestCompleted { model, .. } => assert!(model.is_none()),
            _ => panic!("wrong variant"),
        }
        let base = |model: Option<&str>| Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 2,
            status: Some(200),
            latency_ms: 3,
            bytes_tx: 4,
            bytes_rx: 5,
            tokens_in: 6,
            tokens_out: 7,
            tokens_measured: true,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: model.map(str::to_string),
        };
        // None is skipped, not serialized as null.
        assert!(!serde_json::to_string(&base(None)).unwrap().contains("model"));
        // Some serializes and round-trips the model string.
        let s = serde_json::to_string(&base(Some("claude-opus-4-20250514"))).unwrap();
        assert!(s.contains("\"model\":\"claude-opus-4-20250514\""));
        match serde_json::from_str::<Event>(&s).unwrap() {
            Event::RequestCompleted { model, .. } => {
                assert_eq!(model.as_deref(), Some("claude-opus-4-20250514"))
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_completed_cache_tokens_default_and_round_trip() {
        // Old wire (no cache fields) -> default 0 (back-compat with pre-cache
        // shims).
        let old = r#"{"kind":"RequestCompleted","conn_id":1,"ts_ms":2,"status":200,"latency_ms":3,"bytes_tx":4,"bytes_rx":5,"tokens_in":6,"tokens_out":7}"#;
        match serde_json::from_str::<Event>(old).unwrap() {
            Event::RequestCompleted {
                cache_read_tokens,
                cache_write_tokens,
                ..
            } => {
                assert_eq!(cache_read_tokens, 0);
                assert_eq!(cache_write_tokens, 0);
            }
            _ => panic!("wrong variant"),
        }
        // New wire round-trips the cache buckets; tokens_in stays the summed
        // billable total, independent of the breakdown.
        let e = Event::RequestCompleted {
            conn_id: 1,
            ts_ms: 2,
            status: Some(200),
            latency_ms: 3,
            bytes_tx: 4,
            bytes_rx: 5,
            tokens_in: 45302,
            tokens_out: 13,
            tokens_measured: true,
            cache_read_tokens: 45000,
            cache_write_tokens: 300,
            tool_calls: 0,
            tool_call_errors: 0,
            tool_call_names: vec![],
            tool_call_error_names: vec![],
            chat_id: None,
            model: Some("claude-sonnet-4-5".into()),
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"cache_read_tokens\":45000"));
        assert!(s.contains("\"cache_write_tokens\":300"));
        match serde_json::from_str::<Event>(&s).unwrap() {
            Event::RequestCompleted {
                tokens_in,
                cache_read_tokens,
                cache_write_tokens,
                ..
            } => {
                assert_eq!(tokens_in, 45302);
                assert_eq!(cache_read_tokens, 45000);
                assert_eq!(cache_write_tokens, 300);
            }
            _ => panic!("wrong variant"),
        }
    }
}
