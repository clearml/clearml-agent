//! Whitelist loader for the v1 JSON whitelist format.
//!
//! The whitelist arrives as base64-encoded JSON in the
//! `CLEARML_SNUG_WHITELIST` env var (the agent serializes the inline
//! `agent.snug.whitelist` config block). It is decoded + parsed once at
//! ctor time into a hot-swappable `OnceLock<ArcSwap<Whitelist>>`. The hot
//! path reads with a single `.load_full()` (lock-free; one atomic
//! increment per read), well within budget for the LLM-traffic scale we
//! target.
//!
//! Failures (env var unset, base64 decode error, JSON parse error) fall
//! through to an empty whitelist. The shim still meters bytes; it just
//! doesn't recognize any host. The safe default is "meter".
//!
//! Mid-task hot-reload: the EFFECTIVE whitelist is `BASE` (the immutable
//! launch-time config, admin > env > file) plus runtime ADDITIONS parsed from
//! the task's `_snug_whitelist` User Property. `apply_whitelist_additions()`
//! (the poll callback) re-merges additions onto BASE and atomically swaps the
//! result in; clearing the property reverts to exactly BASE. Additions can only
//! ADD hosts — a rule colliding with a BASE host is dropped, so a task can never
//! override an admin-defined rule or change `default_action`.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::OnceLock;

use arc_swap::ArcSwap;
use base64::Engine as _;
use clearml_snug_messages::Event;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Whitelist {
    /// Schema version. Currently must be 1; future versions add fields rather
    /// than break existing ones (additive evolution). Deserialized but not
    /// consulted at runtime (reserved for forward-compat), hence `dead_code` —
    /// scoped to this field so a future genuinely-unused field still warns.
    #[allow(dead_code)]
    pub version: u32,
    /// "meter" (count bytes) or "ignore" (pass-through). Behavior for
    /// hosts with no matching rule.
    #[serde(default = "default_action_default")]
    pub default_action: String,
    #[serde(default)]
    pub rules: Vec<WhitelistRule>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhitelistRule {
    /// Host pattern matched (ASCII case-insensitively) against the `Host:`
    /// header value. A leading and/or trailing `*` makes it a wildcard:
    /// `*.anthropic.com` (suffix), `api.anthropic.*` (prefix), `*anthropic*`
    /// (substring), `*` (any host); no `*` is an exact match. Only a boundary
    /// `*` is special — see [`host_matches`].
    pub host: String,
    /// Default "/" - matches any path on the host.
    #[serde(default = "default_path_prefix")]
    pub path_prefix: String,
    /// Default false. When true, the shim splices `project:` and
    /// `session:` headers into outbound requests on the matched
    /// connection.
    #[serde(default)]
    pub inject_headers: bool,
    /// Default "approx". Per-host override for the byte-ratio token
    /// estimator (one of "claude" | "cl100k" | "approx").
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
    /// Default false. When true, an UNMEASURED request on this host (a response
    /// carrying a model but no `usage` object — a consumer chat wire) is
    /// byte-estimated instead of dropped, but only for a request the proxy can
    /// tell is a real generation (see `completion_path`). Real-API hosts leave
    /// this false so their `count_tokens`/telemetry calls are never estimated.
    /// Read by the proxy's estimate gate; unused in the shim (shared source).
    #[serde(default)]
    #[allow(dead_code)]
    pub estimate_unmeasured: bool,
    /// Default "" (never a completion). When `estimate_unmeasured` is set, a
    /// `POST` whose decoded path matches this boundary-`*` wildcard (e.g.
    /// `*/completion`, matched like `host` but case-sensitive, query stripped) is
    /// treated as a real generation whose input is estimated from the request
    /// body; every other request on the host stays at 0 tokens.
    #[serde(default)]
    #[allow(dead_code)]
    pub completion_path: String,
    /// Default "". Provider hint ("anthropic" | "openai" | "gemini") for a host
    /// whose wire format matches a known provider but whose hostname isn't in the
    /// shared base host->provider map (e.g. a consumer chat host such as
    /// claude.ai speaking the Anthropic wire). Empty falls back to the base map.
    #[serde(default)]
    #[allow(dead_code)]
    pub provider: String,
}

fn default_action_default() -> String { "meter".into() }
fn default_path_prefix() -> String { "/".into() }
fn default_tokenizer() -> String { "approx".into() }

impl WhitelistRule {
    /// Whether a decoded request line is a real generation (a "completion") for
    /// this rule: `completion_path` non-empty and matching `path` (boundary-`*`
    /// wildcard, query string stripped, case-sensitive), with method `POST`. An
    /// empty `completion_path` never matches, so a host that doesn't declare one
    /// is never estimated from its request line. `method`/`path` are `None` when
    /// HPACK decoding gave up — treated as NOT a completion (fail closed).
    /// Read by the proxy's estimate gate; unused in the shim (shared source).
    #[allow(dead_code)]
    pub fn is_completion(&self, method: Option<&str>, path: Option<&str>) -> bool {
        if self.completion_path.is_empty() {
            return false;
        }
        if method != Some("POST") {
            return false;
        }
        match path {
            Some(p) => wildcard_matches(&self.completion_path, p.split('?').next().unwrap_or(p)),
            None => false,
        }
    }
}

impl Whitelist {
    /// The zero-rule whitelist. Equivalent to "SNUG is on but nothing
    /// special applies to any host" - byte counting still happens via the
    /// state machine, but no injection or call-history capture.
    pub fn empty() -> Self {
        Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![],
        }
    }

    pub fn load_from_env() -> Self {
        match std::env::var("CLEARML_SNUG_WHITELIST") {
            Ok(b64) if !b64.is_empty() => Self::load_from_b64(&b64),
            _ => Self::empty(),
        }
    }

    /// Decode base64 whitelist content and parse it as v1 JSON. Both a
    /// decode failure and a JSON parse failure fall through to `empty()`,
    /// preserving the "bad config -> safe default = meter" contract.
    pub fn load_from_b64(b64: &str) -> Self {
        let bytes = match base64::engine::general_purpose::STANDARD.decode(b64.trim()) {
            Ok(b) => b,
            Err(_) => return Self::empty(),
        };
        match serde_json::from_slice::<Whitelist>(&bytes) {
            Ok(wl) => wl,
            Err(_) => Self::empty(),
        }
    }

    pub fn matches(&self, host: &str, path: &str) -> Option<&WhitelistRule> {
        self.rules
            .iter()
            .find(|r| host_matches(&r.host, host) && path.starts_with(&r.path_prefix))
    }

    /// The `provider` hint of the first rule whose host matches `host` (ignoring
    /// path_prefix, since provider is a host-level property), or "" if none. The
    /// proxy uses this to resolve a host->provider override from config for a
    /// host not in the shim-shared base map. Read by the proxy; unused in the
    /// shim (shared source).
    #[allow(dead_code)]
    pub fn provider_hint(&self, host: &str) -> &str {
        self.rules
            .iter()
            .find(|r| host_matches(&r.host, host) && !r.provider.is_empty())
            .map(|r| r.provider.as_str())
            .unwrap_or("")
    }
}

/// Match a request `host` against a rule's `host` pattern (ASCII
/// case-insensitive). A leading and/or trailing `*` turns the pattern into a
/// wildcard; everything else is matched literally:
///
///   - `*.anthropic.com` (leading `*`)  → suffix: host ends with `.anthropic.com`
///   - `api.anthropic.*`  (trailing `*`) → prefix: host starts with `api.anthropic.`
///   - `*anthropic*`      (both)         → substring: host contains `anthropic`
///   - `api.anthropic.com` (neither)     → exact match (the original behavior)
///   - `*` (or `**`) alone               → matches any host
///
/// Only a boundary `*` is special; a `*` in the middle is literal (so
/// `api.*.com` matches only the literal host `api.*.com`, which never occurs in
/// practice). Note the footgun: a suffix wildcard WITHOUT a leading dot is
/// broader than it looks — `*anthropic.com` also matches `evilanthropic.com`;
/// prefer `*.anthropic.com`.
fn host_matches(pattern: &str, host: &str) -> bool {
    // Fast path: exact match, no allocation (the common case).
    let lead = pattern.starts_with('*');
    let trail = pattern.len() > 1 && pattern.ends_with('*');
    if !lead && !trail {
        return pattern.eq_ignore_ascii_case(host);
    }
    // Case-insensitive host matching: lowercase both sides, then boundary-`*`
    // match. `matches()` runs once per connection (the outcome is cached in the
    // ConnectionState), so this per-call lowercasing is off the per-byte hot
    // path.
    wildcard_matches(&pattern.to_ascii_lowercase(), &host.to_ascii_lowercase())
}

/// Boundary-`*` wildcard match of `value` against `pattern`, CASE-SENSITIVE (see
/// [`host_matches`] for the case-insensitive host variant, which lowercases both
/// sides before delegating here). A leading and/or trailing `*` makes the pattern
/// a suffix/prefix/substring/any matcher; no `*` is an exact match. Only a
/// boundary `*` is special; a `*` in the middle is literal.
fn wildcard_matches(pattern: &str, value: &str) -> bool {
    let lead = pattern.starts_with('*');
    // `len() > 1` so a bare "*" is treated as lead-only, not lead AND trail.
    let trail = pattern.len() > 1 && pattern.ends_with('*');
    if !lead && !trail {
        return pattern == value;
    }
    // Strip the boundary `*`(s) to get the literal core. `*` is ASCII (1 byte),
    // so these byte offsets always land on char boundaries.
    let core = &pattern[lead as usize..pattern.len() - trail as usize];
    if core.is_empty() {
        // Bare "*" / "**": match anything.
        return true;
    }
    match (lead, trail) {
        (true, true) => value.contains(core),
        (true, false) => value.ends_with(core),
        (false, true) => value.starts_with(core),
        // `(false, false)` returned early above.
        (false, false) => unreachable!(),
    }
}

// --- Effective whitelist = immutable BASE + runtime additions ---------------
//
// `BASE` is the launch-time whitelist (admin > env > file, decoded from
// `CLEARML_SNUG_WHITELIST`); it is never mutated. The hot-swappable `WHITELIST`
// holds the EFFECTIVE whitelist = BASE plus any rules added at runtime from the
// task's `_snug_whitelist` User Property. Every re-merge starts from BASE, so
// clearing the property reverts the effective whitelist to exactly BASE.

/// Per-task cap on how many rules `_snug_whitelist` may add (a runaway/hostile
/// value can't bloat the per-connection `matches()` scan).
const MAX_WHITELIST_ADDITIONS: usize = 50;
/// Hard cap on the raw property value we'll parse (defensive; the backend also
/// bounds hyperparam value size).
const MAX_WHITELIST_INPUT_BYTES: usize = 16 * 1024;

static BASE: OnceLock<Arc<Whitelist>> = OnceLock::new();
static WHITELIST: OnceLock<ArcSwap<Whitelist>> = OnceLock::new();

/// The immutable launch-time whitelist. Runtime additions merge ON TOP of this;
/// it is the floor a task can never override.
fn base() -> &'static Arc<Whitelist> {
    BASE.get_or_init(|| Arc::new(Whitelist::load_from_env()))
}

/// Hot-swappable EFFECTIVE whitelist, seeded from BASE.
fn cell() -> &'static ArcSwap<Whitelist> {
    WHITELIST.get_or_init(|| ArcSwap::new(base().clone()))
}

/// Snapshot the current effective whitelist. Returns an `Arc<Whitelist>` whose
/// lifetime is independent of subsequent swaps - once you've got the Arc, your
/// view is stable for the rest of the request.
pub fn current() -> Arc<Whitelist> {
    cell().load_full()
}

/// Force BASE + the effective cell at ctor time (so a parse failure surfaces as
/// `rules=0` in the init log) and apply any launch-time additions handed via
/// `CLEARML_SNUG_WHITELIST_ADDITIONS` (the per-task predefine), BEFORE any hook
/// fires. Idempotent.
pub fn initialize() -> Arc<Whitelist> {
    let _ = base();
    let _ = cell();
    if let Ok(raw) = std::env::var("CLEARML_SNUG_WHITELIST_ADDITIONS") {
        if !raw.trim().is_empty() {
            // Quiet: the event channel isn't installed yet during the ctor, and
            // the init log line already reports the resulting rule count.
            let _ = swap_with_additions(&raw);
        }
    }
    current()
}

/// Stats from one additions merge (surfaced in the console NOTICE).
#[derive(Default, Debug, PartialEq)]
struct MergeStats {
    added: usize,
    dropped_covered: usize,
    dropped_over_cap: usize,
    dropped_invalid: usize,
    cleared: bool,
    /// Set when the merge declined to apply (the caller keeps the prior
    /// whitelist): a short human reason for the operator-facing notice.
    reason: &'static str,
}

impl MergeStats {
    fn summary(&self) -> String {
        if self.cleared {
            "whitelist: additions cleared".to_string()
        } else {
            format!(
                "whitelist: added={} dropped_covered={} over_cap={} invalid={}",
                self.added, self.dropped_covered, self.dropped_over_cap, self.dropped_invalid
            )
        }
    }
}

fn rule_from_host(host: &str) -> WhitelistRule {
    WhitelistRule {
        host: host.to_string(),
        path_prefix: default_path_prefix(),
        inject_headers: false,
        tokenizer: default_tokenizer(),
        estimate_unmeasured: false,
        completion_path: String::new(),
        provider: String::new(),
    }
}

/// A host token is usable if it's a plausible host pattern: non-empty, within
/// the DNS length bound, and free of path/whitespace/control characters.
fn is_valid_host_token(t: &str) -> bool {
    !t.is_empty()
        && t.len() <= 253
        && !t.contains('/')
        && !t.chars().any(|c| c.is_whitespace() || c.is_control())
}

fn base_only(base: &Whitelist) -> Whitelist {
    Whitelist {
        version: base.version,
        default_action: base.default_action.clone(),
        rules: base.rules.clone(),
    }
}

/// Parse `raw` (a JSON rule array, or a comma/space/newline host-list shorthand)
/// into additions and merge them ADDITIVELY onto `base`. Pure + unit-testable.
///
/// Returns `(Some(effective), stats)` for an applied merge, or `(None, stats)`
/// when `raw` is a non-empty value we couldn't parse into any usable rule — the
/// caller then KEEPS the prior whitelist. An empty / all-separators value clears
/// the additions (effective == base). Admin protection: a candidate whose host
/// is already covered by a base rule is dropped, `version`/`default_action` are
/// always taken from base, and shorthand rules are metering-only.
fn merge_whitelist_additions(
    base: &Whitelist,
    raw: &str,
    cap: usize,
) -> (Option<Whitelist>, MergeStats) {
    let mut stats = MergeStats::default();
    if raw.len() > MAX_WHITELIST_INPUT_BYTES {
        stats.reason = "oversized";
        return (None, stats); // oversized: keep prior
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        stats.cleared = true;
        return (Some(base_only(base)), stats);
    }

    let candidates: Vec<WhitelistRule> = if trimmed.starts_with('[') {
        // Full JSON rule objects (host + optional path_prefix /
        // inject_headers / tokenizer; serde fills omitted fields).
        match serde_json::from_str::<Vec<WhitelistRule>>(trimmed) {
            Ok(rules) => rules,
            Err(_) => {
                stats.reason = "invalid json";
                return (None, stats); // malformed JSON: keep prior
            }
        }
    } else {
        // Host-list shorthand → metering-only rules with default fields.
        let tokens: Vec<&str> = trimmed
            .split(|c: char| c == ',' || c.is_whitespace())
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            // Pure separators: treat as an intentional clear.
            stats.cleared = true;
            return (Some(base_only(base)), stats);
        }
        let mut v = Vec::new();
        for t in tokens {
            if is_valid_host_token(t) {
                v.push(rule_from_host(t));
            } else {
                stats.dropped_invalid += 1;
            }
        }
        if v.is_empty() {
            // Had tokens, but every one was invalid: keep prior.
            stats.reason = "no valid hosts";
            return (None, stats);
        }
        v
    };

    let mut seen: HashSet<String> = HashSet::new();
    let mut additions: Vec<WhitelistRule> = Vec::new();
    for mut rule in candidates {
        let host = rule.host.trim().to_string();
        if !is_valid_host_token(&host) {
            stats.dropped_invalid += 1;
            continue;
        }
        let key = host.to_ascii_lowercase();
        if !seen.insert(key) {
            continue; // intra-list duplicate (case-insensitive)
        }
        // Admin protection: a host already covered by a BASE rule can't be
        // re-specified (the task can't flip inject_headers/tokenizer on it).
        if base.rules.iter().any(|r| host_matches(&r.host, &host)) {
            stats.dropped_covered += 1;
            continue;
        }
        if additions.len() >= cap {
            stats.dropped_over_cap += 1;
            continue;
        }
        rule.host = host;
        additions.push(rule);
    }
    stats.added = additions.len();

    let mut rules = base.rules.clone();
    rules.extend(additions); // base FIRST → first-match-wins keeps base authoritative
    (
        Some(Whitelist {
            version: base.version,
            default_action: base.default_action.clone(),
            rules,
        }),
        stats,
    )
}

/// Merge `raw` additions onto BASE and atomically swap in the result. Returns
/// the merge stats, or `None` when the prior whitelist was kept (unparseable
/// non-empty input). Shared by the runtime poll callback and the launch-time
/// predefine.
fn swap_with_additions(raw: &str) -> (bool, MergeStats) {
    let (merged, stats) = merge_whitelist_additions(base(), raw, MAX_WHITELIST_ADDITIONS);
    match merged {
        Some(wl) => {
            cell().store(Arc::new(wl));
            // Re-evaluate connections suppressed under default_action="ignore"
            // so a newly-added host meters their NEXT request — hot-reload on an
            // already-open keep-alive connection, not just new ones.
            crate::state::rearm_whitelist_suppressions();
            (true, stats)
        }
        None => (false, stats),
    }
}

/// Poll-callback target for a `_snug_whitelist` User-Property change (registered
/// in `init.rs`). Re-merges the additions onto BASE, swaps atomically, and emits
/// a one-line `[SNUG-CALL]` NOTICE (or a `whitelist_reload_failed` diagnostic
/// when the value was unparseable and the prior whitelist was kept). Runs on the
/// poll thread, so it uses `meter::emit` like the call-history setter. Affects
/// only NEW connections — a connection's match decision is cached on its first
/// write and not re-evaluated.
pub fn apply_whitelist_additions(raw: &str) {
    let (applied, stats) = swap_with_additions(raw);
    if applied {
        emit_notice(stats.summary());
    } else {
        // Keep the prior whitelist; tell the operator WHY (human-readable) and
        // also emit the machine-readable diagnostic.
        emit_notice(format!(
            "whitelist: ignored update ({}); kept previous",
            stats.reason
        ));
        emit_reload_failed();
    }
}

fn emit_notice(text: String) {
    crate::meter::emit(Event::CallHistoryNotice {
        ts_ms: Event::now_ts_ms(),
        text,
    });
}

fn emit_reload_failed() {
    crate::meter::emit(Event::ShimDiagnostic {
        ts_ms: Event::now_ts_ms(),
        kind_detail: "whitelist_reload_failed".to_string(),
        conn_id: None,
        dropped_events: None,
        host: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_rule() {
        let json = r#"{
            "version": 1,
            "default_action": "meter",
            "rules": [
                {"host": "api.anthropic.com", "path_prefix": "/v1/",
                 "meter": true, "debug": false, "inject_headers": true,
                 "tokenizer": "claude"}
            ]
        }"#;
        // `meter` and `debug` are not v1 rule fields; the deserializer must
        // tolerate (ignore) unknown keys so configs carrying them still parse.
        let wl: Whitelist = serde_json::from_str(json).unwrap();
        assert_eq!(wl.version, 1);
        assert_eq!(wl.rules.len(), 1);
        assert_eq!(wl.rules[0].host, "api.anthropic.com");
        assert!(wl.rules[0].inject_headers);
        assert_eq!(wl.rules[0].tokenizer, "claude");
    }

    #[test]
    fn rule_defaults_apply_when_fields_omitted() {
        let json = r#"{
            "version": 1,
            "default_action": "ignore",
            "rules": [{"host": "x.example"}]
        }"#;
        let wl: Whitelist = serde_json::from_str(json).unwrap();
        assert_eq!(wl.rules[0].path_prefix, "/");
        assert!(!wl.rules[0].inject_headers);
        assert_eq!(wl.rules[0].tokenizer, "approx");
    }

    #[test]
    fn matches_exact_host_and_prefix_path() {
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![WhitelistRule {
                path_prefix: "/v1/".into(),
                inject_headers: true,
                ..rule_from_host("api.anthropic.com")
            }],
        };
        assert!(wl.matches("api.anthropic.com", "/v1/messages").is_some());
        assert!(wl.matches("API.anthropic.com", "/v1/messages").is_some());
        assert!(wl.matches("api.anthropic.com", "/v2/messages").is_none());
        assert!(wl.matches("other.example", "/v1/").is_none());
    }

    // --- host wildcard matching ----------------------------------------

    #[test]
    fn host_matches_exact_is_case_insensitive() {
        assert!(host_matches("api.anthropic.com", "api.anthropic.com"));
        assert!(host_matches("API.Anthropic.COM", "api.anthropic.com"));
        assert!(!host_matches("api.anthropic.com", "api.openai.com"));
    }

    #[test]
    fn host_matches_suffix_wildcard() {
        // Leading '*' => host must END WITH the literal remainder.
        assert!(host_matches("*.anthropic.com", "api.anthropic.com"));
        assert!(host_matches("*.anthropic.com", "eu.anthropic.com"));
        assert!(host_matches("*.ANTHROPIC.com", "api.anthropic.com")); // case-insensitive
        // The apex lacks the leading dot, so it doesn't match `*.anthropic.com`.
        assert!(!host_matches("*.anthropic.com", "anthropic.com"));
        assert!(!host_matches("*.anthropic.com", "api.openai.com"));
    }

    #[test]
    fn host_matches_prefix_wildcard() {
        // Trailing '*' => host must START WITH the literal remainder.
        assert!(host_matches("api.anthropic.*", "api.anthropic.com"));
        assert!(host_matches("api.anthropic.*", "api.anthropic.ai"));
        assert!(!host_matches("api.anthropic.*", "eu.anthropic.com"));
    }

    #[test]
    fn host_matches_substring_wildcard() {
        // Both ends => substring (contains).
        assert!(host_matches("*anthropic*", "api.anthropic.com"));
        assert!(host_matches("*anthropic*", "anthropic.ai"));
        assert!(!host_matches("*anthropic*", "api.openai.com"));
    }

    #[test]
    fn host_matches_bare_star_matches_any() {
        assert!(host_matches("*", "api.anthropic.com"));
        assert!(host_matches("*", "anything.example"));
        assert!(host_matches("**", "anything.example"));
    }

    #[test]
    fn host_matches_middle_star_is_literal() {
        // A '*' that isn't on a boundary is literal, so it never matches a real
        // DNS host (only the literal pattern itself).
        assert!(!host_matches("api.*.com", "api.anthropic.com"));
        assert!(host_matches("api.*.com", "api.*.com"));
    }

    #[test]
    fn host_matches_suffix_without_dot_is_broad() {
        // Documented footgun, locked in: a suffix wildcard without a leading dot
        // matches more than intended (write `*.anthropic.com` to avoid it).
        assert!(host_matches("*anthropic.com", "api.anthropic.com"));
        assert!(host_matches("*anthropic.com", "evilanthropic.com"));
    }

    #[test]
    fn matches_wildcard_rule_respects_path_prefix() {
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![WhitelistRule {
                path_prefix: "/v1/".into(),
                inject_headers: true,
                tokenizer: "claude".into(),
                ..rule_from_host("*.anthropic.com")
            }],
        };
        // The wildcard host AND the path prefix both have to hold.
        assert!(wl.matches("api.anthropic.com", "/v1/messages").is_some());
        assert!(wl.matches("eu.anthropic.com", "/v1/messages").is_some());
        assert!(wl.matches("api.anthropic.com", "/v2/messages").is_none());
        assert!(wl.matches("api.openai.com", "/v1/messages").is_none());
    }

    #[test]
    fn first_matching_rule_wins() {
        let wl = Whitelist {
            version: 1,
            default_action: "meter".into(),
            rules: vec![
                WhitelistRule {
                    path_prefix: "/api/".into(),
                    ..rule_from_host("x.com")
                },
                WhitelistRule {
                    inject_headers: true,
                    ..rule_from_host("x.com")
                },
            ],
        };
        let m = wl.matches("x.com", "/api/foo").unwrap();
        assert!(!m.inject_headers);
        let m = wl.matches("x.com", "/dashboard").unwrap();
        assert!(m.inject_headers);
    }

    #[test]
    fn no_rules_yields_no_matches() {
        let wl = Whitelist::empty();
        assert!(wl.matches("anything", "/path").is_none());
    }

    #[test]
    fn load_from_invalid_b64_yields_empty() {
        let wl = Whitelist::load_from_b64("!!!not-valid-base64!!!");
        assert_eq!(wl.rules.len(), 0);
    }

    #[test]
    fn load_from_b64_of_non_json_yields_empty() {
        let b64 = base64::engine::general_purpose::STANDARD.encode("this is not json");
        let wl = Whitelist::load_from_b64(&b64);
        assert_eq!(wl.rules.len(), 0);
    }

    #[test]
    fn load_from_b64_valid_json_parses() {
        let json = r#"{"version":1,"default_action":"meter","rules":[
            {"host":"api.anthropic.com","path_prefix":"/v1/",
             "inject_headers":true,"tokenizer":"claude"}
        ]}"#;
        let b64 = base64::engine::general_purpose::STANDARD.encode(json);
        let wl = Whitelist::load_from_b64(&b64);
        assert_eq!(wl.rules.len(), 1);
        assert_eq!(wl.rules[0].host, "api.anthropic.com");
        assert!(wl.rules[0].inject_headers);
        assert_eq!(wl.rules[0].tokenizer, "claude");
    }

    #[test]
    fn load_from_b64_distinct_payloads() {
        // Two different encoded whitelists decode to their respective
        // rules (distinct payloads load to distinct content).
        let v1 = base64::engine::general_purpose::STANDARD.encode(
            r#"{"version":1,"default_action":"meter","rules":[
                {"host":"reload-v1.example","inject_headers":false}
            ]}"#,
        );
        let wl1 = Whitelist::load_from_b64(&v1);
        assert_eq!(wl1.rules.len(), 1);
        assert_eq!(wl1.rules[0].host, "reload-v1.example");
        assert!(!wl1.rules[0].inject_headers);

        let v2 = base64::engine::general_purpose::STANDARD.encode(
            r#"{"version":1,"default_action":"meter","rules":[
                {"host":"reload-v2.example","inject_headers":true},
                {"host":"reload-v3.example"}
            ]}"#,
        );
        let wl2 = Whitelist::load_from_b64(&v2);
        assert_eq!(wl2.rules.len(), 2);
        assert!(wl2.rules[0].inject_headers);
        assert_eq!(wl2.rules[1].host, "reload-v3.example");
    }

    // --- additive hot-reload merge (merge_whitelist_additions) ----------

    /// A base whitelist with `default_action` and the given metering-only hosts.
    fn base_wl(default_action: &str, hosts: &[&str]) -> Whitelist {
        Whitelist {
            version: 1,
            default_action: default_action.into(),
            rules: hosts.iter().map(|h| rule_from_host(h)).collect(),
        }
    }

    #[test]
    fn shorthand_splits_on_comma_space_newline_with_defaults() {
        let (merged, stats) =
            merge_whitelist_additions(&Whitelist::empty(), "a.com, b.com\nc.com d.com", 50);
        let wl = merged.unwrap();
        assert_eq!(wl.rules.len(), 4);
        assert_eq!(stats.added, 4);
        // shorthand rules carry default fields
        let r = &wl.rules[0];
        assert_eq!(r.host, "a.com");
        assert_eq!(r.path_prefix, "/");
        assert!(!r.inject_headers);
        assert_eq!(r.tokenizer, "approx");
    }

    #[test]
    fn shorthand_dedups_case_insensitively() {
        let (merged, stats) =
            merge_whitelist_additions(&Whitelist::empty(), "Api.X.com, api.x.com", 50);
        assert_eq!(merged.unwrap().rules.len(), 1);
        assert_eq!(stats.added, 1);
    }

    #[test]
    fn empty_or_separator_only_input_yields_base_only_and_clears() {
        let base = base_wl("ignore", &["base.com"]);
        for raw in ["", "   ", ",,", " , \n "] {
            let (merged, stats) = merge_whitelist_additions(&base, raw, 50);
            let wl = merged.unwrap();
            assert_eq!(wl.rules.len(), 1, "raw={raw:?}");
            assert_eq!(wl.rules[0].host, "base.com");
            assert!(stats.cleared, "raw={raw:?}");
        }
    }

    #[test]
    fn additions_are_appended_after_base_rules() {
        let base = base_wl("ignore", &["base.com"]);
        let (merged, _) = merge_whitelist_additions(&base, "add.com", 50);
        let wl = merged.unwrap();
        assert_eq!(wl.rules[0].host, "base.com"); // base first → base wins on overlap
        assert_eq!(wl.rules.last().unwrap().host, "add.com");
    }

    #[test]
    fn json_array_preserves_full_fields() {
        let raw = r#"[{"host":"x.com","inject_headers":true,"tokenizer":"cl100k","path_prefix":"/v1/"}]"#;
        let (merged, stats) = merge_whitelist_additions(&Whitelist::empty(), raw, 50);
        let wl = merged.unwrap();
        assert_eq!(stats.added, 1);
        assert!(wl.rules[0].inject_headers);
        assert_eq!(wl.rules[0].tokenizer, "cl100k");
        assert_eq!(wl.rules[0].path_prefix, "/v1/");
    }

    #[test]
    fn json_parse_error_keeps_prior() {
        // Mirrors the operator typo `"monitor:true"` (a string, not `"x":true`):
        // malformed JSON → keep prior, with a human reason for the notice.
        let (merged, stats) =
            merge_whitelist_additions(&Whitelist::empty(), "[not valid json", 50);
        assert!(merged.is_none());
        assert_eq!(stats.reason, "invalid json");
    }

    #[test]
    fn all_invalid_shorthand_keeps_prior_partial_applies() {
        // Every token invalid → keep prior (None).
        let (merged, stats) =
            merge_whitelist_additions(&Whitelist::empty(), "foo/bar baz/qux", 50);
        assert!(merged.is_none());
        assert_eq!(stats.dropped_invalid, 2);
        assert_eq!(stats.reason, "no valid hosts");
        // Mixed → good ones applied, bad ones dropped.
        let (merged, stats) =
            merge_whitelist_additions(&Whitelist::empty(), "good.com, bad/path, also.com", 50);
        let wl = merged.unwrap();
        assert_eq!(stats.added, 2);
        assert_eq!(stats.dropped_invalid, 1);
        assert_eq!(wl.rules.len(), 2);
    }

    #[test]
    fn addition_covered_by_base_is_dropped() {
        // exact
        let (m, s) = merge_whitelist_additions(&base_wl("ignore", &["api.x.com"]), "api.x.com", 50);
        assert_eq!(s.dropped_covered, 1);
        assert_eq!(s.added, 0);
        assert_eq!(m.unwrap().rules.len(), 1);
        // suffix wildcard base covers a concrete added host
        let (_, s) =
            merge_whitelist_additions(&base_wl("ignore", &["*.acme.com"]), "api.acme.com", 50);
        assert_eq!(s.dropped_covered, 1);
        // base "*" (match-any) drops all additions
        let (_, s) = merge_whitelist_additions(&base_wl("meter", &["*"]), "a.com, b.com", 50);
        assert_eq!(s.dropped_covered, 2);
        assert_eq!(s.added, 0);
    }

    #[test]
    fn task_cannot_override_an_admin_host() {
        // base rule is metering-only; the task tries to re-add it with injection.
        let base = base_wl("meter", &["api.anthropic.com"]);
        let raw = r#"[{"host":"api.anthropic.com","inject_headers":true,"tokenizer":"claude"}]"#;
        let (merged, stats) = merge_whitelist_additions(&base, raw, 50);
        let wl = merged.unwrap();
        assert_eq!(stats.dropped_covered, 1);
        assert_eq!(wl.rules.len(), 1);
        // the admin rule is untouched: still metering-only.
        assert!(!wl.rules[0].inject_headers);
        assert_eq!(wl.rules[0].tokenizer, "approx");
    }

    #[test]
    fn version_and_default_action_always_from_base() {
        let mut base = base_wl("ignore", &["base.com"]);
        base.version = 1;
        let (merged, _) = merge_whitelist_additions(&base, "add.com", 50);
        let wl = merged.unwrap();
        assert_eq!(wl.default_action, "ignore");
        assert_eq!(wl.version, 1);
    }

    #[test]
    fn cap_limits_surviving_additions() {
        let raw = "h0.com,h1.com,h2.com,h3.com,h4.com";
        let (merged, stats) = merge_whitelist_additions(&Whitelist::empty(), raw, 3);
        let wl = merged.unwrap();
        assert_eq!(stats.added, 3);
        assert_eq!(stats.dropped_over_cap, 2);
        assert_eq!(wl.rules.len(), 3);
    }

    #[test]
    fn oversized_input_keeps_prior() {
        let big = "x.com,".repeat(4000); // > 16 KiB
        let (merged, stats) = merge_whitelist_additions(&Whitelist::empty(), &big, 50);
        assert!(merged.is_none());
        assert_eq!(stats.reason, "oversized");
    }

    #[test]
    fn added_wildcard_is_kept_metering_only() {
        let (merged, stats) = merge_whitelist_additions(&Whitelist::empty(), "*.evil.com", 50);
        let wl = merged.unwrap();
        assert_eq!(stats.added, 1);
        assert_eq!(wl.rules[0].host, "*.evil.com");
        assert!(!wl.rules[0].inject_headers); // shorthand forces metering-only
    }

    #[test]
    fn merged_whitelist_matches_added_host_on_hot_path() {
        let (merged, _) = merge_whitelist_additions(&Whitelist::empty(), "added.com", 50);
        let wl = merged.unwrap();
        assert!(wl.matches("added.com", "/anything").is_some());
        assert!(wl.matches("ADDED.com", "/v1/x").is_some()); // case-insensitive
    }

    // --- new estimate-path rule fields (read by the proxy) ------------------

    #[test]
    fn new_rule_fields_default_off() {
        // A rule that omits the estimate fields deserializes with them off/empty,
        // so an existing whitelist keeps its meaning (no estimation, no provider
        // override) — the backward-compat contract for the schema-v1 additions.
        let wl: Whitelist = serde_json::from_str(
            r#"{"version":1,"rules":[{"host":"api.anthropic.com"}]}"#,
        )
        .unwrap();
        let r = &wl.rules[0];
        assert!(!r.estimate_unmeasured);
        assert!(r.completion_path.is_empty());
        assert!(r.provider.is_empty());
        assert!(!r.is_completion(Some("POST"), Some("/v1/messages")));
    }

    #[test]
    fn is_completion_matches_only_post_completion_path() {
        let wl: Whitelist = serde_json::from_str(
            r#"{"version":1,"rules":[{"host":"claude.ai","estimate_unmeasured":true,"completion_path":"*/completion"}]}"#,
        )
        .unwrap();
        let r = &wl.rules[0];
        assert!(r.estimate_unmeasured);
        let load = Some("/api/organizations/o/chat_conversations/c?tree=True&rendering_mode=messages");
        let comp = Some("/api/organizations/o/chat_conversations/c/completion");
        assert!(r.is_completion(Some("POST"), comp));
        assert!(r.is_completion(Some("POST"), Some("/x/completion?stream=true")), "query stripped");
        assert!(!r.is_completion(Some("GET"), comp), "GET is never a completion");
        assert!(!r.is_completion(Some("POST"), load), "history-load path is not a completion");
        assert!(!r.is_completion(None, None), "undecodable request fails closed");
        assert!(!r.is_completion(Some("POST"), None));
    }

    #[test]
    fn is_completion_empty_pattern_never_matches() {
        // A rule without a completion_path (the default) is never a completion,
        // even on a POST — so a host that doesn't declare one is never estimated.
        let wl: Whitelist = serde_json::from_str(
            r#"{"version":1,"rules":[{"host":"api.anthropic.com","estimate_unmeasured":true}]}"#,
        )
        .unwrap();
        assert!(!wl.rules[0].is_completion(Some("POST"), Some("/v1/messages/completion")));
    }

    #[test]
    fn provider_hint_resolves_host_ignoring_path() {
        let wl: Whitelist = serde_json::from_str(
            r#"{"version":1,"rules":[
                {"host":"claude.ai","provider":"anthropic","path_prefix":"/api/"},
                {"host":"example.com"}
            ]}"#,
        )
        .unwrap();
        // provider hint is host-level: it resolves even though "/" is not under
        // the rule's "/api/" path_prefix.
        assert_eq!(wl.provider_hint("claude.ai"), "anthropic");
        assert_eq!(wl.provider_hint("CLAUDE.AI"), "anthropic", "case-insensitive host");
        assert_eq!(wl.provider_hint("example.com"), "", "rule without a provider field");
        assert_eq!(wl.provider_hint("unknown.com"), "", "no matching rule");
    }
}
