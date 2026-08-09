//! Self-host exclusion: the ClearML backend hostnames this task reports to.
//!
//! The task's own ClearML SDK (the `clearml` Python package) talks to the
//! backend over urllib3/OpenSSL, so without this its api/files/web-server calls
//! would be hooked and metered as if they were LLM traffic — inflating the
//! request/byte counters and (when that host is whitelisted) billed as model
//! usage. The reporter's OWN backend traffic is already invisible to the hooks
//! (it uses rustls, not the hooked system OpenSSL); the task's SDK calls are
//! not, which is the gap this closes.
//!
//! The agent resolves the api/files/web server hostnames from the session
//! config and ships them in the handoff descriptor (`self_hosts`). The shim
//! installs them once at ctor (before any hook can fire) and suppresses any
//! connection whose `Host:` header resolves to one of them — regardless of
//! whitelist rules or `default_action`, since a self host is never something we
//! want to bill.
//!
//! Matching is hostname-only (port-stripped, lowercased): ClearML's api/files/
//! web servers frequently differ only by port on a single self-hosted host, and
//! we never want to meter our own backend on any port.

use std::sync::OnceLock;

static SELF_HOSTS: OnceLock<Vec<String>> = OnceLock::new();

/// Install the backend self-hosts (from the agent descriptor). Values are
/// normalized to bare lowercase hostnames; empties are dropped. Idempotent —
/// the first install wins (the ctor calls it once, before any hook can fire).
pub fn install(hosts: Vec<String>) {
    let normalized: Vec<String> = hosts
        .iter()
        .map(|h| normalize(h))
        .filter(|h| !h.is_empty())
        .collect();
    let _ = SELF_HOSTS.set(normalized);
}

/// Snapshot of the installed self-hosts. Empty before install (e.g. the
/// stderr-fallback path with no descriptor), in which case nothing is excluded.
pub fn current() -> &'static [String] {
    SELF_HOSTS.get().map(Vec::as_slice).unwrap_or(&[])
}

/// True iff `host` (a raw `Host:` header value) resolves to one of `self_hosts`.
/// Pure over the injected slice so the state machine stays unit-testable without
/// touching the process-global. `self_hosts` is assumed already normalized (as
/// `install` produces); only the incoming `host` is normalized here, keeping the
/// hot path to one allocation.
pub fn matches(self_hosts: &[String], host: &str) -> bool {
    if self_hosts.is_empty() {
        return false;
    }
    let h = normalize(host);
    !h.is_empty() && self_hosts.iter().any(|s| *s == h)
}

/// Normalize a host (a `Host:` header value or a configured backend host) to a
/// bare lowercase hostname: strip IPv6 brackets and a single trailing `:port`.
pub fn normalize(host: &str) -> String {
    let h = host.trim();
    // Bracketed IPv6 literal, optionally with a port: `[::1]:8008` -> `::1`.
    if let Some(rest) = h.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return rest[..end].to_ascii_lowercase();
        }
    }
    // Bare IPv6 (two or more colons, unbracketed): no `:port` to strip.
    if h.bytes().filter(|&b| b == b':').count() >= 2 {
        return h.to_ascii_lowercase();
    }
    // `hostname[:port]` — strip a single trailing numeric port.
    let bare = match h.rfind(':') {
        Some(idx)
            if !h[idx + 1..].is_empty() && h[idx + 1..].bytes().all(|b| b.is_ascii_digit()) =>
        {
            &h[..idx]
        }
        _ => h,
    };
    bare.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_plain_host_lowercases() {
        assert_eq!(normalize("API.Clear.ML"), "api.clear.ml");
        assert_eq!(normalize("  api.clear.ml  "), "api.clear.ml");
    }

    #[test]
    fn normalize_strips_port() {
        assert_eq!(normalize("localhost:8008"), "localhost");
        assert_eq!(normalize("api.clear.ml:443"), "api.clear.ml");
        assert_eq!(normalize("127.0.0.1:8081"), "127.0.0.1");
    }

    #[test]
    fn normalize_handles_ipv6() {
        assert_eq!(normalize("[::1]:8008"), "::1");
        assert_eq!(normalize("[::1]"), "::1");
        // Bare (unbracketed) IPv6 has 2+ colons and no port to strip.
        assert_eq!(normalize("::1"), "::1");
        assert_eq!(normalize("fe80::1"), "fe80::1");
    }

    #[test]
    fn normalize_leaves_non_numeric_suffix_alone() {
        // A trailing `:something` that isn't a port is not stripped.
        assert_eq!(normalize("host:"), "host:");
    }

    #[test]
    fn empty_self_hosts_never_matches() {
        assert!(!matches(&[], "api.clear.ml"));
        assert!(!matches(&[], ""));
    }

    #[test]
    fn matches_exact_hostname() {
        let hosts = vec!["api.clear.ml".to_string()];
        assert!(matches(&hosts, "api.clear.ml"));
        assert!(matches(&hosts, "API.CLEAR.ML"));
        assert!(!matches(&hosts, "api.anthropic.com"));
        // Not a suffix/substring match — a sibling subdomain doesn't match.
        assert!(!matches(&hosts, "evil-api.clear.ml.attacker.com"));
    }

    #[test]
    fn matches_is_port_insensitive() {
        // self_hosts arrives pre-normalized (install does this); the incoming
        // Host header may still carry a port, which matches() strips.
        let hosts = vec!["localhost".to_string()];
        assert!(matches(&hosts, "localhost:8008"));
        assert!(matches(&hosts, "localhost:8081"));
        // A configured host carrying a port still matches once normalized the
        // way install() would (api 8008 / files 8081 / web 8080 all collapse to
        // the bare host, so any of the three ports matches the rule).
        let installed: Vec<String> = ["localhost:8008"].iter().map(|h| normalize(h)).collect();
        assert!(matches(&installed, "localhost:9999"));
    }

    #[test]
    fn install_normalizes_and_drops_empties() {
        // install() snapshots into the global; current() reads it back. Only
        // assert on what install does to its input via a fresh comparison
        // (the global is process-wide / set-once, so we test the pure pieces
        // through matches() above and the normalization here).
        let raw = vec![
            "  API.Clear.ML:443  ".to_string(),
            "".to_string(),
            "   ".to_string(),
        ];
        let normalized: Vec<String> = raw
            .iter()
            .map(|h| normalize(h))
            .filter(|h| !h.is_empty())
            .collect();
        assert_eq!(normalized, vec!["api.clear.ml".to_string()]);
    }
}
