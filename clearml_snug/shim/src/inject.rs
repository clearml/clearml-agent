//! HTTP/1.x header rewriting for outbound requests.
//!
//! Called by `state::observe_write` on the first write per connection when the
//! matched whitelist rule says `inject_headers: true` (splice `project:` /
//! `session:`) and/or usage parsing is on for a whitelisted host (force
//! `accept-encoding: identity` so the response body is readable). Builds a new
//! buffer with the header section adjusted immediately before the empty-line
//! terminator.
//!
//! Hot-path note: each call allocates one `Vec<u8>` of size
//! `original + ~96`. This only fires once per connection (state machine
//! caches the parse outcome), so allocation cost is dwarfed by the TLS
//! handshake + httparse parse already in the path. Pre-allocated
//! per-connection scratch buffers could be added as a future
//! optimization if needed.
//!
//! **Critical contract**: callers of this module must return the *original*
//! `num` to the libssl caller, not the spliced buffer's length. Reporting
//! the larger length triggers assertion failures in libssl's higher layers.

/// Rewrite the request header block, doing any of:
///   * `inject_ids`: splice `project:`/`session:` headers (omitting either id
///     that's empty).
///   * `force_identity`: strip any existing `Accept-Encoding` header and add
///     `accept-encoding: identity`, so the LLM response comes back
///     uncompressed and the body scanner can read provider `usage` without a
///     decompressor in-process. A *duplicate* Accept-Encoding would be
///     comma-combined by the server per RFC 7230 (e.g. `gzip, identity`) and
///     gzip could still win, so the original must be removed, not appended to.
///
/// Returns `None` when neither action is requested or when the header
/// terminator (`\r\n\r\n`) isn't present (partial header write). The new
/// buffer may be longer OR shorter than the input; callers must still report
/// the *original* `num` back to libssl (see the module-level contract).
pub fn rewrite_headers(
    buf: &[u8],
    project_id: &str,
    task_id: &str,
    inject_ids: bool,
    force_identity: bool,
) -> Option<Vec<u8>> {
    if !inject_ids && !force_identity {
        return None;
    }
    let eoh = find_eoh(buf)?;
    // `eoh` points to the start of `\r\n\r\n`. The header section is
    // `buf[..eoh+2]` (request line + headers, ending in the last header's
    // `\r\n`); `buf[eoh+2..]` is the empty-line terminator + body. New
    // headers slot in at `eoh + 2`.
    let insert_at = eoh + 2;

    let mut out = Vec::with_capacity(buf.len() + 96);
    if force_identity {
        copy_headers_stripping_accept_encoding(&mut out, &buf[..insert_at]);
    } else {
        out.extend_from_slice(&buf[..insert_at]);
    }

    if inject_ids && !project_id.is_empty() {
        out.extend_from_slice(b"project: ");
        out.extend_from_slice(project_id.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    if inject_ids && !task_id.is_empty() {
        out.extend_from_slice(b"session: ");
        out.extend_from_slice(task_id.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    if force_identity {
        out.extend_from_slice(b"accept-encoding: identity\r\n");
    }

    out.extend_from_slice(&buf[insert_at..]);
    Some(out)
}

/// Copy the request line + headers from `head` (which ends at the last
/// header's `\r\n`, NOT including the empty-line terminator) into `out`,
/// dropping any `Accept-Encoding` header line.
fn copy_headers_stripping_accept_encoding(out: &mut Vec<u8>, head: &[u8]) {
    let mut start = 0;
    while start < head.len() {
        let end = match head[start..].windows(2).position(|w| w == b"\r\n") {
            Some(p) => start + p,
            None => head.len(),
        };
        let line = &head[start..end];
        if !is_accept_encoding(line) {
            out.extend_from_slice(line);
            out.extend_from_slice(b"\r\n");
        }
        start = (end + 2).min(head.len());
    }
}

/// True if `line` is an `Accept-Encoding` header (name compared
/// case-insensitively). The request line has no `:` before its first space-
/// delimited token... actually it may (e.g. `CONNECT host:443`), but that
/// name part won't equal `accept-encoding`, so it's never stripped.
fn is_accept_encoding(line: &[u8]) -> bool {
    match line.iter().position(|&b| b == b':') {
        Some(colon) => line[..colon].eq_ignore_ascii_case(b"accept-encoding"),
        None => false,
    }
}

/// Index of the *first* `\r\n\r\n` in `buf`, or None if absent. Windowing
/// is essential here: a body containing `\r\n\r\n` later in the buffer
/// must not be mistaken for the header terminator. `windows(4).position(...)`
/// returns the earliest match by construction, so we're safe.
fn find_eoh(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 3-arg inject helper for these tests; production code calls
    // `rewrite_headers` directly: project/session splice, no
    // Accept-Encoding rewrite.
    fn splice_headers(buf: &[u8], project_id: &str, task_id: &str) -> Option<Vec<u8>> {
        rewrite_headers(buf, project_id, task_id, true, false)
    }

    #[test]
    fn splices_both_headers_at_correct_position() {
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\n\r\nbody";
        let out = splice_headers(buf, "proj-1", "task-1").unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        assert_eq!(
            s,
            "GET / HTTP/1.1\r\nHost: x\r\nproject: proj-1\r\nsession: task-1\r\n\r\nbody"
        );
    }

    #[test]
    fn first_crlfcrlf_only_not_one_in_body() {
        // Body contains a stray \r\n\r\n. The splice must NOT inject into
        // it; only the header terminator counts.
        let buf = b"POST / HTTP/1.1\r\nHost: x\r\n\r\nbody with \r\n\r\n inside";
        let out = splice_headers(buf, "p", "t").unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        // Headers were augmented, body bytes still appear verbatim.
        assert!(s.contains("project: p\r\nsession: t\r\n\r\nbody with"));
        assert!(s.contains("\r\n\r\n inside"));
    }

    #[test]
    fn returns_none_when_no_eoh_present() {
        // Partial header write - shouldn't happen in practice (httparse
        // would have classified this as Incomplete and we wouldn't reach
        // the splice path), but defensively None.
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\n";
        assert!(splice_headers(buf, "p", "t").is_none());
    }

    #[test]
    fn omits_project_header_when_id_empty() {
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let out = splice_headers(buf, "", "task-1").unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        assert!(!s.contains("project:"));
        assert!(s.contains("session: task-1"));
    }

    #[test]
    fn omits_session_header_when_id_empty() {
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let out = splice_headers(buf, "proj", "").unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("project: proj"));
        assert!(!s.contains("session:"));
    }

    #[test]
    fn returns_unchanged_when_both_ids_empty() {
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\n\r\nbody";
        let out = splice_headers(buf, "", "").unwrap();
        // No new bytes added; the output is byte-identical to the input
        // (just heap-copied through our Vec).
        assert_eq!(out, buf);
    }

    #[test]
    fn output_length_is_predictable_for_byte_counting() {
        // The caller (state.rs) and the hook (openssl.rs) need to be
        // able to reason about this: out.len() == buf.len() + sum of
        // ("project: <id>\r\n" + "session: <id>\r\n") that fired.
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let out = splice_headers(buf, "P", "T").unwrap();
        let added_project = "project: P\r\n".len();
        let added_session = "session: T\r\n".len();
        assert_eq!(out.len(), buf.len() + added_project + added_session);
    }

    // --- force-identity rewrite ----------------------------------------

    #[test]
    fn force_identity_strips_existing_accept_encoding() {
        let buf = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\
                    Accept-Encoding: gzip, deflate, br\r\nContent-Length: 5\r\n\r\nhello";
        let out = rewrite_headers(buf, "", "", false, true).unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        let lower = s.to_ascii_lowercase();
        assert!(!lower.contains("gzip"), "original encoding must be gone: {s}");
        assert!(s.contains("accept-encoding: identity\r\n"));
        assert_eq!(
            lower.matches("accept-encoding:").count(),
            1,
            "exactly one Accept-Encoding header"
        );
        assert!(s.contains("Host: api.anthropic.com\r\n"));
        assert!(s.contains("Content-Length: 5\r\n"));
        assert!(s.ends_with("\r\n\r\nhello"));
        // inject_ids=false -> no project/session
        assert!(!s.contains("project:"));
        assert!(!s.contains("session:"));
    }

    #[test]
    fn force_identity_adds_header_when_absent() {
        let buf = b"POST / HTTP/1.1\r\nHost: x\r\n\r\nbody";
        let out = rewrite_headers(buf, "", "", false, true).unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("accept-encoding: identity\r\n"));
        assert!(s.ends_with("\r\n\r\nbody"));
    }

    #[test]
    fn force_identity_case_insensitive_strip() {
        let buf = b"GET / HTTP/1.1\r\nHost: x\r\nACCEPT-ENCODING: gzip\r\n\r\n";
        let out = rewrite_headers(buf, "", "", false, true).unwrap();
        let lower = String::from_utf8(out).unwrap().to_ascii_lowercase();
        assert_eq!(lower.matches("accept-encoding:").count(), 1);
        assert!(lower.contains("accept-encoding: identity"));
    }

    #[test]
    fn rewrite_combines_ids_and_identity() {
        let buf = b"POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n\
                    Accept-Encoding: gzip\r\n\r\nbody";
        let out = rewrite_headers(buf, "proj", "task", true, true).unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("project: proj\r\n"));
        assert!(s.contains("session: task\r\n"));
        assert!(s.contains("accept-encoding: identity\r\n"));
        assert!(!s.to_ascii_lowercase().contains("gzip"));
    }

    #[test]
    fn rewrite_returns_none_when_nothing_requested() {
        let buf = b"POST / HTTP/1.1\r\nHost: x\r\n\r\nbody";
        assert!(rewrite_headers(buf, "p", "t", false, false).is_none());
    }

    #[test]
    fn rewrite_returns_none_without_header_terminator() {
        let buf = b"POST / HTTP/1.1\r\nHost: x\r\n";
        assert!(rewrite_headers(buf, "", "", false, true).is_none());
    }
}
