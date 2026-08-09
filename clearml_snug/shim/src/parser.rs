//! HTTP/1.x request-line parser + HTTP/2 connection-preface detector.
//!
//! Called at most once per connection - the outcome is cached in the
//! ConnectionState so subsequent writes don't re-enter the parser. That
//! caps the per-hook allocation cost (`httparse` allocates the header
//! array on the stack, but we still copy method/host/path into owned
//! `String`s).

/// The HTTP/2 connection preface (RFC 7540 section 3.5). Sent by the client
/// immediately after the TLS handshake on every HTTP/2 connection. Acts as
/// a sentinel so we don't try to parse the binary frames that follow as
/// HTTP/1.x.
const H2_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Maximum number of headers we'll parse on the first write. 64 is well
/// above what realistic HTTP/1.x clients send and keeps the stack frame
/// bounded.
const MAX_HEADERS: usize = 64;

/// What the parser decided about the first write on a connection.
#[derive(Debug, PartialEq, Eq)]
pub enum ParseOutcome {
    /// Successfully parsed HTTP/1.x request line + headers.
    Http1 {
        method: String,
        host: String,
        path: String,
    },
    /// Caller is speaking HTTP/2. HPACK is not parsed.
    Http2,
    /// Headers were partial - more bytes needed. We don't retry on
    /// subsequent writes; the connection just lives as "not parsed"
    /// and only byte-counted.
    Incomplete,
    /// Doesn't look like HTTP at all (gRPC binary frames, custom protocols
    /// over TLS, etc.).
    NotHttp,
}

pub fn parse_first_write(buf: &[u8]) -> ParseOutcome {
    if buf.starts_with(H2_PREFACE) {
        return ParseOutcome::Http2;
    }

    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers);
    match req.parse(buf) {
        Ok(httparse::Status::Complete(_)) => {
            let method = req.method.unwrap_or("").to_string();
            let path = req.path.unwrap_or("").to_string();
            let host = req
                .headers
                .iter()
                .find(|h| h.name.eq_ignore_ascii_case("host"))
                .and_then(|h| std::str::from_utf8(h.value).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            ParseOutcome::Http1 { method, host, path }
        }
        Ok(httparse::Status::Partial) => ParseOutcome::Incomplete,
        Err(_) => ParseOutcome::NotHttp,
    }
}

/// What we extract from an HTTP/1.x response head (status line + headers).
/// Just the bits the usage scanner needs: the status code, whether the body
/// is an SSE stream vs a JSON document, and whether it's chunked.
#[derive(Debug, PartialEq, Eq)]
pub struct ResponseHead {
    pub status: u16,
    pub is_event_stream: bool,
    pub is_json: bool,
    pub chunked: bool,
}

/// Parse an HTTP/1.x response head. `buf` must contain the full head
/// (through the `\r\n\r\n` terminator); the caller accumulates read chunks
/// until that boundary appears, then passes the head slice here. Returns
/// `None` if it isn't a complete, parseable HTTP response head.
pub fn parse_response_head(buf: &[u8]) -> Option<ResponseHead> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut resp = httparse::Response::new(&mut headers);
    match resp.parse(buf) {
        Ok(httparse::Status::Complete(_)) => {
            let status = resp.code?;
            let mut is_event_stream = false;
            let mut is_json = false;
            let mut chunked = false;
            for h in resp.headers.iter() {
                if h.name.eq_ignore_ascii_case("content-type") {
                    let v = h.value.to_ascii_lowercase();
                    if bytes_contains(&v, b"text/event-stream") {
                        is_event_stream = true;
                    }
                    if bytes_contains(&v, b"application/json") {
                        is_json = true;
                    }
                } else if h.name.eq_ignore_ascii_case("transfer-encoding") {
                    let v = h.value.to_ascii_lowercase();
                    if bytes_contains(&v, b"chunked") {
                        chunked = true;
                    }
                }
            }
            Some(ResponseHead {
                status,
                is_event_stream,
                is_json,
                chunked,
            })
        }
        _ => None,
    }
}

/// Substring search for byte slices. `needle` is always a non-empty literal
/// at the call sites, so the `windows` call never sees a zero size.
fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_http1_get() {
        let buf = b"GET /v1/messages HTTP/1.1\r\n\
                    Host: api.anthropic.com\r\n\
                    User-Agent: test\r\n\r\n";
        match parse_first_write(buf) {
            ParseOutcome::Http1 { method, host, path } => {
                assert_eq!(method, "GET");
                assert_eq!(host, "api.anthropic.com");
                assert_eq!(path, "/v1/messages");
            }
            other => panic!("expected Http1, got {:?}", other),
        }
    }

    #[test]
    fn parses_post_with_body_in_same_buffer() {
        // httparse only parses the header section; the body following the
        // \r\n\r\n boundary is ignored. We just need method/host/path.
        let buf = b"POST /v1/messages HTTP/1.1\r\n\
                    Host: api.anthropic.com\r\n\
                    Content-Length: 17\r\n\r\n\
                    {\"model\":\"opus\"}\n";
        match parse_first_write(buf) {
            ParseOutcome::Http1 { method, host, path } => {
                assert_eq!(method, "POST");
                assert_eq!(host, "api.anthropic.com");
                assert_eq!(path, "/v1/messages");
            }
            other => panic!("expected Http1, got {:?}", other),
        }
    }

    #[test]
    fn detects_http2_preface() {
        // Real H2 preface is followed by SETTINGS frame; doesn't matter
        // here - we only need the prefix match.
        let buf = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x18\x04\x00\x00\x00\x00\x00";
        assert_eq!(parse_first_write(buf), ParseOutcome::Http2);
    }

    #[test]
    fn partial_headers_yields_incomplete() {
        let buf = b"GET /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n";
        assert_eq!(parse_first_write(buf), ParseOutcome::Incomplete);
    }

    #[test]
    fn garbage_yields_not_http() {
        let buf = b"\x00\x01\x02\x03 binary nonsense";
        assert_eq!(parse_first_write(buf), ParseOutcome::NotHttp);
    }

    #[test]
    fn host_header_is_case_insensitive() {
        let buf = b"POST / HTTP/1.1\r\nHOST: example.com\r\nContent-Length: 0\r\n\r\n";
        match parse_first_write(buf) {
            ParseOutcome::Http1 { host, .. } => assert_eq!(host, "example.com"),
            other => panic!("expected Http1, got {:?}", other),
        }
    }

    #[test]
    fn host_header_trimmed() {
        let buf = b"GET / HTTP/1.1\r\nHost:    api.anthropic.com   \r\n\r\n";
        match parse_first_write(buf) {
            ParseOutcome::Http1 { host, .. } => assert_eq!(host, "api.anthropic.com"),
            other => panic!("expected Http1, got {:?}", other),
        }
    }

    #[test]
    fn missing_host_header_yields_empty_string() {
        // HTTP/1.0 didn't require Host; some embedded clients omit it too.
        // We don't fail; we just record an empty host.
        let buf = b"GET / HTTP/1.0\r\n\r\n";
        match parse_first_write(buf) {
            ParseOutcome::Http1 { host, method, path } => {
                assert_eq!(host, "");
                assert_eq!(method, "GET");
                assert_eq!(path, "/");
            }
            other => panic!("expected Http1, got {:?}", other),
        }
    }

    // --- Response head parsing -----------------------------------------

    #[test]
    fn response_head_sse_chunked() {
        let buf = b"HTTP/1.1 200 OK\r\n\
                    Content-Type: text/event-stream; charset=utf-8\r\n\
                    Transfer-Encoding: chunked\r\n\r\n";
        let h = parse_response_head(buf).expect("should parse");
        assert_eq!(h.status, 200);
        assert!(h.is_event_stream);
        assert!(!h.is_json);
        assert!(h.chunked);
    }

    #[test]
    fn response_head_json_content_length() {
        let buf = b"HTTP/1.1 200 OK\r\n\
                    Content-Type: application/json\r\n\
                    Content-Length: 1234\r\n\r\n";
        let h = parse_response_head(buf).expect("should parse");
        assert_eq!(h.status, 200);
        assert!(h.is_json);
        assert!(!h.is_event_stream);
        assert!(!h.chunked);
    }

    #[test]
    fn response_head_error_status_parsed() {
        let buf =
            b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n\r\n";
        let h = parse_response_head(buf).expect("should parse");
        assert_eq!(h.status, 429);
    }

    #[test]
    fn response_head_incomplete_returns_none() {
        let buf = b"HTTP/1.1 200 OK\r\nContent-Type: app";
        assert!(parse_response_head(buf).is_none());
    }
}
