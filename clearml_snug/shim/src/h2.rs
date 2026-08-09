//! Minimal, synchronous HTTP/2 frame demultiplexer.
//!
//! The shim sees a raw, decrypted h2 byte stream per direction (client→server
//! on `SSL_write`, server→client on `SSL_read`), arriving in arbitrary chunks.
//! This parses the 9-byte frame headers (RFC 7540 §4.1) and hands back complete
//! frames, buffering any partial trailing frame across `feed` calls. It does NOT
//! decode HPACK (that's the `HEADERS`/`CONTINUATION` payload) — the usage path
//! only needs `DATA` payloads, which are the raw request/response body bytes.
//!
//! `state.rs` owns per-`stream_id` request lifecycle; this module is only the
//! framing layer, mirroring the incremental `new`/`feed` shape of
//! `body_scan::RespParse`.

/// The HTTP/2 client connection preface (RFC 7540 §3.5), sent once by the client
/// before its first frame. Same constant as `parser::H2_PREFACE`, duplicated
/// here so the client-side demux can skip it without a cross-module dep.
const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

pub const FRAME_DATA: u8 = 0x0;
// Used by state.rs to detect END_STREAM on a headers-only response.
#[allow(dead_code)]
pub const FRAME_HEADERS: u8 = 0x1;
// Used by state.rs to bootstrap h2 from the server->client (SSL_read) side: an
// h2 connection's first server frame is a SETTINGS frame on stream 0.
pub const FRAME_SETTINGS: u8 = 0x4;
// A header block continues across CONTINUATION frames (RFC 7540 §6.10) until a
// frame carries END_HEADERS; the request HPACK reassembler stitches them back.
#[allow(dead_code)]
pub const FRAME_CONTINUATION: u8 = 0x9;

const FLAG_END_STREAM: u8 = 0x1;
const FLAG_PADDED: u8 = 0x8;
// HEADERS/CONTINUATION: this frame ends the header block (RFC 7540 §6.2, §6.10).
#[allow(dead_code)]
const FLAG_END_HEADERS: u8 = 0x4;
// HEADERS: the payload is prefixed with 5 priority octets to strip (RFC 7540 §6.2).
#[allow(dead_code)]
const FLAG_PRIORITY: u8 = 0x20;

/// A defensive cap on a single frame's declared length. The h2 default
/// SETTINGS_MAX_FRAME_SIZE is 16 KiB and the protocol maximum is 16 MiB; a
/// larger declared length means a desync/garbage stream, so we stop parsing
/// rather than buffer unboundedly.
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

/// One decoded frame. For `DATA` frames `payload` has any padding stripped, so
/// it is exactly the body bytes for `stream_id`. For other types `payload` is
/// the raw frame payload (unused today beyond `HEADERS` END_STREAM detection).
pub struct Frame {
    pub stream_id: u32,
    pub ftype: u8,
    pub flags: u8,
    pub payload: Vec<u8>,
}

impl Frame {
    /// The stream's data ends after this frame (END_STREAM flag). Meaningful on
    /// `DATA` and `HEADERS` frames.
    pub fn end_stream(&self) -> bool {
        self.flags & FLAG_END_STREAM != 0
    }
}

pub struct FrameParser {
    buf: Vec<u8>,
    /// Client side must consume the 24-byte connection preface before frames.
    expect_preface: bool,
    /// Set on an unrecoverable desync (absurd frame length); further feeds are
    /// cheap no-ops so one bad connection can't wedge or balloon memory.
    broken: bool,
}

impl FrameParser {
    /// Client→server side: skips the leading connection preface.
    pub fn new_client() -> Self {
        Self { buf: Vec::new(), expect_preface: true, broken: false }
    }

    /// Server→client side: no preface.
    pub fn new_server() -> Self {
        Self { buf: Vec::new(), expect_preface: false, broken: false }
    }

    /// Feed one decrypted chunk; return every complete frame now available. A
    /// partial trailing frame is buffered for the next call.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<Frame> {
        let mut out = Vec::new();
        if self.broken {
            return out;
        }
        self.buf.extend_from_slice(chunk);

        if self.expect_preface {
            if self.buf.len() < PREFACE.len() {
                // Not enough yet; verify what we have is still a valid prefix.
                if !PREFACE.starts_with(&self.buf[..]) {
                    self.broken = true;
                }
                return out;
            }
            if &self.buf[..PREFACE.len()] != PREFACE {
                self.broken = true;
                return out;
            }
            self.buf.drain(..PREFACE.len());
            self.expect_preface = false;
        }

        loop {
            if self.buf.len() < 9 {
                break;
            }
            let len =
                ((self.buf[0] as usize) << 16) | ((self.buf[1] as usize) << 8) | self.buf[2] as usize;
            if len > MAX_FRAME_LEN {
                self.broken = true;
                self.buf.clear();
                break;
            }
            if self.buf.len() < 9 + len {
                break; // partial frame; wait for more.
            }
            let ftype = self.buf[3];
            let flags = self.buf[4];
            let stream_id = (((self.buf[5] as u32) << 24)
                | ((self.buf[6] as u32) << 16)
                | ((self.buf[7] as u32) << 8)
                | self.buf[8] as u32)
                & 0x7fff_ffff;

            let raw = &self.buf[9..9 + len];
            let payload = if ftype == FRAME_DATA && flags & FLAG_PADDED != 0 {
                // Padded DATA: first octet is Pad Length; that many trailing
                // octets are padding. Both are stripped to leave body bytes.
                if raw.is_empty() {
                    Vec::new()
                } else {
                    let pad = raw[0] as usize;
                    if 1 + pad <= raw.len() {
                        raw[1..raw.len() - pad].to_vec()
                    } else {
                        // Malformed padding — drop the payload but keep framing.
                        Vec::new()
                    }
                }
            } else {
                raw.to_vec()
            };

            out.push(Frame { stream_id, ftype, flags, payload });
            self.buf.drain(..9 + len);
        }
        out
    }
}

/// Strip a `HEADERS` frame payload down to its Header Block Fragment: drop the
/// optional Pad Length octet, the optional 5 priority octets, and any trailing
/// padding (RFC 7540 §6.2). Returns `None` when the declared priority prefix or
/// padding doesn't fit the payload (malformed framing), so the caller can stop
/// decoding rather than hand HPACK a garbage block.
#[allow(dead_code)]
pub fn headers_block_fragment(flags: u8, payload: &[u8]) -> Option<Vec<u8>> {
    let mut rest = payload;
    let mut pad = 0usize;
    if flags & FLAG_PADDED != 0 {
        let (first, tail) = rest.split_first()?;
        pad = *first as usize;
        rest = tail;
    }
    if flags & FLAG_PRIORITY != 0 {
        if rest.len() < 5 {
            return None;
        }
        rest = &rest[5..];
    }
    if pad > rest.len() {
        return None;
    }
    Some(rest[..rest.len() - pad].to_vec())
}

/// The result of feeding one frame to `HeaderBlockAssembler`.
#[allow(dead_code)]
pub enum HeaderBlock {
    /// A complete block: `stream_id` and the concatenated, stripped Header Block
    /// Fragment, ready to hand to an HPACK decoder.
    Complete(u32, Vec<u8>),
    /// Framing couldn't be reassembled (bad padding/priority, a CONTINUATION with
    /// no open block, or a new HEADERS while one was still open). The connection's
    /// HPACK dynamic table can no longer be trusted, so the caller must give up.
    Malformed,
}

/// Reassembles an HPACK header block from a `HEADERS` frame plus any following
/// `CONTINUATION` frames on the same stream (RFC 7540 §6.10). The RFC forbids any
/// other frame interleaving between a `HEADERS` and its `CONTINUATION`s, so a
/// single in-progress block is enough state.
///
/// Feed every frame in arrival order (non-header frames are ignored); a
/// `HeaderBlock` is returned the moment a block's `END_HEADERS` is seen, or as
/// soon as framing is malformed. HPACK's dynamic table is cumulative across the
/// whole connection, so the caller must feed the returned `Complete` blocks to a
/// single decoder, in order, exactly once.
#[allow(dead_code)]
#[derive(Default)]
pub struct HeaderBlockAssembler {
    /// `(stream_id, fragment)` for a block still awaiting its `END_HEADERS` across
    /// CONTINUATION frames; `None` between blocks.
    pending: Option<(u32, Vec<u8>)>,
}

#[allow(dead_code)]
impl HeaderBlockAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one frame. Returns `Some(HeaderBlock)` when a block completes or
    /// framing is malformed, `None` otherwise (a non-header frame, or a
    /// HEADERS/CONTINUATION still awaiting its `END_HEADERS`).
    pub fn feed(&mut self, f: &Frame) -> Option<HeaderBlock> {
        match f.ftype {
            FRAME_HEADERS => {
                // A HEADERS arriving while a prior block is still open violates the
                // framing rule that CONTINUATIONs immediately follow their HEADERS.
                if self.pending.is_some() {
                    self.pending = None;
                    return Some(HeaderBlock::Malformed);
                }
                let fragment = match headers_block_fragment(f.flags, &f.payload) {
                    Some(frag) => frag,
                    None => return Some(HeaderBlock::Malformed),
                };
                if f.flags & FLAG_END_HEADERS != 0 {
                    Some(HeaderBlock::Complete(f.stream_id, fragment))
                } else {
                    self.pending = Some((f.stream_id, fragment));
                    None
                }
            }
            FRAME_CONTINUATION => {
                match &mut self.pending {
                    Some((sid, buf)) if *sid == f.stream_id => {
                        buf.extend_from_slice(&f.payload);
                    }
                    // A CONTINUATION with no matching open block is a framing desync.
                    _ => {
                        self.pending = None;
                        return Some(HeaderBlock::Malformed);
                    }
                }
                if f.flags & FLAG_END_HEADERS != 0 {
                    self.pending.take().map(|(sid, buf)| HeaderBlock::Complete(sid, buf))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_frame(stream_id: u32, end: bool, body: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        let len = body.len();
        f.push((len >> 16) as u8);
        f.push((len >> 8) as u8);
        f.push(len as u8);
        f.push(FRAME_DATA);
        f.push(if end { FLAG_END_STREAM } else { 0 });
        f.extend_from_slice(&stream_id.to_be_bytes());
        f.extend_from_slice(body);
        f
    }

    #[test]
    fn client_skips_preface_then_parses_frames() {
        let mut p = FrameParser::new_client();
        let mut wire = PREFACE.to_vec();
        // SETTINGS frame (type 0x4, stream 0, empty payload).
        wire.extend_from_slice(&[0, 0, 0, 0x4, 0, 0, 0, 0, 0]);
        wire.extend_from_slice(&data_frame(1, false, b"hello"));
        let frames = p.feed(&wire);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].ftype, 0x4);
        assert_eq!(frames[1].ftype, FRAME_DATA);
        assert_eq!(frames[1].stream_id, 1);
        assert_eq!(frames[1].payload, b"hello");
        assert!(!frames[1].end_stream());
    }

    #[test]
    fn reassembles_across_split_chunks() {
        let mut p = FrameParser::new_server();
        let f = data_frame(3, true, b"world!");
        // Split mid-frame across two feeds.
        let mut frames = p.feed(&f[..4]);
        assert!(frames.is_empty());
        frames = p.feed(&f[4..]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].stream_id, 3);
        assert_eq!(frames[0].payload, b"world!");
        assert!(frames[0].end_stream());
    }

    #[test]
    fn strips_data_padding() {
        let mut p = FrameParser::new_server();
        // Padded DATA: pad-len=3, body="ab", 3 pad bytes.
        let body = b"ab";
        let pad = 3usize;
        let mut payload = vec![pad as u8];
        payload.extend_from_slice(body);
        payload.extend_from_slice(&[0, 0, 0]);
        let len = payload.len();
        let mut f = vec![(len >> 16) as u8, (len >> 8) as u8, len as u8, FRAME_DATA, FLAG_PADDED];
        f.extend_from_slice(&5u32.to_be_bytes());
        f.extend_from_slice(&payload);
        let frames = p.feed(&f);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, b"ab");
    }

    #[test]
    fn multiplexes_streams() {
        let mut p = FrameParser::new_server();
        let mut wire = data_frame(1, false, b"aaa");
        wire.extend_from_slice(&data_frame(3, false, b"bbb"));
        wire.extend_from_slice(&data_frame(1, true, b"ccc"));
        let frames = p.feed(&wire);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].stream_id, 1);
        assert_eq!(frames[1].stream_id, 3);
        assert_eq!(frames[2].stream_id, 1);
        assert!(frames[2].end_stream());
    }

    #[test]
    fn absurd_length_marks_broken() {
        let mut p = FrameParser::new_server();
        // Declared length way over MAX_FRAME_LEN.
        let f = [0xff, 0xff, 0xff, FRAME_DATA, 0, 0, 0, 0, 1];
        let frames = p.feed(&f);
        assert!(frames.is_empty());
        // Subsequent feeds are no-ops.
        assert!(p.feed(&data_frame(1, true, b"x")).is_empty());
    }

    #[test]
    fn strips_headers_padding() {
        // Padded HEADERS: pad-len=2, fragment=[0xaa,0xbb], 2 pad octets.
        let mut payload = vec![2u8];
        payload.extend_from_slice(&[0xaa, 0xbb]);
        payload.extend_from_slice(&[0, 0]);
        let out = headers_block_fragment(FLAG_PADDED, &payload).unwrap();
        assert_eq!(out, [0xaa, 0xbb]);
    }

    #[test]
    fn strips_headers_priority() {
        // PRIORITY HEADERS: 5 priority octets (stream dep + weight) then fragment.
        let mut payload = vec![0x80, 0, 0, 1, 10];
        payload.extend_from_slice(&[0x11, 0x22, 0x33]);
        let out = headers_block_fragment(FLAG_PRIORITY, &payload).unwrap();
        assert_eq!(out, [0x11, 0x22, 0x33]);
    }

    #[test]
    fn strips_headers_padding_and_priority() {
        // Both flags: pad-len, 5 priority octets, fragment, then padding.
        let mut payload = vec![3u8];
        payload.extend_from_slice(&[0, 0, 0, 1, 5]);
        payload.extend_from_slice(&[0x01, 0x02]);
        payload.extend_from_slice(&[0, 0, 0]);
        let out = headers_block_fragment(FLAG_PADDED | FLAG_PRIORITY, &payload).unwrap();
        assert_eq!(out, [0x01, 0x02]);
    }

    #[test]
    fn malformed_headers_padding_returns_none() {
        // Declared pad length larger than the payload that follows it.
        assert!(headers_block_fragment(FLAG_PADDED, &[9u8, 0xaa]).is_none());
    }

    #[test]
    fn single_headers_with_end_headers_completes() {
        let mut asm = HeaderBlockAssembler::new();
        let h = Frame {
            stream_id: 3,
            ftype: FRAME_HEADERS,
            flags: FLAG_END_HEADERS | FLAG_END_STREAM,
            payload: vec![0x82],
        };
        match asm.feed(&h) {
            Some(HeaderBlock::Complete(sid, block)) => {
                assert_eq!(sid, 3);
                assert_eq!(block, [0x82]);
            }
            _ => panic!("expected a completed header block"),
        }
    }

    #[test]
    fn reassembles_headers_then_continuation() {
        let mut asm = HeaderBlockAssembler::new();
        // HEADERS without END_HEADERS carrying the first fragment half.
        let h = Frame {
            stream_id: 1,
            ftype: FRAME_HEADERS,
            flags: FLAG_END_STREAM,
            payload: vec![0xde, 0xad],
        };
        assert!(asm.feed(&h).is_none());
        // CONTINUATION with END_HEADERS carrying the rest.
        let c = Frame {
            stream_id: 1,
            ftype: FRAME_CONTINUATION,
            flags: FLAG_END_HEADERS,
            payload: vec![0xbe, 0xef],
        };
        match asm.feed(&c) {
            Some(HeaderBlock::Complete(sid, block)) => {
                assert_eq!(sid, 1);
                assert_eq!(block, [0xde, 0xad, 0xbe, 0xef]);
            }
            _ => panic!("expected a completed header block"),
        }
    }

    #[test]
    fn continuation_without_open_block_is_malformed() {
        let mut asm = HeaderBlockAssembler::new();
        let c = Frame {
            stream_id: 1,
            ftype: FRAME_CONTINUATION,
            flags: FLAG_END_HEADERS,
            payload: vec![0x00],
        };
        assert!(matches!(asm.feed(&c), Some(HeaderBlock::Malformed)));
    }
}
