//! Response-body decompression for the usage scanner.
//!
//! Some upstreams gzip/zstd-compress their HTTP/2 response bodies. The shim's
//! normal defence — forcing
//! `Accept-Encoding: identity` on the request (`inject::rewrite_headers`) —
//! only reaches HTTP/1 requests; over h2 the request headers are HPACK-encoded
//! and the shim does not rewrite them. So the usage scanner would otherwise see
//! compressed bytes and extract nothing.
//!
//! We instead detect the encoding from the body's own magic bytes (independent
//! of the HPACK-hidden `Content-Encoding` header) and inflate it before the
//! scan. gzip via miniz_oxide (flate2, pure Rust), zstd via ruzstd (pure Rust)
//! — no C in the task's address space.

use std::io::Read;

/// Transfer encoding of a response body, decided from its leading bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Encoding {
    /// Not yet classified (no body bytes seen).
    #[default]
    Undecided,
    /// Plaintext — scanned incrementally, no buffering/inflate.
    Identity,
    Gzip,
    Zstd,
}

impl Encoding {
    pub fn is_compressed(self) -> bool {
        matches!(self, Encoding::Gzip | Encoding::Zstd)
    }
}

/// Classify a response body from its first bytes (gzip `1f 8b`, zstd magic
/// `28 b5 2f fd`; anything else is treated as plaintext).
pub fn detect(data: &[u8]) -> Encoding {
    if data.starts_with(&[0x1f, 0x8b]) {
        Encoding::Gzip
    } else if data.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Encoding::Zstd
    } else {
        Encoding::Identity
    }
}

/// Inflate a fully-buffered compressed body, capping output at `out_cap` bytes.
/// Returns whatever plaintext was produced (partial output is kept if the
/// stream is truncated — the leading usage prelude often survives), or `None`
/// when the encoding isn't compressed or nothing could be decoded.
pub fn decompress(enc: Encoding, data: &[u8], out_cap: usize) -> Option<Vec<u8>> {
    match enc {
        Encoding::Gzip => read_capped(flate2::read::MultiGzDecoder::new(data), out_cap),
        Encoding::Zstd => {
            let dec = ruzstd::StreamingDecoder::new(data).ok()?;
            read_capped(dec, out_cap)
        }
        Encoding::Identity | Encoding::Undecided => None,
    }
}

/// Read a decoder to EOF or `cap`, keeping partial output on a mid-stream error
/// (a truncated body still often carries the leading usage prelude).
fn read_capped<R: Read>(mut r: R, cap: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                out.extend_from_slice(&buf[..n]);
                if out.len() >= cap {
                    break;
                }
            }
            Err(_) => break, // keep whatever decoded before the error
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detects_magic() {
        assert_eq!(detect(&[0x1f, 0x8b, 0x08, 0x00]), Encoding::Gzip);
        assert_eq!(detect(&[0x28, 0xb5, 0x2f, 0xfd, 0x00]), Encoding::Zstd);
        assert_eq!(detect(b"plaintext body, not compressed\n"), Encoding::Identity);
        assert_eq!(detect(b""), Encoding::Identity);
    }

    #[test]
    fn roundtrips_gzip() {
        let payload = b"the quick brown fox jumps over the lazy dog, repeatedly, so this compresses\n\n";
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(payload).unwrap();
        let gz = e.finish().unwrap();
        assert_eq!(detect(&gz), Encoding::Gzip);
        let out = decompress(Encoding::Gzip, &gz, 1 << 20).expect("inflate");
        assert_eq!(out, payload);
    }

    #[test]
    fn non_compressed_is_none() {
        assert!(decompress(Encoding::Identity, b"plain", 1 << 20).is_none());
        assert!(decompress(Encoding::Gzip, b"not really gzip", 1 << 20).is_none());
    }
}
