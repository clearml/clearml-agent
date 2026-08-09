//! Local CA + per-host leaf minting for TLS termination.
//!
//! Mints one CA and a leaf cert per SNI host, signed by the CA, cached. The
//! client is told to trust the CA (via `NODE_EXTRA_CA_CERTS` / `SSL_CERT_FILE`),
//! so a leaf presented on the terminated connection verifies against it.
//!
//! The CA is PERSISTED to disk (cert + key) and reused across proxy restarts.
//! This matters because clients (e.g. Electron renderer/network services and
//! bun-compiled CLIs) read the trusted CA once at their own startup and cache it
//! for their lifetime. If the proxy minted a fresh
//! random CA on every launch, a rebuild/redeploy of the proxy would leave those
//! long-lived clients trusting the previous CA while the proxy presents leaves
//! signed by the new one — every handshake would then fail with a
//! `CertificateUnknown` alert (surfacing in the app as "Connection lost" /
//! "SSL certificate verification failed"). Reusing the on-disk key keeps already
//! running clients trusting the proxy across restarts.

use std::collections::HashMap;
use std::sync::Mutex;

use base64::Engine as _;
use rcgen::{
    BasicConstraints, Certificate, CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// A minted leaf ready to hand to rustls' `with_single_cert`.
pub struct Leaf {
    pub cert_chain: Vec<CertificateDer<'static>>,
    pub key: PrivateKeyDer<'static>,
}

pub struct Ca {
    ca_cert: Certificate,
    ca_kp: KeyPair,
    pem: String,
    /// DER of the CA cert clients trust, appended to every minted leaf chain so
    /// the served chain is `[leaf, CA]`. A client pinning the CA by its
    /// SubjectPublicKeyInfo (Chromium's `--ignore-certificate-errors-spki-list`)
    /// matches on the CA cert *in the chain* — the per-host leaf's SPKI differs,
    /// so without the CA in the chain the pin never matches.
    ca_der: CertificateDer<'static>,
    /// host -> (leaf cert DER, leaf key PKCS#8 DER), cached so repeated
    /// connections to the same host reuse one leaf.
    cache: Mutex<HashMap<String, (Vec<u8>, Vec<u8>)>>,
}

/// Compute the Chromium SPKI pin for a DER-encoded SubjectPublicKeyInfo:
/// base64(SHA-256(spki_der)). This is the value passed to
/// `--ignore-certificate-errors-spki-list` and produced by HTTP Toolkit's
/// `generateSPKIFingerprint`, so a client trusts the proxy's leaves by pinning
/// the CA key rather than installing the cert.
pub fn spki_sha256_b64(spki_der: &[u8]) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, spki_der);
    base64::engine::general_purpose::STANDARD.encode(digest.as_ref())
}

/// Parse the first certificate from a PEM string into its DER bytes.
fn first_cert_der(cert_pem: &str) -> Option<CertificateDer<'static>> {
    let mut rd = std::io::BufReader::new(cert_pem.as_bytes());
    let first = rustls_pemfile::certs(&mut rd).next();
    first.and_then(|c| c.ok())
}

impl Ca {
    /// Deterministic CA certificate parameters. Rebuilt identically whether the
    /// CA is freshly generated or reloaded, so a reloaded key produces an issuer
    /// with the same distinguished name / key-usages as the persisted cert (only
    /// the signing key identity matters for a leaf to verify against it).
    fn ca_params() -> CertificateParams {
        let mut params = CertificateParams::new(Vec::new()).expect("ca params");
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params
            .distinguished_name
            .push(DnType::CommonName, "SNUG Proxy CA");
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        params.not_before = rcgen::date_time_ymd(2020, 1, 1);
        params.not_after = rcgen::date_time_ymd(2035, 1, 1);
        params
    }

    pub fn generate() -> Self {
        let ca_kp = KeyPair::generate().expect("generate CA key");
        let ca_cert = Self::ca_params().self_signed(&ca_kp).expect("self-sign CA");
        let pem = ca_cert.pem();
        let ca_der = ca_cert.der().clone();
        Self {
            ca_cert,
            ca_kp,
            pem,
            ca_der,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Reconstruct a CA from a persisted cert PEM + key PEM. The reloaded key is
    /// the source of trust; the in-memory issuer cert is re-derived from the same
    /// deterministic params so leaves it signs carry the matching issuer name and
    /// authority-key-id and verify against the on-disk cert clients already hold.
    fn from_pems(cert_pem: &str, key_pem: &str) -> Result<Self, rcgen::Error> {
        let ca_kp = KeyPair::from_pem(key_pem)?;
        let ca_cert = Self::ca_params().self_signed(&ca_kp)?;
        // Serve the exact on-disk cert bytes in the chain (byte-identical to what
        // a `NODE_EXTRA_CA_CERTS` client trusts), falling back to the re-derived
        // cert if the stored PEM can't be parsed. Both carry the same key, so the
        // SPKI pin matches either way.
        let ca_der = first_cert_der(cert_pem).unwrap_or_else(|| ca_cert.der().clone());
        Ok(Self {
            ca_cert,
            ca_kp,
            // Hand clients the exact on-disk cert (stable across restarts), not a
            // freshly re-serialized one.
            pem: cert_pem.to_string(),
            ca_der,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Load the CA cert+key from `cert_path`/`key_path` if both exist and parse;
    /// otherwise generate a fresh CA and persist it to those paths for reuse by
    /// the next run. `key_path` is written 0600 (it can mint trusted leaves).
    pub fn load_or_generate(cert_path: &str, key_path: &str) -> (Self, bool) {
        if let (Ok(cert_pem), Ok(key_pem)) =
            (std::fs::read_to_string(cert_path), std::fs::read_to_string(key_path))
        {
            if let Ok(ca) = Self::from_pems(&cert_pem, &key_pem) {
                return (ca, false);
            }
        }
        let ca = Self::generate();
        let _ = ca.write_pem(cert_path);
        let _ = ca.write_key(key_path);
        (ca, true)
    }

    pub fn write_pem(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.pem.as_bytes())
    }

    /// Persist the CA private key (PKCS#8 PEM) with owner-only permissions.
    fn write_key(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.ca_kp.serialize_pem().as_bytes())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    /// Mint (or fetch cached) a leaf for `host`.
    pub fn leaf_for(&self, host: &str) -> Leaf {
        let (cert_der, key_der) = {
            let mut cache = self.cache.lock().unwrap();
            if let Some((c, k)) = cache.get(host) {
                (c.clone(), k.clone())
            } else {
                let (c, k) = self.mint(host);
                cache.insert(host.to_string(), (c.clone(), k.clone()));
                (c, k)
            }
        };
        Leaf {
            // `[leaf, CA]`: the CA cert must ride in the served chain so a client
            // pinning the CA's SPKI can match it (see `ca_der`).
            cert_chain: vec![CertificateDer::from(cert_der), self.ca_der.clone()],
            key: PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_der)),
        }
    }

    /// DER-encoded SubjectPublicKeyInfo of the CA — the bytes a client hashes to
    /// derive the SPKI pin. rcgen embeds exactly `KeyPair::public_key_der()` as
    /// the cert's SPKI, so this matches the SPKI in the served CA cert.
    pub fn spki_der(&self) -> Vec<u8> {
        self.ca_kp.public_key_der()
    }

    /// The CA's SPKI pin: base64(SHA-256(SPKI DER)). Handed to the launcher for
    /// `--ignore-certificate-errors-spki-list`.
    pub fn spki_sha256_b64(&self) -> String {
        spki_sha256_b64(&self.spki_der())
    }

    fn mint(&self, host: &str) -> (Vec<u8>, Vec<u8>) {
        let mut params = CertificateParams::new(vec![host.to_string()]).expect("leaf params");
        params.distinguished_name.push(DnType::CommonName, host);
        params.not_before = rcgen::date_time_ymd(2020, 1, 1);
        params.not_after = rcgen::date_time_ymd(2035, 1, 1);
        params.use_authority_key_identifier_extension = true;
        let leaf_kp = KeyPair::generate().expect("generate leaf key");
        let leaf_cert = params
            .signed_by(&leaf_kp, &self.ca_cert, &self.ca_kp)
            .expect("sign leaf");
        (leaf_cert.der().to_vec(), leaf_kp.serialize_der())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reloaded CA must sign leaves with the SAME key as before, so a client
    /// that cached the persisted cert keeps trusting new leaves across restarts.
    #[test]
    fn reloaded_ca_keeps_the_same_signing_key() {
        let gen = Ca::generate();
        let key_pem = gen.ca_kp.serialize_pem();
        let reloaded = Ca::from_pems(&gen.pem, &key_pem).expect("reload CA");
        assert_eq!(
            gen.ca_kp.public_key_der(),
            reloaded.ca_kp.public_key_der(),
            "reloaded CA signs with the original key"
        );
        assert_eq!(gen.pem, reloaded.pem, "reloaded CA hands out the on-disk cert");
    }

    /// `spki_sha256_b64` must match the Chromium/HTTP-Toolkit algorithm
    /// (base64(SHA-256(SPKI DER))) exactly, so a client's
    /// `--ignore-certificate-errors-spki-list` pin matches our CA. Vector taken
    /// from an EC P-256 self-signed cert via
    /// `openssl x509 -pubkey -noout | openssl pkey -pubin -outform der |
    ///  openssl dgst -sha256 -binary | openssl base64`.
    #[test]
    fn spki_sha256_matches_openssl_vector() {
        let spki_der = base64::engine::general_purpose::STANDARD
            .decode(
                "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE/WMrjGA5Da82hHvOq+Ua0hxmFSpt\
                 Ha6fSsDBLPpl07OxmbOlaq5m6iAq68xvUAYqd22+CuH7I1jDgmFO2Zgjpg==",
            )
            .unwrap();
        assert_eq!(
            spki_sha256_b64(&spki_der),
            "eufYZzcEATRLzyiibHAIOy7q3fyv4gUcWs2w2bEy+xg="
        );
    }

    /// The CA's SPKI pin is a stable base64 of a 32-byte SHA-256, and reloading
    /// the CA (same key) yields the SAME pin — the launcher can persist it.
    #[test]
    fn ca_spki_pin_is_stable_across_reload() {
        let gen = Ca::generate();
        let pin = gen.spki_sha256_b64();
        // base64 of 32 bytes is 44 chars (43 + '=' padding).
        assert_eq!(pin.len(), 44, "sha256 base64 length");
        let key_pem = gen.ca_kp.serialize_pem();
        let reloaded = Ca::from_pems(&gen.pem, &key_pem).expect("reload CA");
        assert_eq!(pin, reloaded.spki_sha256_b64(), "pin stable across reload");
    }

    /// The served chain is `[leaf, CA]` (not just the leaf), so a CA-SPKI pin can
    /// match the CA cert riding in the chain.
    #[test]
    fn leaf_chain_includes_ca_cert() {
        let ca = Ca::generate();
        let leaf = ca.leaf_for("api.anthropic.com");
        assert_eq!(leaf.cert_chain.len(), 2, "chain is [leaf, CA]");
        assert_eq!(
            leaf.cert_chain[1].as_ref(),
            ca.ca_der.as_ref(),
            "second cert is the trusted CA"
        );
        assert_ne!(
            leaf.cert_chain[0].as_ref(),
            ca.ca_der.as_ref(),
            "first cert is the per-host leaf, not the CA"
        );
    }

    /// `load_or_generate` persists on first run and reuses on the next.
    #[test]
    fn load_or_generate_persists_then_reuses() {
        let dir = std::env::temp_dir().join(format!("snug_ca_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cert = dir.join("ca.pem");
        let key = dir.join("ca.key");
        let cert_s = cert.to_str().unwrap();
        let key_s = key.to_str().unwrap();

        let (a, fresh_a) = Ca::load_or_generate(cert_s, key_s);
        assert!(fresh_a, "first call generates");
        let (b, fresh_b) = Ca::load_or_generate(cert_s, key_s);
        assert!(!fresh_b, "second call reuses the persisted CA");
        assert_eq!(
            a.ca_kp.public_key_der(),
            b.ca_kp.public_key_der(),
            "persisted signing key is reused"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
