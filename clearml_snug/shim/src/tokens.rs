//! Token-count estimator.
//!
//! Uses byte-ratio approximations rather than real vocabulary-based
//! tokenization. The numbers below come from each
//! tokenizer family's documented English averages and are accurate to
//! ±15% for English-heavy LLM traffic. Non-English (Japanese, Chinese,
//! emoji-dense) can be 2-3x off - that's a fundamental property of the
//! byte-ratio approach.
//!
//! For the Claude family the ratio is generation-aware: the current models
//! tokenize the same English text into roughly 30% more tokens than the 3.x /
//! older 4.x models, so one fixed ratio undercounts current traffic. The served
//! model id selects the generation (see `claude_generation`), which lets the
//! chat-panel estimate (claude.ai, usage-less) track the current tokenizer.
//!
//! CALIBRATION: the per-generation ratios (`CLAUDE_*_BYTES_PER_TOKEN`) and the
//! model→generation thresholds are APPROXIMATE — derived from model naming and
//! documented English averages, not a shipped tokenizer. They are deliberately
//! kept as named constants plus a small classifier so they are cheap to
//! re-calibrate against real tokenizer counts on representative traffic.
//!
//! Trade-off accepted: a real tokenizer requires shipping the
//! vocabulary (cl100k_base is ~300 KiB; Claude comparable), which
//! roughly doubles the .so size. The contract field is `tokens_est`
//! (not `tokens`) - "est" makes the approximation contract-visible.
//!
//! Swap-in path: `estimate_tokens` can later be replaced with a call
//! into `tokenx-rs` or `tiktoken-rs` without changing any caller;
//! the function signature is the source of truth.

/// Bytes-per-token for the current-generation Claude tokenizer (Opus 4.7+,
/// Sonnet 5+, Fable 5+, Haiku 4.5+). APPROXIMATE — needs calibration.
const CLAUDE_NEW_BYTES_PER_TOKEN: f64 = 2.7;
/// Bytes-per-token for the legacy Claude tokenizer (Claude 3.x and the older
/// 4.x models). Anthropic's long-documented approximation. APPROXIMATE.
const CLAUDE_LEGACY_BYTES_PER_TOKEN: f64 = 3.5;

/// Claude tokenizer generation, inferred from the served model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeGen {
    /// Current tokenizer: Opus >= 4.7, Sonnet >= 5, Fable >= 5, Haiku >= 4.5.
    New,
    /// Claude 3.x and the older 4.x models below those thresholds.
    Legacy,
}

/// Classify a Claude model id into its tokenizer generation. The family/version
/// thresholds encode "current models use the new tokenizer": NEW for Opus 4.7+,
/// Sonnet 5+, Fable 5+, Haiku 4.5+ (and anything newer); LEGACY for Claude 3.x
/// and the older 4.x. A version-less or unrecognized id defaults to NEW, since
/// that matches the models the chat panel serves today.
///
/// The thresholds are derived from model NAMING (not a shipped tokenizer) and
/// need calibration — this is the single place to adjust the mapping.
fn claude_generation(model: &str) -> ClaudeGen {
    let m = model.to_ascii_lowercase();
    let new = match claude_version(&m) {
        // No parseable version (e.g. an alias): assume current-gen.
        None => true,
        Some(v) => {
            if m.contains("opus") {
                v >= (4, 7)
            } else if m.contains("haiku") {
                v >= (4, 5)
            } else if m.contains("sonnet") || m.contains("fable") {
                v.0 >= 5
            } else {
                // Unknown family carrying a version: assume current-gen.
                true
            }
        }
    };
    if new {
        ClaudeGen::New
    } else {
        ClaudeGen::Legacy
    }
}

/// Parse the `(major, minor)` version from a Claude model id, handling both the
/// current `claude-{family}-{major}-{minor}` naming (`claude-opus-4-8`,
/// `claude-sonnet-4-5-20250929`, `claude-sonnet-5`) and the legacy
/// `claude-{major}-{minor}-{family}` naming (`claude-3-5-sonnet-20241022`,
/// `claude-3-opus-20240229`). Trailing 8-digit date stamps are ignored — only
/// 1-2 digit tokens count as version numbers. `None` when no version digits sit
/// next to the family word.
fn claude_version(model: &str) -> Option<(u32, u32)> {
    let toks: Vec<&str> = model.split('-').collect();
    let fam = toks
        .iter()
        .position(|t| matches!(*t, "opus" | "sonnet" | "haiku" | "fable"))?;
    // A version token is 1-2 ASCII digits (an 8-digit date stamp is not one).
    let ver_at = |i: usize| -> Option<u32> {
        let t = *toks.get(i)?;
        (!t.is_empty() && t.len() <= 2 && t.bytes().all(|b| b.is_ascii_digit()))
            .then(|| t.parse().ok())
            .flatten()
    };
    // Current naming: digits immediately after the family word.
    if let Some(maj) = ver_at(fam + 1) {
        return Some((maj, ver_at(fam + 2).unwrap_or(0)));
    }
    // Legacy naming: digits immediately before the family word — major then
    // minor (`3-5-sonnet`); a lone `3-opus` has only the major.
    if fam >= 2 {
        if let Some(maj) = ver_at(fam - 2) {
            return Some((maj, ver_at(fam - 1).unwrap_or(0)));
        }
    }
    if fam >= 1 {
        if let Some(maj) = ver_at(fam - 1) {
            return Some((maj, 0));
        }
    }
    None
}

/// Bytes-per-token ratio for a tokenizer name, refined by the served `model`
/// where the family is generation-sensitive (Claude). Unknown names fall back
/// to "approx".
fn bytes_per_token(tokenizer: &str, model: Option<&str>) -> f64 {
    match tokenizer {
        // Anthropic: generation-aware. Without a served model to classify (the
        // shim's incremental per-chunk estimate runs before the model is parsed)
        // keep the long-documented legacy ratio.
        "claude" => match model {
            Some(m) => match claude_generation(m) {
                ClaudeGen::New => CLAUDE_NEW_BYTES_PER_TOKEN,
                ClaudeGen::Legacy => CLAUDE_LEGACY_BYTES_PER_TOKEN,
            },
            None => CLAUDE_LEGACY_BYTES_PER_TOKEN,
        },
        // OpenAI GPT-3.5 / GPT-4 / GPT-4o cl100k_base, English average.
        "cl100k" => 4.0,
        // Fallback: natural log of e. Conservative for tokens-per-byte:
        // undercounts slightly for English so we don't overcharge.
        _ => 2.72,
    }
}

/// Estimate the token count for `bytes` bytes of payload using the named
/// tokenizer, optionally refined by the served `model` (used for the
/// generation-aware Claude ratio). Returns 0 only when `bytes` is 0; otherwise
/// clamps to ≥ 1 (an actual request-line always represents at least a token's
/// worth of meaning).
pub fn estimate_tokens(bytes: u64, tokenizer: &str, model: Option<&str>) -> u64 {
    if bytes == 0 {
        return 0;
    }
    let ratio = bytes_per_token(tokenizer, model);
    let est = (bytes as f64 / ratio).round() as u64;
    est.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_bytes_zero_tokens() {
        assert_eq!(estimate_tokens(0, "claude", None), 0);
        assert_eq!(estimate_tokens(0, "cl100k", None), 0);
        assert_eq!(estimate_tokens(0, "approx", None), 0);
    }

    #[test]
    fn claude_legacy_ratio_3_5() {
        // No model / legacy model -> 3.5. 35 bytes / 3.5 = 10 tokens.
        assert_eq!(estimate_tokens(35, "claude", None), 10);
        // 7 bytes / 3.5 = 2.
        assert_eq!(estimate_tokens(7, "claude", None), 2);
        // An explicit legacy model id resolves to the same ratio.
        assert_eq!(estimate_tokens(35, "claude", Some("claude-3-opus-20240229")), 10);
    }

    #[test]
    fn claude_new_gen_ratio_2_7() {
        // A current-gen model tokenizes into more tokens for the same bytes.
        // 27 bytes / 2.7 = 10 tokens.
        assert_eq!(estimate_tokens(27, "claude", Some("claude-opus-4-8")), 10);
        // Same bytes: new-gen yields strictly more tokens than legacy.
        let n = 100_000;
        let new = estimate_tokens(n, "claude", Some("claude-sonnet-5"));
        let legacy = estimate_tokens(n, "claude", Some("claude-3-5-sonnet-20241022"));
        assert!(new > legacy, "new-gen ({new}) should exceed legacy ({legacy})");
    }

    #[test]
    fn claude_generation_classifies_representative_ids() {
        use ClaudeGen::{Legacy, New};
        // Current tokenizer.
        assert_eq!(claude_generation("claude-opus-4-8"), New);
        assert_eq!(claude_generation("claude-opus-4-7"), New);
        assert_eq!(claude_generation("claude-sonnet-5"), New);
        assert_eq!(claude_generation("claude-fable-5"), New);
        assert_eq!(claude_generation("claude-haiku-4-5"), New);
        // Version-less / unknown id defaults to current-gen.
        assert_eq!(claude_generation("claude-sonnet"), New);
        assert_eq!(claude_generation("some-future-model"), New);
        // Legacy tokenizer: Claude 3.x and the older 4.x below the thresholds.
        assert_eq!(claude_generation("claude-opus-4-1-20250805"), Legacy);
        assert_eq!(claude_generation("claude-opus-4-20250514"), Legacy);
        assert_eq!(claude_generation("claude-sonnet-4-5-20250929"), Legacy);
        assert_eq!(claude_generation("claude-3-5-sonnet-20241022"), Legacy);
        assert_eq!(claude_generation("claude-3-opus-20240229"), Legacy);
        assert_eq!(claude_generation("claude-3-haiku-20240307"), Legacy);
    }

    #[test]
    fn claude_version_parses_both_naming_schemes() {
        assert_eq!(claude_version("claude-opus-4-8"), Some((4, 8)));
        assert_eq!(claude_version("claude-sonnet-4-5-20250929"), Some((4, 5)));
        assert_eq!(claude_version("claude-opus-4-20250514"), Some((4, 0)));
        assert_eq!(claude_version("claude-sonnet-5"), Some((5, 0)));
        assert_eq!(claude_version("claude-3-5-sonnet-20241022"), Some((3, 5)));
        assert_eq!(claude_version("claude-3-opus-20240229"), Some((3, 0)));
        assert_eq!(claude_version("claude-sonnet"), None);
    }

    #[test]
    fn cl100k_ratio_4_0() {
        // 40 bytes / 4 = 10 tokens.
        assert_eq!(estimate_tokens(40, "cl100k", None), 10);
        // 100 bytes / 4 = 25 tokens.
        assert_eq!(estimate_tokens(100, "cl100k", None), 25);
    }

    #[test]
    fn approx_ratio_2_72() {
        // ~272 bytes / 2.72 ≈ 100 tokens.
        assert_eq!(estimate_tokens(272, "approx", None), 100);
    }

    #[test]
    fn unknown_tokenizer_falls_back_to_approx() {
        // "unknown" should behave like "approx".
        let a = estimate_tokens(272, "approx", None);
        let u = estimate_tokens(272, "unknown", None);
        assert_eq!(a, u);
    }

    #[test]
    fn tiny_input_clamps_to_one_token() {
        // 1 byte under any tokenizer is mathematically < 0.5 tokens; the
        // estimator clamps non-zero input to at least 1 token because
        // the wire still carried *something* meaningful.
        assert_eq!(estimate_tokens(1, "claude", None), 1);
        assert_eq!(estimate_tokens(1, "cl100k", None), 1);
        assert_eq!(estimate_tokens(1, "approx", None), 1);
    }

    #[test]
    fn large_input_is_monotonic_across_tokenizers() {
        // For the same byte count, lower bytes-per-token gives more
        // tokens. With no model, claude uses the legacy 3.5 ratio, so
        // approx (2.72) > claude (3.5) > cl100k (4.0).
        let n = 10_000;
        let a = estimate_tokens(n, "approx", None);
        let c = estimate_tokens(n, "claude", None);
        let k = estimate_tokens(n, "cl100k", None);
        assert!(a > c, "approx ({}) should give more tokens than claude ({})", a, c);
        assert!(c > k, "claude ({}) should give more tokens than cl100k ({})", c, k);
    }
}
