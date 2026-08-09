//! Per-task session registry: turns each request's conversation
//! `Fingerprint` (`body_scan::conversation_fingerprint`) into a stable
//! `chat_id`, matching a request to the chat it continues even as the
//! transcript evolves (appended, retried, edited, trimmed). One shim load ==
//! one task process, so the registry is process-global and starts empty per
//! task.
//!
//! Matching is gated on an equal `system_hash`, then scored over the turn-hash
//! lists by, in order of confidence:
//!   * **clean append/extension** — the entire stored turn list reappears at
//!     the head of the new request (`incoming.starts_with(stored)`). Accepted
//!     at any length, including 1, so a one-turn chat growing to many turns
//!     stays one chat.
//!   * **longest-common-prefix ≥ 2** — retry/regeneration and trailing edits
//!     (the last turn changed) share their leading turns. The ≥2 floor means a
//!     single coincidentally-shared opening turn does NOT merge two chats
//!     (so a templated first turn doesn't collapse every conversation).
//!   * **tail→head overlap ≥ 2** — sliding-window trimming, where the front of
//!     the transcript is dropped but a window of recent turns is resent.
//!
//! The `chat_id` is a per-task running ordinal ("1", "2", …) assigned when a
//! session is first opened and then inherited by every continuation, so it is
//! stable for the chat's lifetime and reads as a human-friendly counter on the
//! scalar series ("Anthropic / chat 1") rather than an opaque content hash.
//! Distinct chats get distinct ordinals; the matching itself still keys on the
//! `system_hash` + turn hashes (below), independent of how the id is rendered.
//!
//! Limitations (acceptable for a metering heuristic):
//!   * Two chats with an identical system AND identical first ≥2 turns merge
//!     (indistinguishable by content).
//!   * Editing a turn in the MIDDLE of the history forks (the prefix diverges).
//!   * Trimming that drops the front while retaining > `OVERLAP_SCAN_CAP` old
//!     turns in one step may fork (the overlap scan is bounded for cost).
//!
//! The bias is deliberately toward *merging* over *splitting*: a split (one
//! chat fragmenting into many series) is the failure mode this guards against.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use crate::body_scan::Fingerprint;

/// Cap on retained sessions; the least-recently-touched is evicted past this.
const MAX_SESSIONS: usize = 256;
/// Cap on turn-hashes stored per session (real conversations are far smaller;
/// bounds memory and the match scan against pathological lengths).
const MAX_TURNS_STORED: usize = 512;
/// Upper bound on the tail→head overlap scan (sliding-window-trim detection).
/// Trimming that retains a window larger than this in one step may fork.
const OVERLAP_SCAN_CAP: usize = 128;
/// Minimum shared turns for the LCP and overlap matches. The clean-append match
/// is exempt (it is high-confidence at any length).
const MIN_SHARED: usize = 2;

struct Session {
    system_hash: u64,
    turns: Vec<u64>,
    chat_id: String,
    /// Last-touched ordinal, for LRU eviction and tie-breaking.
    seq: u64,
}

pub struct SessionRegistry {
    sessions: Vec<Session>,
    seq: u64,
    /// Monotonic chat counter: the next ordinal handed to a freshly opened
    /// session. Never reused (an evicted-then-reappearing chat gets a new
    /// number), so series ids stay distinct for the task's lifetime.
    next_ordinal: u64,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            seq: 0,
            next_ordinal: 0,
        }
    }

    /// Match `fp` to an existing chat, or open a new one. Returns the chat id.
    pub fn assign(&mut self, fp: &Fingerprint) -> String {
        self.seq += 1;
        let now = self.seq;

        // Best matching same-system session, by (score, recency).
        let mut best: Option<(usize, usize, u64)> = None; // (index, score, seq)
        for (i, s) in self.sessions.iter().enumerate() {
            if s.system_hash != fp.system_hash {
                continue;
            }
            let score = match_score(&s.turns, &fp.turn_hashes);
            if score == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, bscore, bseq)) => score > bscore || (score == bscore && s.seq > bseq),
            };
            if better {
                best = Some((i, score, s.seq));
            }
        }

        if let Some((i, _, _)) = best {
            let id = self.sessions[i].chat_id.clone();
            self.sessions[i].turns = cap_tail(&fp.turn_hashes);
            self.sessions[i].seq = now;
            return id;
        }

        // New chat: hand out the next running ordinal as its id.
        self.next_ordinal += 1;
        let id = self.next_ordinal.to_string();
        if self.sessions.len() >= MAX_SESSIONS {
            if let Some((idx, _)) = self.sessions.iter().enumerate().min_by_key(|(_, s)| s.seq) {
                self.sessions.swap_remove(idx);
            }
        }
        self.sessions.push(Session {
            system_hash: fp.system_hash,
            turns: cap_tail(&fp.turn_hashes),
            chat_id: id.clone(),
            seq: now,
        });
        id
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Score how strongly `stored` turns continue into `incoming` turns. 0 = no
/// match; higher = longer / more confident alignment.
fn match_score(stored: &[u64], incoming: &[u64]) -> usize {
    // Clean append/extension: the whole prior turn list reappears at the head
    // of the new request. Highest confidence, accepted at any length.
    if !stored.is_empty() && incoming.starts_with(stored) {
        return stored.len();
    }
    // Retry / trailing-edit / branch: share ≥ MIN_SHARED leading turns.
    let lcp = stored
        .iter()
        .zip(incoming)
        .take_while(|(x, y)| x == y)
        .count();
    let mut best = if lcp >= MIN_SHARED { lcp } else { 0 };
    // Sliding-window trim: the prior's tail equals the new's head, ≥ MIN_SHARED.
    let maxk = stored.len().min(incoming.len()).min(OVERLAP_SCAN_CAP);
    let mut k = maxk;
    while k >= MIN_SHARED {
        if stored[stored.len() - k..] == incoming[..k] {
            best = best.max(k);
            break;
        }
        k -= 1;
    }
    best
}

/// Keep only the last `MAX_TURNS_STORED` turn-hashes (preserves the tail used
/// for overlap matching; the head matters only up to the same bound).
fn cap_tail(turns: &[u64]) -> Vec<u64> {
    if turns.len() <= MAX_TURNS_STORED {
        turns.to_vec()
    } else {
        turns[turns.len() - MAX_TURNS_STORED..].to_vec()
    }
}

/// Content-derived id, used **only** as the poison fallback in
/// `assign_chat_id` (the normal path hands out a running ordinal). Keeps a
/// distinct id when the registry mutex is unusable so metering still gets one.
fn mint_id(fp: &Fingerprint) -> String {
    let mut h = DefaultHasher::new();
    fp.system_hash.hash(&mut h);
    fp.turn_hashes.hash(&mut h);
    format!("{:016x}", h.finish())
}

// --- process-global instance ---------------------------------------------

static REGISTRY: OnceLock<Mutex<SessionRegistry>> = OnceLock::new();

/// Match-or-open a chat id for `fp` in the process-global registry. Called from
/// `state::build_request_completed` while it holds the connection-state lock;
/// this is the only acquirer of the registry lock, so the nested order is
/// consistent (no deadlock).
pub fn assign_chat_id(fp: &Fingerprint) -> String {
    let reg = REGISTRY.get_or_init(|| Mutex::new(SessionRegistry::new()));
    match reg.lock() {
        Ok(mut r) => r.assign(fp),
        // assign() can't panic, so poisoning shouldn't happen; if it somehow
        // does, fall back to a standalone mint so metering still gets an id
        // (it just won't group with the poisoned registry's sessions).
        Err(_) => mint_id(fp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(system: u64, turns: &[u64]) -> Fingerprint {
        Fingerprint {
            system_hash: system,
            turn_hashes: turns.to_vec(),
        }
    }

    #[test]
    fn append_keeps_same_id() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10]));
        // 1-turn chat grows to many turns: clean append, same id.
        assert_eq!(r.assign(&fp(1, &[10, 11, 12])), a);
        assert_eq!(r.assign(&fp(1, &[10, 11, 12, 13, 14])), a);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn retry_shorter_prefix_keeps_id() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10, 11, 12]));
        // Regeneration drops the last turn: shares ≥2 leading turns.
        assert_eq!(r.assign(&fp(1, &[10, 11])), a);
    }

    #[test]
    fn trailing_edit_keeps_id() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10, 11, 12]));
        // Last turn edited (12 -> 99): LCP=2, same chat.
        assert_eq!(r.assign(&fp(1, &[10, 11, 99])), a);
    }

    #[test]
    fn sliding_window_trim_keeps_id() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10, 11, 12, 13, 14]));
        // Front trimmed, recent window + a new turn resent: tail→head overlap.
        assert_eq!(r.assign(&fp(1, &[12, 13, 14, 15])), a);
    }

    #[test]
    fn new_chat_sharing_one_opening_turn_does_not_merge() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10, 11])); // chat A, ≥2 turns
        // A different chat that happens to share only the first turn (e.g. a
        // templated opener) must NOT merge into A.
        let b = r.assign(&fp(1, &[10, 99]));
        assert_ne!(b, a);
        // A single-turn newcomer sharing just turn 0 also stays separate.
        let c = r.assign(&fp(1, &[10]));
        assert_ne!(c, a);
        assert_ne!(c, b);
    }

    #[test]
    fn system_gate_separates_identical_turns() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10, 11]));
        // Same turns, different system prompt -> different chat.
        assert_ne!(r.assign(&fp(2, &[10, 11])), a);
    }

    #[test]
    fn distinct_openings_distinct_ids() {
        let mut r = SessionRegistry::new();
        let a = r.assign(&fp(1, &[10]));
        let b = r.assign(&fp(1, &[20]));
        assert_ne!(a, b);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn interleaved_chats_stay_distinct() {
        // Two chats advancing in lockstep keep their own ids.
        let mut r = SessionRegistry::new();
        let a1 = r.assign(&fp(1, &[10]));
        let b1 = r.assign(&fp(2, &[20]));
        let a2 = r.assign(&fp(1, &[10, 11]));
        let b2 = r.assign(&fp(2, &[20, 21]));
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert_ne!(a1, b1);
    }

    #[test]
    fn evicts_past_cap() {
        let mut r = SessionRegistry::new();
        for i in 0..(MAX_SESSIONS as u64 + 10) {
            // Distinct system each -> all new sessions.
            r.assign(&fp(i, &[i]));
        }
        assert_eq!(r.len(), MAX_SESSIONS, "registry is bounded");
    }

    #[test]
    fn assigns_sequential_ordinals() {
        // The id is now a running counter ("1", "2", …) handed out in
        // first-seen order; a continuation reuses its chat's ordinal.
        let mut r = SessionRegistry::new();
        assert_eq!(r.assign(&fp(1, &[10])), "1");
        assert_eq!(r.assign(&fp(2, &[20])), "2"); // distinct chat -> next ordinal
        assert_eq!(r.assign(&fp(1, &[10, 11])), "1"); // continuation of chat 1
        assert_eq!(r.assign(&fp(3, &[30])), "3"); // another distinct chat
    }

    #[test]
    fn evicted_chat_does_not_reuse_ordinal() {
        // The counter is monotonic: filling past the cap evicts old sessions,
        // and a later new chat still gets a fresh number (never a recycled one).
        let mut r = SessionRegistry::new();
        for i in 0..(MAX_SESSIONS as u64) {
            r.assign(&fp(i, &[i]));
        }
        let next = r.assign(&fp(9_999, &[9_999]));
        assert_eq!(next, (MAX_SESSIONS as u64 + 1).to_string());
    }
}
