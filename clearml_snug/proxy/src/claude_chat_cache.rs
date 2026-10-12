//! Prompt-cache split for usage-less consumer chat requests (claude.ai).
//!
//! The claude.ai chat panel returns no `usage`, so its input is byte-estimated
//! from the request body. That body is not the conversation: the server keeps the
//! history, and the request carries only the new `prompt` plus the connectors'
//! full `tools` array (hundreds of KB for a user with several connectors),
//! resent on every turn. Estimating all of it as fresh input reports ~100K tokens
//! for a one-word message, and none of the history the model actually reads.
//!
//! Anthropic builds a prompt as `tools`, then `system`, then the messages, and
//! caches that prefix. The first request that carries a given prefix writes it to
//! the cache; later requests within the cache lifetime (five minutes by default)
//! read it. This module models exactly that, in two parts:
//!
//! * the `tools` + `system` bytes of the request are the cacheable prefix, a cache
//!   write on first sight and a cache read afterwards;
//! * the conversation history, which the request does not carry, is rebuilt from
//!   the turns seen on the wire (each turn's prompt, attachments and reply text,
//!   per conversation id) and counted as cache read, or as a cache write when the
//!   conversation has been idle past the cache lifetime.
//!
//! The rest of the body (the new prompt, attachments, parameters) is fresh input.
//! Only byte counts are kept per conversation, never text.
//!
//! The app loads a chat whole (`GET` of the conversation) when it is opened and
//! again after each completion, so the history is taken from that response
//! (`history_bytes`) and covers turns the proxy never saw. It counts what the
//! model re-reads: text, tool calls and tool results (a web search shows only its
//! titles and URLs on the wire, so those undercount the page text the model gets),
//! but not earlier thinking. Seeded history is cold: a chat reopened later is no
//! longer in the provider's cache, so its first turn writes it. Between a
//! completion and the reload that follows, the live count (prompt and reply text)
//! stands in.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use serde::Deserialize;
use serde_json::value::RawValue;

/// How long a seen prefix or conversation is assumed to still be served from the
/// provider's prompt cache: Anthropic's default lifetime of five minutes (the
/// one-hour lifetime is opt-in, and the claude.ai wire does not show it being
/// used). Each read refreshes it. A longer gap is counted as a new cache write,
/// which prices a sparse chat conservatively rather than as a cheap read.
const CACHE_TTL_MS: u64 = 5 * 60 * 1000;

/// Below this the prefix is not cacheable (Anthropic requires at least ~1024
/// tokens; at the ~2.7 bytes/token estimate that is ~2.8 KB), so it stays fresh.
const MIN_CACHEABLE_BYTES: u64 = 3072;

/// Bound on remembered prefixes; the oldest is dropped past it.
const MAX_PREFIXES: usize = 64;

/// Bound on remembered conversations; the least recently used is dropped past it.
const MAX_CONVERSATIONS: usize = 512;

/// The request's byte count, split by how the provider bills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    pub fresh_bytes: u64,
    pub cache_read_bytes: u64,
    pub cache_write_bytes: u64,
}

/// The fields of an Anthropic-shaped request that the split needs. Unknown fields
/// are ignored; `RawValue` keeps each value's exact wire bytes.
#[derive(Deserialize)]
struct Request<'a> {
    #[serde(borrow, default)]
    tools: Option<&'a RawValue>,
    #[serde(borrow, default)]
    system: Option<&'a RawValue>,
    #[serde(borrow, default)]
    prompt: Option<&'a RawValue>,
    #[serde(borrow, default)]
    attachments: Option<&'a RawValue>,
}

/// What is remembered about one conversation.
struct History {
    /// Prompt, attachment and reply bytes of the turns seen so far.
    bytes: u64,
    last_ms: u64,
    /// Seeded from a loaded chat and not yet sent from: not in the provider's cache.
    cold: bool,
}

/// The conversation id in a claude.ai chat path
/// (`/api/organizations/<org>/chat_conversations/<id>/completion`).
pub fn conversation_id(path: &str) -> Option<&str> {
    let rest = path.split("/chat_conversations/").nth(1)?;
    let id = rest.split(|c| c == '/' || c == '?').next()?;
    (!id.is_empty()).then_some(id)
}

/// Whether a request is the app loading a whole conversation: a `GET` of
/// `/chat_conversations/<id>` (with or without a query), not one of its sub-paths.
pub fn is_conversation_load(method: Option<&str>, path: Option<&str>) -> bool {
    if method != Some("GET") {
        return false;
    }
    let Some(path) = path else { return false };
    let Some(id) = conversation_id(path) else { return false };
    let after = path.split("/chat_conversations/").nth(1).map(|r| &r[id.len()..]).unwrap_or("/");
    after.is_empty() || after.starts_with('?')
}

/// Length of a JSON string value, 0 when absent or not a string.
fn str_len(v: Option<&serde_json::Value>) -> u64 {
    v.and_then(|t| t.as_str()).map(|t| t.len() as u64).unwrap_or(0)
}

/// The text of a tool result's `content` that the model reads: a plain string, or
/// a list of items whose `text`, `title`, `url` and nested `content` count. The
/// `metadata` of an item (site name, favicon, ...) is the app's own and does not.
fn result_bytes(v: Option<&serde_json::Value>) -> u64 {
    use serde_json::Value;
    match v {
        Some(Value::String(s)) => s.len() as u64,
        Some(Value::Array(items)) => items.iter().map(|i| result_bytes(Some(i))).sum(),
        Some(Value::Object(o)) => {
            str_len(o.get("text")) + str_len(o.get("title")) + str_len(o.get("url")) + result_bytes(o.get("content"))
        }
        _ => 0,
    }
}

/// Bytes of one content block that the model re-reads on the next turn: text, a
/// tool call (its name and arguments) and a tool result. Thinking of earlier turns
/// is stripped from the prompt, and `token_budget` is the app's own marker.
fn block_bytes(b: &serde_json::Value) -> u64 {
    match b.get("type").and_then(|t| t.as_str()) {
        Some("tool_use") => {
            str_len(b.get("name")) + b.get("input").map(|i| i.to_string().len() as u64).unwrap_or(0)
        }
        Some("tool_result") => str_len(b.get("name")) + result_bytes(b.get("content")),
        Some("thinking" | "redacted_thinking" | "token_budget") => 0,
        _ => str_len(b.get("text")),
    }
}

/// Bytes of one message of a loaded conversation: its content blocks (or its plain
/// `text`) plus the extracted content of its attachments.
fn message_bytes(m: &serde_json::Value) -> u64 {
    let mut n = match m.get("content").and_then(|c| c.as_array()) {
        Some(blocks) => blocks.iter().map(block_bytes).sum(),
        None => str_len(m.get("text")),
    };
    if let Some(atts) = m.get("attachments").and_then(|a| a.as_array()) {
        n += atts.iter().map(|a| str_len(a.get("extracted_content"))).sum::<u64>();
    }
    n
}

/// Total bytes of the active branch of a conversation-load response body: the
/// messages from `current_leaf_message_uuid` up to the root (every message when the
/// response names no leaf). `None` when the body is not such a response.
pub fn history_bytes(body: &[u8]) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let msgs = v.get("chat_messages")?.as_array()?;
    let mut by_id: HashMap<&str, (&str, u64)> = HashMap::new();
    let mut all = 0u64;
    for m in msgs {
        let Some(id) = m.get("uuid").and_then(|u| u.as_str()) else { continue };
        let parent = m.get("parent_message_uuid").and_then(|p| p.as_str()).unwrap_or("");
        let bytes = message_bytes(m);
        all += bytes;
        by_id.insert(id, (parent, bytes));
    }
    let Some(mut at) = v.get("current_leaf_message_uuid").and_then(|l| l.as_str()) else {
        return Some(all);
    };
    if !by_id.contains_key(at) {
        return Some(all);
    }
    let mut total = 0u64;
    // Bounded by the message count so a malformed parent cycle cannot loop.
    for _ in 0..by_id.len() {
        let Some(&(parent, bytes)) = by_id.get(at) else { break };
        total += bytes;
        at = parent;
    }
    Some(total)
}

/// Remembers which prefixes and conversations were recently seen, to tell a cache
/// write from a read and to carry each conversation's history.
#[derive(Default)]
pub struct ChatCache {
    prefixes: Mutex<HashMap<u64, u64>>,
    conversations: Mutex<HashMap<u64, History>>,
}

fn hash_of<T: Hash + ?Sized>(v: &T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

impl ChatCache {
    /// Split a decompressed request body and record the turn. `conversation` is
    /// the chat's id and `reply_bytes` the generated text of this exchange's
    /// response. `None` when the body is not an Anthropic-shaped JSON object with a
    /// cacheable prefix, so the caller keeps its all-fresh estimate.
    pub fn split(
        &self,
        plain: &[u8],
        conversation: Option<&str>,
        reply_bytes: u64,
        now_ms: u64,
    ) -> Option<Split> {
        let req: Request = serde_json::from_slice(plain).ok()?;
        let tools = req.tools.map(|v| v.get()).unwrap_or("");
        let system = req.system.map(|v| v.get()).unwrap_or("");
        let prefix_bytes = (tools.len() + system.len()) as u64;
        if prefix_bytes < MIN_CACHEABLE_BYTES {
            return None;
        }
        let total = plain.len() as u64;
        let fresh_bytes = total.saturating_sub(prefix_bytes);

        let prefix_cached = self.touch_prefix(hash_of(&(system, tools)), now_ms);
        let (mut cache_read_bytes, mut cache_write_bytes) =
            if prefix_cached { (prefix_bytes, 0) } else { (0, prefix_bytes) };

        if let Some(conv) = conversation {
            let turn_bytes = req.prompt.map(|v| v.get().len()).unwrap_or(0)
                + req.attachments.map(|v| v.get().len()).unwrap_or(0);
            let (history, history_cached) =
                self.record_turn(hash_of(conv), turn_bytes as u64 + reply_bytes, now_ms);
            if history_cached {
                cache_read_bytes += history;
            } else {
                cache_write_bytes += history;
            }
        }
        Some(Split { fresh_bytes, cache_read_bytes, cache_write_bytes })
    }

    /// Seed a conversation's history from a loaded chat. The load is what the server
    /// holds, so it replaces the live count (which also counts the thinking and the
    /// markup of the replies) but a conversation already tracked live keeps its
    /// cache state; an unseen one starts cold.
    pub fn seed_history(&self, conversation: &str, history_bytes: u64) {
        let Ok(mut convs) = self.conversations.lock() else { return };
        match convs.get_mut(&hash_of(conversation)) {
            Some(h) => h.bytes = history_bytes,
            None => {
                convs.insert(hash_of(conversation), History { bytes: history_bytes, last_ms: 0, cold: true });
                Self::evict_oldest(&mut convs);
            }
        }
    }

    /// Drop the least recently used conversation past the bound.
    fn evict_oldest(convs: &mut HashMap<u64, History>) {
        if convs.len() > MAX_CONVERSATIONS {
            if let Some(oldest) = convs.iter().min_by_key(|(_, h)| h.last_ms).map(|(&k, _)| k) {
                convs.remove(&oldest);
            }
        }
    }

    /// Whether `key` was seen within the cache lifetime; marks it seen now.
    fn touch_prefix(&self, key: u64, now_ms: u64) -> bool {
        let Ok(mut seen) = self.prefixes.lock() else { return false };
        let cached =
            seen.get(&key).map(|&last| now_ms.saturating_sub(last) <= CACHE_TTL_MS).unwrap_or(false);
        seen.insert(key, now_ms);
        if seen.len() > MAX_PREFIXES {
            if let Some(oldest) = seen.iter().min_by_key(|(_, &t)| t).map(|(&k, _)| k) {
                seen.remove(&oldest);
            }
        }
        cached
    }

    /// Return the history before this turn and whether it is still cached, then add
    /// the turn's bytes to it.
    fn record_turn(&self, key: u64, turn_bytes: u64, now_ms: u64) -> (u64, bool) {
        let Ok(mut convs) = self.conversations.lock() else { return (0, false) };
        let h = convs.entry(key).or_insert(History { bytes: 0, last_ms: now_ms, cold: false });
        let before = h.bytes;
        let cached = !h.cold && now_ms.saturating_sub(h.last_ms) <= CACHE_TTL_MS;
        h.bytes += turn_bytes;
        h.last_ms = now_ms;
        h.cold = false;
        Self::evict_oldest(&mut convs);
        (before, cached)
    }
}

/// The process-wide cache: one proxy serves one app session.
pub fn global() -> &'static ChatCache {
    static CACHE: OnceLock<ChatCache> = OnceLock::new();
    CACHE.get_or_init(ChatCache::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(prompt: &str, tool_pad: usize) -> Vec<u8> {
        let tools = format!("[{{\"name\":\"t\",\"description\":\"{}\"}}]", "x".repeat(tool_pad));
        format!(
            "{{\"prompt\":{:?},\"timezone\":\"UTC\",\"model\":\"m\",\"tools\":{},\"files\":[]}}",
            prompt, tools
        )
        .into_bytes()
    }

    #[test]
    fn first_sight_writes_then_reads() {
        let c = ChatCache::default();
        let b = body("hello", 10_000);
        let first = c.split(&b, None, 0, 1_000).unwrap();
        assert!(first.cache_write_bytes > 10_000);
        assert_eq!(first.cache_read_bytes, 0);
        // Everything outside the tools array stays fresh and the buckets cover the body.
        assert!(first.fresh_bytes < 200);
        assert_eq!(first.fresh_bytes + first.cache_write_bytes, b.len() as u64);

        let second = c.split(&body("a different prompt", 10_000), None, 0, 2_000).unwrap();
        assert_eq!(second.cache_write_bytes, 0);
        assert_eq!(second.cache_read_bytes, first.cache_write_bytes);
    }

    #[test]
    fn changed_tools_are_written_again() {
        let c = ChatCache::default();
        c.split(&body("hello", 10_000), None, 0, 1_000).unwrap();
        let s = c.split(&body("hello", 11_000), None, 0, 2_000).unwrap();
        assert!(s.cache_write_bytes > 0);
        assert_eq!(s.cache_read_bytes, 0);
    }

    #[test]
    fn prefix_expires_after_the_cache_lifetime() {
        let c = ChatCache::default();
        c.split(&body("hello", 10_000), None, 0, 1_000).unwrap();
        let within = c.split(&body("hello", 10_000), None, 0, 1_000 + CACHE_TTL_MS).unwrap();
        assert!(within.cache_read_bytes > 0, "a read refreshes the lifetime");
        let after =
            c.split(&body("hello", 10_000), None, 0, 1_000 + 2 * CACHE_TTL_MS + 1).unwrap();
        assert!(after.cache_write_bytes > 0);
        assert_eq!(after.cache_read_bytes, 0);
    }

    #[test]
    fn small_or_missing_prefix_is_not_split() {
        let c = ChatCache::default();
        assert!(c.split(&body("hello", 50), None, 0, 1).is_none());
        assert!(c.split(b"{\"prompt\":\"hello\"}", None, 0, 1).is_none());
        assert!(c.split(b"not json at all", None, 0, 1).is_none());
    }

    #[test]
    fn system_counts_toward_the_prefix() {
        let c = ChatCache::default();
        let b = format!("{{\"system\":\"{}\",\"messages\":[]}}", "s".repeat(5_000)).into_bytes();
        let s = c.split(&b, None, 0, 1).unwrap();
        assert!(s.cache_write_bytes > 5_000);
        assert!(s.fresh_bytes < 100);
    }

    #[test]
    fn remembered_prefixes_are_bounded() {
        let c = ChatCache::default();
        for i in 0..(MAX_PREFIXES as u64 + 10) {
            c.split(&body("hello", 4_000 + i as usize), None, 0, i).unwrap();
        }
        assert!(c.prefixes.lock().unwrap().len() <= MAX_PREFIXES);
    }

    #[test]
    fn conversation_id_is_taken_from_the_path() {
        let p = "/api/organizations/o1/chat_conversations/abc-123/completion";
        assert_eq!(conversation_id(p), Some("abc-123"));
        assert_eq!(conversation_id("/api/organizations/o1/chat_conversations/abc?tree=True"), Some("abc"));
        assert_eq!(conversation_id("/api/organizations/o1/chat_conversations/"), None);
        assert_eq!(conversation_id("/api/event_logging/v2/batch"), None);
    }

    #[test]
    fn history_grows_with_each_turn_and_is_read_from_the_cache() {
        let c = ChatCache::default();
        let b = body("x".repeat(100).as_str(), 10_000);
        // Turn 1 sees no history; its prompt (102 bytes with quotes) and a 400-byte
        // reply become the history of turn 2.
        let t1 = c.split(&b, Some("conv-a"), 400, 1_000).unwrap();
        assert_eq!(t1.cache_read_bytes, 0);
        let t2 = c.split(&b, Some("conv-a"), 300, 2_000).unwrap();
        let prefix = t1.cache_write_bytes;
        assert_eq!(t2.cache_read_bytes, prefix + 102 + 400);
        assert_eq!(t2.cache_write_bytes, 0);
        let t3 = c.split(&b, Some("conv-a"), 0, 3_000).unwrap();
        assert_eq!(t3.cache_read_bytes, prefix + (102 + 400) + (102 + 300));
    }

    #[test]
    fn conversations_do_not_share_history() {
        let c = ChatCache::default();
        let b = body("hello", 10_000);
        let a = c.split(&b, Some("conv-a"), 500, 1_000).unwrap();
        let prefix = a.cache_write_bytes;
        // conv-b has no history of its own yet: it reads only the shared prefix.
        let b1 = c.split(&b, Some("conv-b"), 0, 2_000).unwrap();
        assert_eq!(b1.cache_read_bytes, prefix);
        // Its second turn reads its own first prompt (7 bytes with quotes), not conv-a's reply.
        let b2 = c.split(&b, Some("conv-b"), 0, 3_000).unwrap();
        assert_eq!(b2.cache_read_bytes, prefix + 7);
    }

    #[test]
    fn idle_history_is_written_again() {
        let c = ChatCache::default();
        let b = body("hello", 10_000);
        c.split(&b, Some("conv-a"), 500, 1_000).unwrap();
        let later = c.split(&b, Some("conv-a"), 0, 1_000 + 2 * CACHE_TTL_MS).unwrap();
        // The prefix and the history both expired, so both are written again.
        assert_eq!(later.cache_read_bytes, 0);
        assert!(later.cache_write_bytes > 10_000 + 500);
    }

    #[test]
    fn remembered_conversations_are_bounded() {
        let c = ChatCache::default();
        let b = body("hello", 4_000);
        for i in 0..(MAX_CONVERSATIONS as u64 + 10) {
            c.split(&b, Some(&format!("conv-{}", i)), 10, i).unwrap();
        }
        assert!(c.conversations.lock().unwrap().len() <= MAX_CONVERSATIONS);
    }

    fn loaded_chat(leaf: &str) -> String {
        format!(
            r#"{{"uuid":"c","chat_messages":[
            {{"uuid":"m1","parent_message_uuid":"root","sender":"human","content":[{{"type":"text","text":"hello there"}}]}},
            {{"uuid":"m2","parent_message_uuid":"m1","sender":"assistant","content":[{{"type":"text","text":"hi!"}},{{"type":"thinking","thinking":"not counted"}}]}},
            {{"uuid":"m3","parent_message_uuid":"m2","sender":"human","content":[{{"type":"text","text":"again"}}],"attachments":[{{"extracted_content":"abcd"}}]}},
            {{"uuid":"m3b","parent_message_uuid":"m2","sender":"human","content":[{{"type":"text","text":"an edited branch that is not active"}}]}}
            ]{}}}"#,
            if leaf.is_empty() { String::new() } else { format!(",\"current_leaf_message_uuid\":\"{}\"", leaf) }
        )
    }

    #[test]
    fn a_loaded_chat_counts_its_active_branch_only() {
        // m3 (5 text + 4 attachment) + m2 (3, the thinking is skipped) + m1 (11).
        assert_eq!(history_bytes(loaded_chat("m3").as_bytes()), Some(23));
        // The other branch ends at the edited message instead.
        assert_eq!(history_bytes(loaded_chat("m3b").as_bytes()), Some(35 + 3 + 11));
        // No named leaf: every message.
        assert_eq!(history_bytes(loaded_chat("").as_bytes()), Some(23 + 35));
        assert_eq!(history_bytes(b"{\"chat_messages\":[]}"), Some(0));
        assert_eq!(history_bytes(b"{\"other\":1}"), None);
        assert_eq!(history_bytes(b"not json"), None);
    }

    #[test]
    fn tool_calls_and_results_count_but_thinking_and_metadata_do_not() {
        let b = br#"{"chat_messages":[
          {"uuid":"a","parent_message_uuid":"root","sender":"assistant","content":[
            {"type":"thinking","thinking":"a long private thought","signature":"zzzz"},
            {"type":"token_budget"},
            {"type":"tool_use","id":"t1","name":"web_search","input":{"query":"time"},"message":"Searching","icon_name":"globe"},
            {"type":"tool_result","tool_use_id":"t1","name":"web_search","is_error":false,"content":[
              {"type":"knowledge","title":"Local Time","url":"https://a.example/x","metadata":{"site_domain":"a.example","favicon_url":"https://f.example/icon.png"}}]},
            {"type":"tool_use","id":"t2","name":"jira_get","input":{"id":"X-1"}},
            {"type":"tool_result","tool_use_id":"t2","name":"jira_get","content":[{"type":"text","text":"issue body text"}]},
            {"type":"tool_result","tool_use_id":"t3","name":"plain","content":"a plain string result"}
          ]}]}"#;
        let tool_use_1 = "web_search".len() + r#"{"query":"time"}"#.len();
        let result_1 = "web_search".len() + "Local Time".len() + "https://a.example/x".len();
        let tool_use_2 = "jira_get".len() + r#"{"id":"X-1"}"#.len();
        let result_2 = "jira_get".len() + "issue body text".len();
        let result_3 = "plain".len() + "a plain string result".len();
        assert_eq!(
            history_bytes(b),
            Some((tool_use_1 + result_1 + tool_use_2 + result_2 + result_3) as u64)
        );
    }

    #[test]
    fn a_parent_cycle_does_not_loop() {
        let b = br#"{"chat_messages":[{"uuid":"a","parent_message_uuid":"b","content":[{"text":"xx"}]},{"uuid":"b","parent_message_uuid":"a","content":[{"text":"yy"}]}],"current_leaf_message_uuid":"a"}"#;
        assert_eq!(history_bytes(b), Some(4));
    }

    #[test]
    fn only_a_whole_conversation_get_is_a_load() {
        let base = "/api/organizations/o/chat_conversations/abc";
        assert!(is_conversation_load(Some("GET"), Some(base)));
        assert!(is_conversation_load(Some("GET"), Some(&format!("{}?tree=True&x=1", base))));
        assert!(!is_conversation_load(Some("POST"), Some(base)));
        assert!(!is_conversation_load(Some("GET"), Some(&format!("{}/completion", base))));
        assert!(!is_conversation_load(Some("GET"), Some("/api/organizations/o/chat_conversations?limit=5")));
        assert!(!is_conversation_load(None, Some(base)));
        assert!(!is_conversation_load(Some("GET"), None));
    }

    #[test]
    fn seeded_history_starts_cold_then_warms() {
        let c = ChatCache::default();
        let b = body("hello", 10_000);
        c.seed_history("conv-a", 5_000);
        let first = c.split(&b, Some("conv-a"), 100, 1_000).unwrap();
        // The reopened chat is not in the provider's cache: prefix and history are written.
        assert_eq!(first.cache_read_bytes, 0);
        assert!(first.cache_write_bytes > 10_000 + 5_000);
        let second = c.split(&b, Some("conv-a"), 0, 2_000).unwrap();
        assert_eq!(second.cache_write_bytes, 0);
        assert_eq!(second.cache_read_bytes, first.cache_write_bytes + 7 + 100);
    }

    #[test]
    fn a_load_replaces_a_live_total_but_keeps_its_cache_state() {
        let c = ChatCache::default();
        let b = body("hello", 10_000);
        let t1 = c.split(&b, Some("conv-a"), 400, 1_000).unwrap();
        // The load is what the server holds (here less than the live reply markup), and
        // the conversation stays warm, so its history is still read.
        c.seed_history("conv-a", 300);
        let t2 = c.split(&b, Some("conv-a"), 0, 2_000).unwrap();
        assert_eq!(t2.cache_read_bytes, t1.cache_write_bytes + 300);
        assert_eq!(t2.cache_write_bytes, 0);
        // A load that knows more (another client added turns) raises it.
        c.seed_history("conv-a", 9_000);
        let t3 = c.split(&b, Some("conv-a"), 0, 3_000).unwrap();
        assert_eq!(t3.cache_read_bytes, t1.cache_write_bytes + 9_000);
    }
}
