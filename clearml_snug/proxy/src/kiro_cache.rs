//! Prompt-cache split for Kiro's byte-estimated input.
//!
//! kiro-cli resends the whole prompt on every model call, and the provider behind
//! Kiro serves the part it saw before from its prompt cache: the second call of a
//! tool turn costs a fraction of the first's credits. The wire reports no tokens,
//! so the cached part is inferred the way a prefix cache works: the longest leading
//! run of prompt segments that an earlier request carried within the cache
//! lifetime, under the same requested model, is a cache read. The match is by
//! content, not by conversation, as a provider cache's is.
//!
//! The segments, in prompt order, per wire:
//! * the v3 engine (`POST /v1/responses`): `tools`, `instructions`, then each
//!   `input` item;
//! * the v2 engine, kiro-cli 3.x's default (`POST /` `GenerateAssistantResponse`):
//!   the current message's `tools`, each `history` message, then the current
//!   message. The current message comes back in the next request's history without
//!   its `envState` and `tools`, so a user message is keyed by its `content` and
//!   `toolResults` alone, which both shapes share.
//!
//! `tools` is keyed by its canonical form (object keys sorted): kiro-cli 3.1's v2
//! engine serializes each tool schema with a different key order on every call,
//! yet a tool turn's second call still costs about half the first's credits, as
//! on the v3 wire, so the provider sees the same tools.
//!
//! The rest of the cacheable prompt is a cache write when the caller says the
//! model bills one (Anthropic does; OpenAI charges no write premium), else fresh.
//! The request envelope (`metadata`, `envState`, ...) stays fresh. Only hashes of
//! the prefixes are kept, never text.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use serde::Deserialize;
use serde_json::value::RawValue;

pub use crate::claude_chat_cache::Split;
use crate::claude_chat_cache::{CACHE_TTL_MS, MIN_CACHEABLE_BYTES};

/// Bound on remembered prefixes, one per segment of every recent request. Past it
/// the expired ones are dropped, and if that is not enough, all of them: the cost
/// is one request's cache read reported as a write.
const MAX_PREFIXES: usize = 16384;

/// The cacheable fields of either wire's request, as exact wire bytes.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Request<'a> {
    #[serde(borrow, default)]
    tools: Option<&'a RawValue>,
    #[serde(borrow, default)]
    instructions: Option<&'a RawValue>,
    #[serde(borrow, default)]
    input: Option<Vec<&'a RawValue>>,
    #[serde(borrow, default)]
    conversation_state: Option<ConversationState<'a>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationState<'a> {
    #[serde(borrow, default)]
    history: Vec<&'a RawValue>,
    #[serde(borrow)]
    current_message: Message<'a>,
}

/// A v2 message: a user message carries `userInputMessage`, anything else (an
/// assistant reply) has none and is keyed whole.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Message<'a> {
    #[serde(borrow, default)]
    user_input_message: Option<UserInputMessage<'a>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UserInputMessage<'a> {
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
    #[serde(borrow, default)]
    user_input_message_context: Option<MessageContext<'a>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageContext<'a> {
    #[serde(borrow, default)]
    tools: Option<&'a RawValue>,
    #[serde(borrow, default)]
    tool_results: Option<&'a RawValue>,
}

/// One prompt segment: the hash it is matched by, and its wire bytes.
type Segment = (u64, u64);

fn raw(v: Option<&RawValue>) -> &str {
    v.map(RawValue::get).unwrap_or("")
}

/// A segment made of wire byte ranges, matched as they are.
fn segment(parts: &[&str]) -> Segment {
    let mut h = DefaultHasher::new();
    parts.hash(&mut h);
    (h.finish(), parts.iter().map(|p| p.len() as u64).sum())
}

/// The `tools` segment, matched by its canonical form: `Value` keeps object keys
/// sorted (serde_json without `preserve_order`).
fn tools_segment(tools: &RawValue) -> Segment {
    let canonical = serde_json::from_str::<serde_json::Value>(tools.get()).map(|v| v.to_string());
    let (hash, _) = segment(&[canonical.as_deref().unwrap_or(tools.get())]);
    (hash, tools.get().len() as u64)
}

/// A v2 user message as its `content` and `toolResults`; any other message whole.
fn v2_segment(item: &RawValue, msg: Option<UserInputMessage>) -> Segment {
    match msg {
        Some(u) => segment(&[raw(u.content), raw(u.user_input_message_context.and_then(|c| c.tool_results))]),
        None => segment(&[item.get()]),
    }
}

impl Request<'_> {
    /// The prompt segments in prompt order; `None` for neither chat wire.
    fn segments(self) -> Option<Vec<Segment>> {
        if let Some(input) = self.input {
            let tools = self.tools.map(tools_segment);
            let rest = self.instructions.into_iter().chain(input).map(|v| segment(&[v.get()]));
            return Some(tools.into_iter().chain(rest).collect());
        }
        let cs = self.conversation_state?;
        let current = cs.current_message.user_input_message?;
        let tools = current.user_input_message_context.as_ref().and_then(|c| c.tools);
        let mut segments: Vec<Segment> = tools.map(tools_segment).into_iter().collect();
        for item in cs.history {
            let msg = serde_json::from_str::<Message>(item.get()).ok()?.user_input_message;
            segments.push(v2_segment(item, msg));
        }
        segments.push(segment(&[
            raw(current.content),
            raw(current.user_input_message_context.and_then(|c| c.tool_results)),
        ]));
        Some(segments)
    }
}

/// Remembers the prompt prefixes of recent requests, to tell a cache read from a
/// write.
#[derive(Default)]
pub struct KiroCache {
    /// Hash of a prompt prefix (requested model plus its leading segments) -> last
    /// seen ms.
    seen: Mutex<HashMap<u64, u64>>,
}

impl KiroCache {
    /// Split a decompressed request body and record its prefixes. `model` is the
    /// requested model, which scopes the cache; `writes` says whether the served
    /// model bills cache writes. `None` when the body is neither chat wire's request
    /// (a body cut at the scan cap, a non-chat call) or its prompt is too small to
    /// cache, so the caller keeps its all-fresh estimate.
    pub fn split(&self, body: &[u8], model: &str, writes: bool, now_ms: u64) -> Option<Split> {
        let segments = serde_json::from_slice::<Request>(body).ok()?.segments()?;

        let mut h = DefaultHasher::new();
        model.hash(&mut h);
        let Ok(mut seen) = self.seen.lock() else { return None };
        let (mut cacheable, mut read) = (0u64, 0u64);
        for (seg_hash, seg_bytes) in segments {
            h.write_u64(seg_hash);
            let key = h.finish();
            cacheable += seg_bytes;
            if seen.get(&key).is_some_and(|&t| now_ms.saturating_sub(t) <= CACHE_TTL_MS) {
                read = cacheable;
            }
            seen.insert(key, now_ms);
        }
        if seen.len() > MAX_PREFIXES {
            seen.retain(|_, t| now_ms.saturating_sub(*t) <= CACHE_TTL_MS);
            if seen.len() > MAX_PREFIXES {
                seen.clear();
            }
        }
        drop(seen);

        if cacheable < MIN_CACHEABLE_BYTES {
            return None;
        }
        if read < MIN_CACHEABLE_BYTES {
            read = 0;
        }
        let write = if writes { cacheable - read } else { 0 };
        Some(Split {
            fresh_bytes: body.len() as u64 - read - write,
            cache_read_bytes: read,
            cache_write_bytes: write,
        })
    }
}

/// The process-wide cache: one proxy serves one app session.
pub fn global() -> &'static KiroCache {
    static CACHE: OnceLock<KiroCache> = OnceLock::new();
    CACHE.get_or_init(KiroCache::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOLS: &str = r#"[{"type":"function","name":"execute_bash","parameters":{}}]"#;

    /// A v3 Responses API request with `tools`, Kiro's context message padded to
    /// `ctx` bytes, and `items` after it.
    fn request(ctx: usize, items: &[&str]) -> Vec<u8> {
        let mut input = vec![format!(r#"{{"role":"user","content":"{}"}}"#, "c".repeat(ctx))];
        input.extend(items.iter().map(|s| s.to_string()));
        format!(
            r#"{{"model":"auto","input":[{}],"tools":{},"stream":true,"metadata":{{"kiro:conversation_id":"sess_1"}}}}"#,
            input.join(","),
            TOOLS
        )
        .into_bytes()
    }

    fn split(c: &KiroCache, body: &[u8], now: u64) -> Split {
        c.split(body, "auto", true, now).expect("a cacheable request")
    }

    fn total(s: &Split) -> u64 {
        s.fresh_bytes + s.cache_read_bytes + s.cache_write_bytes
    }

    const ASK: &str = r#"{"role":"user","content":"run ls"}"#;
    const CALL: &str = r#"{"type":"function_call","name":"execute_bash","arguments":"{}"}"#;
    const OUT: &str = r#"{"type":"function_call_output","output":"etc"}"#;

    #[test]
    fn tool_turn_reads_the_first_call_and_writes_the_new_items() {
        let c = KiroCache::default();
        let first = request(20_000, &[ASK]);
        let s1 = split(&c, &first, 0);
        assert_eq!(s1.cache_read_bytes, 0);
        assert!(s1.cache_write_bytes > 20_000, "{s1:?}");
        assert_eq!(total(&s1), first.len() as u64);

        let second = request(20_000, &[ASK, CALL, OUT]);
        let s2 = split(&c, &second, 1000);
        assert_eq!(s2.cache_read_bytes, s1.cache_write_bytes);
        assert_eq!(s2.cache_write_bytes, (CALL.len() + OUT.len()) as u64);
        assert_eq!(total(&s2), second.len() as u64);
    }

    #[test]
    fn instructions_are_part_of_the_prefix() {
        let c = KiroCache::default();
        let body = |instructions: &str| {
            format!(r#"{{"instructions":"{}","input":[{ASK}],"tools":{TOOLS}}}"#, instructions).into_bytes()
        };
        let s1 = split(&c, &body(&"i".repeat(5000)), 0);
        assert_eq!(s1.cache_read_bytes, 0);
        let s2 = split(&c, &body(&"i".repeat(5000)), 1);
        assert_eq!(s2.cache_read_bytes, s1.cache_write_bytes);
        let s3 = split(&c, &body(&"j".repeat(5000)), 2);
        assert_eq!(s3.cache_read_bytes, 0, "only the tools are shared, under the cacheable minimum");
    }

    #[test]
    fn a_new_chat_reads_the_shared_tools_and_context() {
        let c = KiroCache::default();
        split(&c, &request(20_000, &[ASK]), 0);
        let other = r#"{"role":"user","content":"something else"}"#;
        let s = split(&c, &request(20_000, &[other]), 1000);
        assert_eq!(s.cache_write_bytes, other.len() as u64);
        assert!(s.cache_read_bytes > 20_000, "{s:?}");
    }

    #[test]
    fn expiry_and_model_change_write_again() {
        let c = KiroCache::default();
        let body = request(20_000, &[ASK]);
        let s1 = split(&c, &body, 0);
        let late = split(&c, &body, CACHE_TTL_MS + 1);
        assert_eq!((late.cache_read_bytes, late.cache_write_bytes), (0, s1.cache_write_bytes));
        let other_model = c.split(&body, "claude-sonnet-4.5", true, CACHE_TTL_MS + 2).unwrap();
        assert_eq!(other_model.cache_read_bytes, 0);
    }

    #[test]
    fn a_model_without_write_billing_only_reads() {
        let c = KiroCache::default();
        let body = request(20_000, &[ASK]);
        let s1 = c.split(&body, "gpt-5.6-sol", false, 0).unwrap();
        assert_eq!((s1.cache_read_bytes, s1.cache_write_bytes, s1.fresh_bytes), (0, 0, body.len() as u64));
        let s2 = c.split(&body, "gpt-5.6-sol", false, 1).unwrap();
        assert_eq!(s2.cache_write_bytes, 0);
        assert!(s2.cache_read_bytes > 20_000, "{s2:?}");
    }

    #[test]
    fn small_or_foreign_bodies_are_not_split() {
        let c = KiroCache::default();
        let small = request(10, &[ASK]);
        assert!(c.split(&small, "auto", true, 0).is_none());
        for foreign in [
            &br#"{"origin":"AI_EDITOR","version":"1"}"#[..],
            br#"{"conversationState":{"conversationId":"c1"}}"#,
            br#"{"input":"a string, not a list"}"#,
        ] {
            assert!(c.split(foreign, "auto", true, 0).is_none());
        }
        let body = request(20_000, &[ASK]);
        assert!(c.split(&body[..body.len() / 2], "auto", true, 0).is_none());
    }

    /// A v2-engine `GenerateAssistantResponse` request, shaped like kiro-cli 3.1's:
    /// the tools and `envState` ride on the current message, the conversation is
    /// `history`.
    fn v2_request(history: &[&str], content: &str, tool_results: Option<&str>) -> Vec<u8> {
        let results = tool_results.map(|r| format!(r#","toolResults":{r}"#)).unwrap_or_default();
        format!(
            r#"{{"conversationState":{{"conversationId":"c1","history":[{}],"currentMessage":{{"userInputMessage":{{"content":"{}","userInputMessageContext":{{"envState":{{"operatingSystem":"linux"}}{},"tools":[{{"toolSpecification":{{"name":"shell","description":"{}"}}}}]}},"origin":"KIRO_CLI","modelId":"auto"}}}},"chatTriggerType":"MANUAL"}}}}"#,
            history.join(","),
            content,
            results,
            "d".repeat(20_000)
        )
        .into_bytes()
    }

    #[test]
    fn tools_match_whatever_their_key_order() {
        let c = KiroCache::default();
        let body = |tools: &str| format!(r#"{{"input":[{ASK}],"tools":{tools}}}"#).into_bytes();
        let pad = "p".repeat(5000);
        let a = format!(r#"[{{"name":"shell","parameters":{{"type":"object","description":"{pad}"}}}}]"#);
        let b = format!(r#"[{{"parameters":{{"description":"{pad}","type":"object"}},"name":"shell"}}]"#);
        let s1 = split(&c, &body(&a), 0);
        let s2 = split(&c, &body(&b), 1);
        assert_eq!(s2.cache_read_bytes, s1.cache_write_bytes);
    }

    #[test]
    fn v2_first_turn_without_history() {
        let c = KiroCache::default();
        let body = v2_request(&[], "hi", None);
        let s = split(&c, &body, 0);
        assert_eq!(s.cache_read_bytes, 0);
        assert!(s.cache_write_bytes > 20_000, "{s:?}");
        assert_eq!(total(&s), body.len() as u64);
    }

    #[test]
    fn v2_tool_turn_reads_the_previous_call_including_its_reshaped_message() {
        let c = KiroCache::default();
        let ctx = r#"{"userInputMessage":{"content":"agent context","origin":"KIRO_CLI"}}"#;
        let ack = r#"{"assistantResponseMessage":{"content":"ok"}}"#;
        let s1 = split(&c, &v2_request(&[ctx, ack], "run ls", None), 0);

        // The previous current message is now in history without envState/tools.
        let asked = r#"{"userInputMessage":{"content":"run ls","origin":"KIRO_CLI"}}"#;
        let call = r#"{"assistantResponseMessage":{"content":"","toolUses":[{"name":"shell"}]}}"#;
        let results = r#"[{"toolUseId":"t1","content":[{"text":"etc"}],"status":"success"}]"#;
        let second = v2_request(&[ctx, ack, asked, call], "", Some(results));
        let s2 = split(&c, &second, 1000);
        assert_eq!(s2.cache_read_bytes, s1.cache_write_bytes);
        assert_eq!(s2.cache_write_bytes, (call.len() + 2 + results.len()) as u64);
        assert_eq!(total(&s2), second.len() as u64);

        // The tool results come back in history with the message's context.
        let returned = format!(
            r#"{{"userInputMessage":{{"content":"","userInputMessageContext":{{"toolResults":{results}}},"origin":"KIRO_CLI"}}}}"#
        );
        let third = v2_request(&[ctx, ack, asked, call, &returned], "next", None);
        let s3 = split(&c, &third, 2000);
        assert_eq!(s3.cache_read_bytes, s2.cache_read_bytes + s2.cache_write_bytes);
    }

    #[test]
    fn the_bound_holds() {
        let c = KiroCache::default();
        for i in 0..(MAX_PREFIXES + 10) {
            let item = format!(r#"{{"role":"user","content":"{i}"}}"#);
            c.split(&request(4000, &[&item]), "auto", true, i as u64);
        }
        assert!(c.seen.lock().unwrap().len() <= MAX_PREFIXES);
    }
}
