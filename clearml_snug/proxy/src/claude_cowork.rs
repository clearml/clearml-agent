//! Usage of Claude Desktop's Cowork tab.
//!
//! Cowork runs each turn in a cloud Claude Code session, so the chat panel's
//! estimate does not apply: the server keeps a running total of everything the
//! session spent, in `external_metadata.usage` of the session object. That total
//! covers every API call of the session (the main thread, subagents, side models
//! such as the summary and title calls, interrupted turns), and the app is handed
//! the object for every session of the account, not only the open one:
//!
//! - `GET /v1/code/sessions/watch` is a long-lived SSE that pushes the full object
//!   as an `added`/`changed` event whenever a session changes,
//! - the session list and single-session reads return it too.
//!
//! This module turns those cumulative totals into increments: the first total
//! seen for a session is its baseline (what it spent before this app run is not
//! this run's), except for a session created after the proxy started, which
//! starts from zero. Re-seeing a total, a replay, or an out-of-date total adds
//! nothing, so every source can be observed freely.
//!
//! One call is in neither total: an interrupted turn ends in a `result` with zero
//! usage, and the server's total skips it, though the call was billed. The
//! session's own event stream (`GET /v1/code/sessions/{id}/events/stream`) carries
//! each call as it happens, so an aborted turn's calls are taken from there.
//!
//! Both streams are gzip with a sync flush after every event, so they are
//! inflated chunk by chunk. Any other encoding is left unmetered rather than
//! buffered forever.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use flate2::write::GzDecoder;
use serde_json::Value;

const SESSIONS: &str = "/v1/code/sessions";

fn bare(path: Option<&str>) -> Option<&str> {
    path.map(|p| p.split('?').next().unwrap_or(p))
}

/// Whether a request is the live feed of session changes.
pub fn is_session_feed(method: Option<&str>, path: Option<&str>) -> bool {
    method == Some("GET") && bare(path) == Some("/v1/code/sessions/watch")
}

/// The session id of a session's own event stream.
fn events_stream_session<'a>(method: Option<&str>, path: Option<&'a str>) -> Option<&'a str> {
    if method != Some("GET") {
        return None;
    }
    let (id, rest) = bare(path)?.strip_prefix("/v1/code/sessions/")?.split_once('/')?;
    (rest == "events/stream" && !id.is_empty()).then_some(id)
}

/// Whether a request lists or reads session objects: the list or one session
/// (not its events, which carry none).
pub fn is_session_read(path: Option<&str>) -> bool {
    match bare(path) {
        Some(p) => {
            p == SESSIONS
                || p.strip_prefix("/v1/code/sessions/")
                    .is_some_and(|id| !id.is_empty() && !id.contains('/') && id != "watch")
        }
        None => false,
    }
}

/// Token counts of one session, in the order they are summed over.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Counts {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

impl Counts {
    fn of(session: &Value) -> Option<Self> {
        let u = session.pointer("/external_metadata/usage")?;
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        Some(Self {
            input: n("input_tokens"),
            output: n("output_tokens"),
            cache_read: n("cache_read_tokens"),
            cache_write: n("cache_write_tokens"),
        })
    }

    fn covers(&self, o: &Self) -> bool {
        self.input >= o.input
            && self.output >= o.output
            && self.cache_read >= o.cache_read
            && self.cache_write >= o.cache_write
    }

    fn minus(&self, o: &Self) -> Self {
        Self {
            input: self.input.saturating_sub(o.input),
            output: self.output.saturating_sub(o.output),
            cache_read: self.cache_read.saturating_sub(o.cache_read),
            cache_write: self.cache_write.saturating_sub(o.cache_write),
        }
    }

    fn plus(&self, o: &Self) -> Self {
        Self {
            input: self.input + o.input,
            output: self.output + o.output,
            cache_read: self.cache_read + o.cache_read,
            cache_write: self.cache_write + o.cache_write,
        }
    }

    /// An Anthropic `usage` object (the API's field names).
    fn of_api(u: &Value) -> Self {
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        Self {
            input: n("input_tokens"),
            output: n("output_tokens"),
            cache_read: n("cache_read_input_tokens"),
            cache_write: n("cache_creation_input_tokens"),
        }
    }

    fn is_zero(&self) -> bool {
        *self == Self::default()
    }
}

/// What a session spent since it was last seen. `input` is the fresh (uncached)
/// input; the cache buckets are disjoint from it.
#[derive(Debug, PartialEq, Eq)]
pub struct Delta {
    pub session: String,
    pub model: Option<String>,
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

/// Sessions kept before the table starts over: a session seen again after that
/// is re-baselined, which only loses the one increment in between.
const MAX_SESSIONS: usize = 4096;

struct Tracker {
    start_ms: u64,
    seen: HashMap<String, Counts>,
}

impl Tracker {
    fn new(start_ms: u64) -> Self {
        Self { start_ms, seen: HashMap::new() }
    }

    fn observe(&mut self, session: &Value) -> Option<Delta> {
        let id = session.get("id")?.as_str()?;
        let now = Counts::of(session)?;
        let base = match self.seen.get(id) {
            Some(prev) if !now.covers(prev) => return None,
            Some(prev) => *prev,
            None => {
                if self.seen.len() >= MAX_SESSIONS {
                    self.seen.clear();
                }
                let created_here = session
                    .get("created_at")
                    .and_then(Value::as_str)
                    .and_then(parse_ms)
                    .is_some_and(|ms| ms >= self.start_ms);
                if created_here { Counts::default() } else { now }
            }
        };
        self.seen.insert(id.to_string(), now);
        let d = now.minus(&base);
        if d == Counts::default() {
            return None;
        }
        let model = ["/external_metadata/last_served_model", "/config/model"]
            .iter()
            .find_map(|p| session.pointer(p).and_then(Value::as_str))
            .map(str::to_string);
        Some(Delta {
            session: id.to_string(),
            model,
            input: d.input,
            output: d.output,
            cache_read: d.cache_read,
            cache_write: d.cache_write,
        })
    }
}

static TRACKER: OnceLock<Mutex<Tracker>> = OnceLock::new();

/// Start tracking; sessions created from `start_ms` on count from zero.
pub fn init(start_ms: u64) {
    let _ = TRACKER.set(Mutex::new(Tracker::new(start_ms)));
}

fn observe(session: &Value) -> Option<Delta> {
    TRACKER.get()?.lock().unwrap().observe(session)
}

/// Session objects inside a JSON response: the body itself, or wrapped (a list's
/// `data`, a single read's `response_shape`/`session`), found by shape so the
/// wrapper's name does not matter.
fn sessions_in<'a>(v: &'a Value, depth: u8, out: &mut Vec<&'a Value>) {
    if v.get("external_metadata").is_some() && v.get("id").is_some_and(Value::is_string) {
        out.push(v);
    } else if depth > 0 {
        match v {
            Value::Object(m) => m.values().for_each(|c| sessions_in(c, depth - 1, out)),
            Value::Array(a) => a.iter().for_each(|c| sessions_in(c, depth - 1, out)),
            _ => {}
        }
    }
}

/// Increments from a complete JSON response of the session list or one session.
pub fn observe_body(body: &[u8]) -> Vec<Delta> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    sessions_in(&v, 3, &mut found);
    found.into_iter().filter_map(observe).collect()
}

enum Decoder {
    Undecided,
    Plain,
    Gzip(GzDecoder<Vec<u8>>),
    Unsupported,
}

/// One API call of a session seen on its event stream.
#[derive(Default)]
struct Call {
    model: Option<String>,
    counts: Counts,
    /// Generated bytes streamed so far, from which an unfinished call's output is
    /// estimated.
    streamed_bytes: u64,
}

/// What a session's event stream has shown since its last `result`.
struct Events {
    session: String,
    calls: HashMap<String, Call>,
    /// The call the streamed deltas belong to (the latest `message_start`; with
    /// subagents streaming at once the attribution is approximate, which only
    /// affects the output estimate of an aborted turn).
    current: Option<String>,
}

enum Kind {
    Sessions,
    Events(Events),
}

/// Decoder and parser state of one live stream.
pub struct Feed {
    dec: Decoder,
    kind: Kind,
    /// Decoded text not yet ending in a blank line.
    pending: String,
    /// With the developer dump enabled, the name the decoded stream is appended
    /// under.
    pub dump_name: Option<String>,
}

/// Calls kept per session event stream between two results.
const MAX_CALLS: usize = 256;

/// `event_id`s of aborted results already reported, across streams: a reconnect
/// replays from the last sequence number, and an aborted turn must be counted
/// once. Cleared when it grows past `SEEN_CAP`.
const SEEN_CAP: usize = 4096;

fn first_sight(event_id: &str) -> bool {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let mut seen = SEEN.get_or_init(Default::default).lock().unwrap();
    if seen.len() >= SEEN_CAP {
        seen.clear();
    }
    seen.insert(event_id.to_string())
}

impl Events {
    fn on_block(&mut self, v: &Value) -> Option<Delta> {
        let payload = v.get("payload")?;
        match v.get("event_type")?.as_str()? {
            "stream_event" => self.on_stream_event(payload.get("event")?),
            "assistant" => self.on_assistant(payload.get("message")?),
            "result" => return self.on_result(v.get("event_id")?.as_str()?, payload),
            _ => {}
        }
        None
    }

    fn call(&mut self, id: &str) -> &mut Call {
        if self.calls.len() >= MAX_CALLS && !self.calls.contains_key(id) {
            self.calls.clear();
        }
        self.calls.entry(id.to_string()).or_default()
    }

    fn on_stream_event(&mut self, e: &Value) {
        match e.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let Some(m) = e.get("message") else { return };
                let Some(id) = m.get("id").and_then(Value::as_str) else { return };
                let model = m.get("model").and_then(Value::as_str).map(str::to_string);
                let counts = m.get("usage").map(Counts::of_api).unwrap_or_default();
                let c = self.call(id);
                c.model = model;
                c.counts = counts;
                self.current = Some(id.to_string());
            }
            Some("content_block_delta") => {
                let Some(d) = e.get("delta") else { return };
                let bytes = ["text", "thinking", "partial_json"]
                    .iter()
                    .filter_map(|k| d.get(k).and_then(Value::as_str))
                    .map(str::len)
                    .sum::<usize>() as u64;
                if let Some(c) = self.current.clone().and_then(|id| self.calls.get_mut(&id)) {
                    c.streamed_bytes += bytes;
                }
            }
            Some("message_delta") => {
                let out = e.pointer("/usage/output_tokens").and_then(Value::as_u64);
                if let (Some(out), Some(c)) =
                    (out, self.current.clone().and_then(|id| self.calls.get_mut(&id)))
                {
                    c.counts.output = out;
                }
            }
            _ => {}
        }
    }

    /// The finished `assistant` message repeats the call's input counts; its output
    /// is only a snapshot, so the larger of what was seen wins.
    fn on_assistant(&mut self, m: &Value) {
        let Some(id) = m.get("id").and_then(Value::as_str) else { return };
        let Some(u) = m.get("usage") else { return };
        let model = m.get("model").and_then(Value::as_str).map(str::to_string);
        let seen = Counts::of_api(u);
        let c = self.call(id);
        c.model = model.or(c.model.take());
        c.counts = Counts { output: seen.output.max(c.counts.output), ..seen };
    }

    /// A turn that ended in an error (an interrupt included) carries no, or only
    /// partial, usage in its `result`, and the server's session total skips it.
    /// The calls streamed since the previous result are the turn's real spend.
    fn on_result(&mut self, event_id: &str, payload: &Value) -> Option<Delta> {
        let calls = std::mem::take(&mut self.calls);
        self.current = None;
        if payload.get("subtype").and_then(Value::as_str) == Some("success")
            || !first_sight(event_id)
        {
            return None;
        }
        let accounted = payload.get("usage").map(Counts::of_api).unwrap_or_default();
        let mut spent = Counts::default();
        let mut model = None;
        for c in calls.into_values() {
            let est = crate::tokens::estimate_tokens(c.streamed_bytes, "claude", c.model.as_deref());
            spent = spent.plus(&Counts { output: c.counts.output.max(est), ..c.counts });
            model = c.model.or(model);
        }
        let d = spent.minus(&accounted);
        (!d.is_zero()).then(|| Delta {
            session: self.session.clone(),
            model,
            input: d.input,
            output: d.output,
            cache_read: d.cache_read,
            cache_write: d.cache_write,
        })
    }
}

impl Feed {
    /// The feed for a request that is one of the two streams, else `None`.
    pub fn for_request(method: Option<&str>, path: Option<&str>) -> Option<Self> {
        let kind = if is_session_feed(method, path) {
            Kind::Sessions
        } else {
            Kind::Events(Events {
                session: events_stream_session(method, path)?.to_string(),
                calls: HashMap::new(),
                current: None,
            })
        };
        Some(Self { dec: Decoder::Undecided, kind, pending: String::new(), dump_name: None })
    }

    /// Feed the next response bytes; returns what was found to be spent in them.
    pub fn feed(&mut self, wire: &[u8]) -> Vec<Delta> {
        let Some(text) = self.decode(wire) else {
            return Vec::new();
        };
        if let Some(name) = &self.dump_name {
            crate::dump::append(name, &text);
        }
        self.pending.push_str(&String::from_utf8_lossy(&text));
        let mut out = Vec::new();
        while let Some(end) = self.pending.find("\n\n") {
            let block: String = self.pending.drain(..end + 2).collect();
            let event = block.lines().find_map(|l| l.strip_prefix("event: "));
            let data = block
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .and_then(|d| serde_json::from_str::<Value>(d).ok());
            let Some(data) = data else { continue };
            match &mut self.kind {
                Kind::Sessions if matches!(event, Some("added" | "changed")) => {
                    out.extend(observe(&data));
                }
                Kind::Events(e) => out.extend(e.on_block(&data)),
                Kind::Sessions => {}
            }
        }
        out
    }

    fn decode(&mut self, wire: &[u8]) -> Option<Vec<u8>> {
        if wire.is_empty() {
            return None;
        }
        if matches!(self.dec, Decoder::Undecided) {
            self.dec = match crate::decompress::detect(wire) {
                crate::decompress::Encoding::Identity => Decoder::Plain,
                crate::decompress::Encoding::Gzip => Decoder::Gzip(GzDecoder::new(Vec::new())),
                _ => Decoder::Unsupported,
            };
        }
        match &mut self.dec {
            Decoder::Plain => Some(wire.to_vec()),
            Decoder::Gzip(g) => {
                // The flush hands over what the write only decoded into the
                // decoder's own buffer, so an event is seen as soon as it arrives.
                if g.write_all(wire).and_then(|_| g.flush()).is_err() {
                    self.dec = Decoder::Unsupported;
                    return None;
                }
                Some(std::mem::take(g.get_mut()))
            }
            Decoder::Undecided | Decoder::Unsupported => None,
        }
    }
}

/// Unix milliseconds of an RFC 3339 UTC timestamp (`2026-10-12T08:04:02.782558Z`).
fn parse_ms(s: &str) -> Option<u64> {
    let n = |r: std::ops::Range<usize>| -> Option<i64> { s.get(r)?.parse().ok() };
    let (y, mo, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (h, mi, sec) = (n(11..13)?, n(14..16)?, n(17..19)?);
    let frac: String = s
        .get(19..)?
        .strip_prefix('.')
        .map(|f| f.chars().take_while(char::is_ascii_digit).take(3).collect())
        .unwrap_or_default();
    let ms = format!("{frac:0<3}").parse::<i64>().ok()?;
    // Days since the epoch from the civil date (proleptic Gregorian).
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let days = era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
    u64::try_from(((days * 24 + h) * 60 + mi) * 60 * 1000 + sec * 1000 + ms).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    fn session(id: &str, created: &str, usage: Option<[u64; 4]>) -> Value {
        let mut s = serde_json::json!({
            "id": id,
            "created_at": created,
            "config": {"model": "claude-sonnet-5-5"},
            "external_metadata": {},
        });
        if let Some([i, o, r, w]) = usage {
            s["external_metadata"] = serde_json::json!({
                "last_served_model": "claude-opus-5-5",
                "usage": {"input_tokens": i, "output_tokens": o,
                          "cache_read_tokens": r, "cache_write_tokens": w, "cost_usd": 0.1},
            });
        }
        s
    }

    const WATCH: &str = "/v1/code/sessions/watch?exclude_tags=-";
    const OLD: &str = "2026-10-12T07:00:00.000000Z";
    const NEW: &str = "2026-10-12T09:00:00.000000Z";
    const START_MS: u64 = 1791792000000; // 2026-10-12T08:00:00Z

    fn delta(d: Option<Delta>) -> (u64, u64, u64, u64) {
        let d = d.expect("a delta");
        (d.input, d.output, d.cache_read, d.cache_write)
    }

    #[test]
    fn paths() {
        let g = Some("GET");
        assert!(is_session_feed(g, Some("/v1/code/sessions/watch?exclude_tags=-&resume_token=x")));
        assert!(!is_session_feed(Some("POST"), Some("/v1/code/sessions/watch")));
        assert!(!is_session_feed(g, Some("/v1/code/sessions/cse_1/events/stream")));
        assert!(is_session_read(Some("/v1/code/sessions?tags=cowork-remote&limit=100")));
        assert!(is_session_read(Some("/v1/code/sessions/cse_1")));
        assert!(!is_session_read(Some("/v1/code/sessions/cse_1/events?limit=100")));
        assert!(!is_session_read(Some("/v1/code/sessions/watch")));
        assert!(!is_session_read(Some("/v1/code/triggers?limit=100")));
        assert!(!is_session_read(None));
        assert_eq!(
            events_stream_session(g, Some("/v1/code/sessions/cse_1/events/stream?from_sequence_num=26")),
            Some("cse_1")
        );
        assert_eq!(events_stream_session(g, Some("/v1/code/sessions/cse_1/events?limit=100")), None);
        assert_eq!(events_stream_session(g, Some("/v1/code/sessions/watch")), None);
        assert_eq!(events_stream_session(Some("POST"), Some("/v1/code/sessions/cse_1/events/stream")), None);
    }

    #[test]
    fn rfc3339_to_ms() {
        assert_eq!(parse_ms("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_ms("2026-10-12T08:00:00.000000Z"), Some(START_MS));
        assert_eq!(parse_ms("2026-10-12T08:04:02.782558Z"), Some(START_MS + 242_782));
        assert_eq!(parse_ms("2024-03-01T00:00:00Z"), Some(1709251200000)); // after a leap day
        assert_eq!(parse_ms("garbage"), None);
    }

    #[test]
    fn an_old_session_is_baselined_a_new_one_starts_from_zero() {
        let mut t = Tracker::new(START_MS);
        // First sight of a session that predates the run: nothing to report yet.
        assert_eq!(t.observe(&session("cse_old", OLD, Some([100, 10, 5000, 700]))), None);
        assert_eq!(
            delta(t.observe(&session("cse_old", OLD, Some([103, 25, 9000, 700])))),
            (3, 15, 4000, 0)
        );
        // A session created during the run counts from its first total.
        assert_eq!(delta(t.observe(&session("cse_new", NEW, Some([2, 4, 84, 22])))), (2, 4, 84, 22));
    }

    #[test]
    fn a_repeated_or_stale_total_adds_nothing() {
        let mut t = Tracker::new(START_MS);
        let first = session("cse_a", NEW, Some([4, 100, 2000, 300]));
        assert!(t.observe(&first).is_some());
        assert_eq!(t.observe(&first), None, "same total again");
        assert_eq!(t.observe(&session("cse_a", NEW, Some([2, 50, 1000, 300]))), None, "older total");
        // The stale one must not have lowered the baseline: only the real growth counts.
        assert_eq!(delta(t.observe(&session("cse_a", NEW, Some([5, 120, 2500, 300])))), (1, 20, 500, 0));
    }

    #[test]
    fn a_session_without_usage_is_ignored_and_the_model_prefers_the_one_served() {
        let mut t = Tracker::new(START_MS);
        assert_eq!(t.observe(&session("cse_n", NEW, None)), None);
        let d = t.observe(&session("cse_n", NEW, Some([1, 1, 1, 1]))).unwrap();
        assert_eq!(d.model.as_deref(), Some("claude-opus-5-5"));
        let mut s = session("cse_m", NEW, Some([1, 1, 1, 1]));
        s["external_metadata"].as_object_mut().unwrap().remove("last_served_model");
        assert_eq!(t.observe(&s).unwrap().model.as_deref(), Some("claude-sonnet-5-5"));
    }

    #[test]
    fn bodies_are_searched_for_sessions_whatever_the_wrapper() {
        init(START_MS);
        let list = serde_json::json!({"data": [
            session("cse_b1", NEW, Some([1, 2, 3, 4])),
            session("cse_b2", NEW, Some([5, 6, 7, 8])),
        ], "resume_token": "x"});
        let got = observe_body(list.to_string().as_bytes());
        assert_eq!(got.len(), 2);
        let one = serde_json::json!({"response_shape": session("cse_b3", NEW, Some([1, 1, 1, 1]))});
        assert_eq!(observe_body(one.to_string().as_bytes()).len(), 1);
        assert!(observe_body(b"not json").is_empty());
        assert!(observe_body(b"{\"data\":[{\"id\":\"cse_x\"}]}").is_empty());
    }

    fn event(kind: &str, s: &Value) -> String {
        format!("event: {kind}\ndata: {s}\n\n")
    }

    #[test]
    fn the_watch_stream_is_read_event_by_event_across_chunk_splits() {
        init(START_MS);
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        for ev in [
            ":keepalive\n\n".to_string(),
            event("added", &session("cse_w", NEW, Some([1, 0, 0, 0]))),
            event("changed", &session("cse_w", NEW, Some([3, 10, 500, 20]))),
            "event: sync\nid: x\ndata: {}\n\n".to_string(),
        ] {
            enc.write_all(ev.as_bytes()).unwrap();
            enc.flush().unwrap(); // sync flush, like the server after every event
            chunks.push(std::mem::take(enc.get_mut()));
        }
        let mut f = Feed::for_request(Some("GET"), Some(WATCH)).unwrap();
        assert!(f.feed(&chunks[0]).is_empty());
        let added = f.feed(&chunks[1]);
        assert_eq!(added.len(), 1, "the first event is read before the stream ends");
        // Split the next event mid-way: nothing may be lost or doubled.
        let (a, b) = chunks[2].split_at(chunks[2].len() / 2);
        assert!(f.feed(a).is_empty());
        let changed = f.feed(b);
        assert_eq!(delta(changed.into_iter().next()), (2, 10, 500, 20));
        assert!(f.feed(&chunks[3]).is_empty());
        assert!(f.feed(&[]).is_empty());
    }

    #[test]
    fn an_unknown_encoding_reports_nothing() {
        let mut f = Feed::for_request(Some("GET"), Some(WATCH)).unwrap();
        assert!(f.feed(&[0x28, 0xb5, 0x2f, 0xfd, 0, 0]).is_empty());
        assert!(f.feed(event("added", &session("cse_z", NEW, Some([1, 1, 1, 1]))).as_bytes()).is_empty());
    }

    fn events(session: &str) -> Feed {
        Feed::for_request(Some("GET"), Some(&format!("/v1/code/sessions/{session}/events/stream"))).unwrap()
    }

    fn sse(kind: &str, event_id: &str, event_type: &str, payload: Value) -> String {
        let d = serde_json::json!({"event_id": event_id, "event_type": event_type, "payload": payload});
        format!("event: {kind}\ndata: {d}\n\n")
    }

    fn message_start(id: &str, usage: [u64; 4]) -> String {
        let [i, o, r, w] = usage;
        sse("ephemeral_event", "", "stream_event", serde_json::json!({"event": {
            "type": "message_start",
            "message": {"id": id, "model": "claude-sonnet-5-5", "usage": {
                "input_tokens": i, "output_tokens": o,
                "cache_read_input_tokens": r, "cache_creation_input_tokens": w}}}}))
    }

    fn text_delta(text: &str) -> String {
        sse("ephemeral_event", "", "stream_event", serde_json::json!({"event": {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text}}}))
    }

    fn result(event_id: &str, subtype: &str, usage: [u64; 4]) -> String {
        let [i, o, r, w] = usage;
        sse("client_event", event_id, "result", serde_json::json!({"subtype": subtype, "usage": {
            "input_tokens": i, "output_tokens": o,
            "cache_read_input_tokens": r, "cache_creation_input_tokens": w}}))
    }

    #[test]
    fn an_aborted_turns_streamed_calls_are_reported() {
        let mut f = events("cse_abort");
        let body = [
            message_start("msg_1", [2, 1, 95744, 9034]),
            text_delta(&"x".repeat(700)),
            text_delta(&"y".repeat(700)),
            result("r_abort", "error_during_execution", [0, 0, 0, 0]),
        ]
        .concat();
        let d = f.feed(body.as_bytes()).into_iter().next().expect("the aborted turn's spend");
        assert_eq!((d.input, d.cache_read, d.cache_write), (2, 95744, 9034));
        assert_eq!(d.output, crate::tokens::estimate_tokens(1400, "claude", Some("claude-sonnet-5-5")));
        assert_eq!((d.session.as_str(), d.model.as_deref()), ("cse_abort", Some("claude-sonnet-5-5")));
    }

    #[test]
    fn a_finished_message_gives_the_exact_output_and_a_successful_turn_adds_nothing() {
        let mut f = events("cse_ok");
        let done = sse("ephemeral_event", "", "stream_event", serde_json::json!({"event": {
            "type": "message_delta", "usage": {"output_tokens": 78}}}));
        let body = [
            message_start("msg_a", [2, 4, 100, 10]),
            text_delta("hello"),
            done.clone(),
            result("r_ok", "success", [2, 78, 100, 10]),
        ]
        .concat();
        assert!(f.feed(body.as_bytes()).is_empty(), "a successful result is in the session total");
        // The calls of the finished turn are forgotten: the next aborted turn is its own.
        let body = [message_start("msg_b", [3, 1, 7, 0]), done, result("r_ab2", "error_during_execution", [0, 0, 0, 0])].concat();
        let d = f.feed(body.as_bytes()).into_iter().next().unwrap();
        assert_eq!((d.input, d.output, d.cache_read, d.cache_write), (3, 78, 7, 0));
    }

    #[test]
    fn the_assistant_event_stands_in_when_no_deltas_were_seen_and_partial_usage_is_netted() {
        let mut f = events("cse_asst");
        let assistant = sse("client_event", "a1", "assistant", serde_json::json!({"message": {
            "id": "msg_c", "model": "claude-sonnet-5-5", "usage": {
                "input_tokens": 2, "output_tokens": 4,
                "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50}}}));
        // The result already accounts for 400 cache-read tokens of that call.
        let body = [assistant, result("r_part", "error_during_execution", [0, 0, 400, 0])].concat();
        let d = f.feed(body.as_bytes()).into_iter().next().unwrap();
        assert_eq!((d.input, d.output, d.cache_read, d.cache_write), (2, 4, 600, 50));
    }

    #[test]
    fn a_replayed_aborted_result_is_reported_once() {
        let body = [
            message_start("msg_r", [1, 1, 10, 0]),
            result("r_replay", "error_during_execution", [0, 0, 0, 0]),
        ]
        .concat();
        assert_eq!(events("cse_rp").feed(body.as_bytes()).len(), 1);
        assert!(events("cse_rp").feed(body.as_bytes()).is_empty());
    }
}
