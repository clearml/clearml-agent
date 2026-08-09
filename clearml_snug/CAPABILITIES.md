# SNUG — capabilities and configuration

SNUG is an **optional, opt-in** component of the ClearML Agent that gives you
per-task visibility into the LLM traffic a task generates: which providers and
models it called, how many tokens each call consumed, how long each call took,
and how many bytes moved. It reports that back to ClearML as task scalars,
organization-wide usage events, or to an endpoint of your choice.

It requires **no changes to the task's code** and no SDK calls. The agent loads a
small in-process library into the task process; that library reads the plaintext
the process has already produced at its TLS library boundary, counts it, and
hands the result to a background reporter.

SNUG is **off by default** (`agent.snug.enabled: false`). With it off, the agent
behaves exactly as a build without it.

---

## Contents

- [What you get](#what-you-get)
- [Scope and guarantees](#scope-and-guarantees)
- [Supported platforms and providers](#supported-platforms-and-providers)
- [Configuration](#configuration)
  - [Quick start](#quick-start)
  - [`agent.snug` reference](#agentsnug-reference)
  - [The host whitelist](#the-host-whitelist)
  - [Environment overrides](#environment-overrides)
  - [Per-task runtime controls](#per-task-runtime-controls)
- [Token accuracy](#token-accuracy)
- [Reading the output](#reading-the-output)
- [Desktop app metering](#desktop-app-metering)
- [Limitations](#limitations)

---

## What you get

### Per-request LLM usage, with exact token counts where available

For every metered request SNUG records the host, method, path, HTTP status,
latency, plaintext bytes sent and received, the model, and token counts.

For the supported providers (Anthropic, OpenAI, Gemini) the token counts are the
provider's **own reported numbers**, read out of the response body — input,
output, and the cache-read / cache-write split where the provider reports it. For
any other host, or a response whose usage can't be read, SNUG falls back to a
byte-ratio estimate. Every record carries a flag saying which of the two it is,
so estimates are never silently mixed in with measured counts.

### Per-model attribution

The model is parsed from the response (falling back to the request), so usage is
attributed to `claude-haiku-4-5` or `gpt-4o`, not just to "Anthropic" or
"OpenAI". The model is both a reporting dimension and the grouping key for usage.

### Task scalars

With `report_task_metrics: true`, each metered request is plotted on the task's
own **SCALARS** tab. You get one chart per signal, with a series per
provider/model:

| Chart | What it plots |
|---|---|
| `LLM Input Tokens` | fresh (non-cached) prompt tokens |
| `LLM Cache Read Tokens` | prompt tokens served from the provider's cache |
| `LLM Cache Write Tokens` | prompt tokens written into the cache |
| `LLM Output Tokens` | completion tokens |
| `LLM Requests` | cumulative metered request count |
| `LLM Latency (ms)` | wall-clock request duration |
| `LLM Bytes Sent` / `LLM Bytes Received` | plaintext bytes per direction |
| `LLM Tool Calls (signal)` | 0 baseline, +1 when a request used a tool, −1 when a tool result errored |
| `LLM Tool Calls by Tool` | one series per tool name |
| `LLM Tool Calls (cumulative)` | running tool-call totals |
| `LLM Input Tokens (cumulative)` etc. | running total of each token chart above |

The three input charts are a **disjoint split** — fresh, cache-read, and
cache-write never double-count, so they sum to the provider's total prompt
tokens. Each of the four token charts also has a `(cumulative)` twin plotting a
running total. Pick which charts you want with `task_metrics_fields`,
(selecting a token field gives you both its per-call and cumulative chart).

### Organization-wide usage reporting

With `report_usage_events: true`, each completed request is also reported to the
ClearML server's LLM-usage endpoint (`routers.report_llm_usage`) as one event
carrying the model, provider, prompt/completion tokens, and the owning task, user
and project. This is what feeds cross-project usage reporting, as opposed to the
single task's own scalars.

### Forwarding to your own endpoint

Set `aggregator_url` and the reporter also POSTs each completed request to that
URL, batched as **newline-delimited JSON** (`Content-Type:
application/json-lines`, one event per line), flushed on a size trigger or a
periodic timer. Use this to ship usage into a system other than ClearML. It is
independent of the whitelist gate the other destinations apply.

### Request attribution headers

Hosts whose whitelist rule sets `inject_headers: true` get two headers spliced
into their outbound HTTP/1.x requests:

```
project: <ClearML project id>
session: <ClearML task id>
```

That lets an upstream gateway or provider-side proxy attribute a request back to
the ClearML project and task that made it. This is the **only** change SNUG makes
to a request's content.

**It is off in every shipped rule, and off by default for any rule you add.**
Injecting sends ClearML identifiers to whoever operates that host, so opting a
host in is deliberate — turn it on when something downstream actually reads these
headers, typically an LLM gateway you run yourself. Injection applies to HTTP/1.x
only; it does not happen on HTTP/2 connections or on the
[app-metering path](#desktop-app-metering).

### Call-history capture for debugging

SNUG can retain the last N full request/response pairs to LLM providers in a ring
buffer and print them, decoded, into the task log. Four modes:

| Mode | Behavior |
|---|---|
| `off` | capture nothing |
| `collect` | maintain the ring buffer, print nothing |
| `dump` | print the buffered backlog once, then revert to `collect` |
| `continuous` | print each request/response pair as it completes |

`dump` is one-shot: selecting it prints the backlog and the mode automatically
settles back to `collect`, so selecting `dump` again re-dumps the newest window.

**Credentials are redacted by default** — `Authorization`, `X-Api-Key`,
`X-Goog-Api-Key`, `Proxy-Authorization` and API keys passed as a `?key=` query
parameter are replaced with `<redacted>` before anything reaches the log.

### Live control per task

The capture mode, the extra hosts to meter, and the poll cadence can all be
changed **on a task that is already running**, from the ClearML UI, without
restarting anything. See [Per-task runtime controls](#per-task-runtime-controls).

---

## Scope and guarantees

**It does not weaken TLS.** SNUG reads the plaintext at the boundary of the TLS
library the task already uses. Certificate validation, protocol version, and
cipher selection all stay in that library, untouched. Nothing is downgraded and
no certificate is substituted.

**It meters only the hosts you allow.** Every connection is matched against the
whitelist; hosts with no matching rule follow `default_action`, which you set.
Setting `default_action: "ignore"` limits metering to exactly your rules.

**Your task's own ClearML traffic is excluded automatically.** The agent hands
SNUG the API/files/web hostnames of the ClearML backend it is talking to, and
connections to those hosts are suppressed regardless of whitelist rules — so a
task's own SDK calls are never counted as LLM usage.

**SNUG's own reporting is invisible to itself.** The reporter uses an independent
pure-Rust TLS stack that never calls the functions SNUG observes, so its backend
traffic is neither metered nor able to recurse.

**It cannot block or slow your task.** Events cross into the reporter over a
bounded in-process queue with drop-on-full semantics: if the reporter falls
behind, records are dropped rather than the task's I/O being stalled on the
network. There is no sidecar process and no socket.

**Bodies are read, not stored.** Response bodies are scanned for token usage and
the model, and are only retained when call-history capture is explicitly on (and
then redacted, capped per direction, and kept in a bounded ring buffer in
memory).

**One extra request header on metered hosts.** To read a response body without
shipping a decompressor into the task process, SNUG forces
`Accept-Encoding: identity` on whitelisted requests. Body parsing runs only when
a reporting destination is on; with all of them off, no body is parsed at all.

**Enabling it where it isn't available is a no-op, not an error.** If the
platform isn't supported, or the installed wheel carries no prebuilt binary,
`enabled: true` simply does nothing.

---

## Supported platforms and providers

| | Support |
|---|---|
| **Linux** | x86_64 and aarch64. Injected via `LD_PRELOAD`. Built against glibc 2.17 (manylinux2014), so it loads on current task images. |
| **macOS** | x86_64 and arm64. Injected via `DYLD_INSERT_LIBRARIES`. Whether a *specific* interpreter can be instrumented depends on System Integrity Protection and the hardened runtime; SNUG probes each interpreter and degrades gracefully when it can't. |
| **Windows** | Not supported — enabling SNUG is a no-op. |

| Execution mode | Support |
|---|---|
| **venv / native** | Yes. |
| **`--docker`** | Yes. The agent mounts the Linux library into the task container and points the task at it. Works from a macOS agent running a Linux task container. |
| **Kubernetes** | Task pods pick SNUG up through the agent configuration propagated into the pod, provided the pod image carries the agent wheel. |

**Runtimes:** any process whose TLS goes through a dynamically linked OpenSSL —
CPython, Node, and the usual provider SDKs on top of them. A runtime that
*statically* links its TLS stack has no boundary to read, and needs the separate
[app-metering path](#desktop-app-metering).

**Protocols:** HTTP/1.x is fully parsed (host, path, method, body). HTTP/2 is
metered for token usage from its DATA frames; because HTTP/2 request headers are
compressed, per-request host and path detail is limited on that path.

**Providers with exact usage and model parsing:** Anthropic, OpenAI, Gemini.
Any other host is still metered — bytes, latency, and estimated tokens — it just
has no provider-reported numbers to read.

---

## Configuration

All configuration lives under `agent.snug` in `clearml.conf`. Every key is
documented inline in the shipped `agent.conf`; this is the same surface, grouped
by what you are trying to do.

### Quick start

Meter the built-in LLM providers and chart usage on each task:

```hocon
agent {
    snug {
        enabled: true
        report_task_metrics: true
    }
}
```

Also feed organization-wide usage reporting:

```hocon
agent {
    snug {
        enabled: true
        report_task_metrics: true
        report_usage_events: true
    }
}
```

Meter *only* your own gateway, and nothing else:

```hocon
agent {
    snug {
        enabled: true
        report_task_metrics: true
        whitelist {
            version: 1
            default_action: "ignore"
            rules: [
                { host: "llm-gateway.internal", inject_headers: true, tokenizer: "cl100k" }
            ]
        }
    }
}
```

### `agent.snug` reference

**Master switch**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Load SNUG into task processes. When false, nothing else in this block has any effect. |

**Where usage is reported.** These are independent — enable any combination, or
none (in which case SNUG meters nothing and parses no bodies).

| Key | Default | Meaning |
|---|---|---|
| `report_task_metrics` | `false` | Report per-request usage to the task's own SCALARS tab. |
| `report_usage_events` | `false` | Report per-request usage to the ClearML server's LLM-usage endpoint, for organization-wide reporting. |
| `aggregator_url` | `null` | Forward each completed request, verbatim, to this URL as batched NDJSON. `null` disables it. |
| `task_metrics_fields` | all fields | Which signals become scalar charts. Remove entries to narrow the output; an empty or all-unknown list falls back to all fields. Use `report_task_metrics: false` to turn the sink off entirely. |

**Which hosts are metered**

| Key | Default | Meaning |
|---|---|---|
| `whitelist` | the three common LLM providers | Per-host rules and the fallback action. See [below](#the-host-whitelist). |
| `default_tokenizer` | `"approx"` | Estimator for connections no rule matched, so estimated tokens are populated for every metered request. A rule's own `tokenizer` overrides it per host. One of `claude`, `cl100k`, `approx`. |

**Debugging**

| Key | Default | Meaning |
|---|---|---|
| `call_history` | `"off"` | Starting capture mode: `off`, `collect`, `dump`, `continuous`. Switchable mid-task (see below). |
| `call_history_buffer` | `50` | How many most-recent request/response pairs the ring buffer retains. |
| `call_history_cap_bytes` | `262144` | Per-direction byte cap for a captured pair; past it the entry is flagged truncated. `0` = uncapped. |
| `poll_interval_sec` | `10` | How often (seconds) the reporter re-reads the task's runtime controls. |
| `debug_log` | `false` | Verbose per-process diagnostics. Errors are always logged regardless. |

**Desktop app metering**

| Key | Default | Meaning |
|---|---|---|
| `app_mode` | `""` | Profile id of a desktop app to meter (built-in: `claude_desktop`). Unset or `""` disables it. See [Desktop app metering](#desktop-app-metering). |

### The host whitelist

The whitelist decides, per host, whether traffic is metered and how. It is a
versioned block; the schema is `clearml_agent/snug/whitelist.schema.json`. This is
the shipped default — it meters the common providers and injects nothing:

```hocon
whitelist {
    version: 1
    default_action: "meter"
    rules: [
        { host: "api.openai.com",    path_prefix: "/v1/", inject_headers: false, tokenizer: "cl100k" }
        { host: "api.anthropic.com", path_prefix: "/v1/", inject_headers: false, tokenizer: "claude" }
        { host: "claude.ai",         path_prefix: "/",    inject_headers: false, tokenizer: "claude" }
        { host: "generativelanguage.googleapis.com", path_prefix: "/", inject_headers: false, tokenizer: "approx" }
    ]
}
```

`default_action` — what happens to a host **no** rule matched:

- `"meter"` — count its bytes and estimate its tokens anyway.
- `"ignore"` — pass it through untouched.

Rules are evaluated in order and **first match wins**. A rule's fields:

| Field | Default | Meaning |
|---|---|---|
| `host` | *(required)* | Matched case-insensitively against the request's `Host`. A leading and/or trailing `*` makes it a wildcard: `*.anthropic.com` (suffix), `api.anthropic.*` (prefix), `*anthropic*` (substring), `*` (any host). No `*` means an exact match. Only a boundary `*` is special — a `*` in the middle is literal. |
| `path_prefix` | `"/"` | Matched against the request-target path. |
| `inject_headers` | `false` | Splice the `project:` / `session:` attribution headers into this host's requests. |
| `tokenizer` | `"approx"` | Byte-ratio estimator for this host: `claude`, `cl100k`, or `approx`. Only used when the provider doesn't report exact usage. |

Set `rules: []` for an empty whitelist — hosts then follow `default_action` and
no headers are ever injected.

**How the whitelist merges across configuration layers.** The whitelist is the
one block that **merges** rather than being replaced when it appears in more than
one layer. An administrator-supplied whitelist is a **protected base**: a user's
`clearml.conf` can **add** hosts on top of it, but cannot remove or override an
admin rule (admin rules win on a host collision) and cannot change
`default_action` or `version`. So an organization can push hosts a user cannot
opt out of, while a user can still extend metering with hosts of their own.

### Environment overrides

Each of these overrides the corresponding config key, for cases where editing
`clearml.conf` isn't practical:

| Variable | Overrides |
|---|---|
| `CLEARML_AGENT_SNUG_ENABLED` | `enabled` |
| `CLEARML_AGENT_SNUG_REPORT_TASK_METRICS` | `report_task_metrics` |
| `CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS` | `report_usage_events` |
| `CLEARML_AGENT_SNUG_TASK_METRICS_FIELDS` | `task_metrics_fields` (comma-separated) |
| `CLEARML_AGENT_SNUG_CALL_HISTORY` | `call_history` |
| `CLEARML_AGENT_SNUG_DEBUG_LOG` | `debug_log` |

For development, `CLEARML_SNUG_SHIM_PATH` and `CLEARML_SNUG_PROXY_PATH` point the
agent at binaries other than the ones bundled in the wheel.

### Per-task runtime controls

Three **User Properties** on an individual task (in the ClearML UI: the task's
**CONFIGURATION → USER PROPERTIES**) steer SNUG for that task alone. They are
honored both **before launch** (applied from the task's first request) and **live
while the task runs** — the reporter re-reads them every `poll_interval_sec`. The
agent does not create these fields; add only the ones you want, and anything you
leave out keeps the configured default.

| Property | Effect |
|---|---|
| `_snug_call_history` | Capture mode for this task: `off`, `collect`, `dump`, `continuous`. Overrides the `call_history` default. `dump` prints the backlog once and reverts itself to `collect`, so re-selecting it re-dumps. |
| `_snug_whitelist` | **Add** hosts to meter for this task, on top of the configured whitelist. Admin rules still win on a collision and `default_action` is never changed. Clear the field to revert. |
| `_snug_user_properties_poll_rate` | How often (integer seconds) these properties are re-read. Defaults to `poll_interval_sec`. |

`_snug_whitelist` accepts either a host list:

```
api.my-llm.com, api.other.com
```

or a JSON array of full rules:

```json
[{"host": "api.my-llm.com", "inject_headers": true, "tokenizer": "cl100k"}]
```

Invalid input is rejected with a reason in the task log and the previous
whitelist is kept.

---

## Token accuracy

**Measured, when the provider reports it.** For Anthropic, OpenAI, and Gemini,
SNUG reads the `usage` object out of the response body and reports those exact
counts, including the cache-read / cache-write split. This is the normal case for
any task calling a provider's real API, and it is what the scalar and usage
destinations report.

**Estimated, otherwise.** For a host with no readable usage, tokens are derived
from plaintext byte counts. These are byte-ratio approximations, not
vocabulary-based tokenization:

| Tokenizer | Ratio | For |
|---|---|---|
| `claude` | ~2.7 B/token (current models) / ~3.5 (older) | Anthropic-shaped traffic |
| `cl100k` | ~4.0 B/token | OpenAI `cl100k_base` |
| `approx` | ~2.72 B/token | anything else |

Accuracy is roughly ±15% on English LLM traffic. Estimates are labelled as such
everywhere they appear, and the wire field is named `tokens_est` to keep the
approximation visible.

---

## Reading the output

SNUG's per-process output goes to the task log with a tag you can grep:

| Tag | Meaning |
|---|---|
| `[SNUG]` | lifecycle and per-request metering lines |
| `[SNUG-USAGE]` | one line per usage event queued for the server |
| `[SNUG-METRICS]` | scalar batches sent to the task's SCALARS tab |
| `[SNUG-AGG]` | batches forwarded to `aggregator_url` |
| `[SNUG-CALL]` | a decoded, redacted request/response pair from call history |
| `[SNUG-DIAG]` | diagnostics (only with `debug_log: true`) |
| `[SNUG-WARN]` | warnings; always logged |

At process teardown SNUG flushes and joins its reporter so the run's final
request is not lost. The flush is bounded by a timeout, so an unreachable backend
cannot wedge a task's exit. A task killed with `SIGKILL`, `_exit(2)`, or `abort`
skips the flush by design.

---

## Desktop app metering

Some desktop AI applications **statically link** their TLS stack into their own
binary. There is no library boundary to read, so the in-process mechanism above
cannot see their traffic at all.

For those, SNUG can instead run a **local forward proxy bound to the loopback
interface**. The agent starts the proxy, and configures the app's launcher and
bundled CLI to route their provider requests through it and to trust the proxy's
locally generated certificate authority. The proxy terminates the client leg,
validates the origin's real certificate on the upstream leg, relays bytes
verbatim, and parses usage with the **same** parsers and reports through the
**same** destinations as the in-process path — so metering is uniform either way.

This path is **separately opt-in** and off unless you name an application:

```hocon
agent.snug {
    enabled: true
    app_mode: "claude_desktop"
}
```

`app_mode` takes a **profile id**, not a boolean, because everything about
wiring an app in is app-specific. `claude_desktop` is the only built-in profile
today. An unrecognized id logs a line and does nothing.

The certificate authority is generated locally, persisted per machine, and never
leaves it. Setup is **scoped to the named application**: it does not alter the
task-wide environment, so the agent's own traffic and every other child process
are unaffected. Teardown restores the launcher, removes the certificate from the
stores it was added to, and stops the proxy.

---

## Limitations

- **Estimates for unsupported providers.** Only Anthropic, OpenAI, and Gemini
  wire formats are parsed for exact usage. Any other host falls back to
  byte-ratio estimates, and a genuinely new wire format needs code, not
  configuration.
- **HTTP/2 detail is limited.** Token usage is metered from HTTP/2 DATA frames,
  but request headers on that path are compressed and not decoded, so per-request
  host and path are less precise than on HTTP/1.x.
- **Consumer chat surfaces are estimated.** A consumer chat wire (as opposed to a
  provider's real API) generally reports no `usage` at all, so those tokens are
  byte-estimated and output counts in particular are approximate. This is a
  property of that wire, not something configuration can improve.
- **macOS injection is per-interpreter.** System Integrity Protection and the
  hardened runtime block injection into some interpreters; SNUG detects this and
  skips them rather than failing the task.
- **Windows is not supported.**
- **Desktop app metering does not run under `--docker`.** It runs on the
  native/in-container execute path, where the proxy and the app share a host.
- **`aggregator_url` consumers must accept NDJSON.** Events are batched, one JSON
  object per line — not one object per request.
- **QUIC / HTTP/3 bypasses the proxy path.** A client that reaches a provider over
  UDP/443 is not seen by the loopback proxy; disable QUIC in the client if
  complete coverage matters.
