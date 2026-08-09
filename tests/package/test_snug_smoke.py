"""End-to-end smoke tests for the LD_PRELOAD shim.

These only run when:
  - the host is Linux (LD_PRELOAD is a Linux mechanism for our purposes),
  - a built .so is actually present at the path the resolver finds.

On macOS / Windows / "no .so built" they are cleanly skipped, so CI on
those hosts is unaffected. The actual hard validation happens on the
Linux CI agent that built the shim a stage earlier.
"""
import base64
import json
import os
import platform
import re
import subprocess

import pytest

from clearml_agent.helper.snug import resolve_shim_path


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="LD_PRELOAD shim only loads on Linux",
)


@pytest.fixture(scope="module")
def shim_path():
    p = resolve_shim_path()
    if not p:
        pytest.skip(
            "no built shim available - run `cargo zigbuild --release` and "
            "copy the .so into clearml_agent/snug/lib/<arch>/ first"
        )
    return p


@pytest.fixture
def shimmed_env(shim_path):
    """A copy of os.environ with LD_PRELOAD pointing at our shim. Anything
    that was in LD_PRELOAD before is preserved (prepended-with-':' style),
    matching how the executioner builds the child env."""
    env = os.environ.copy()
    existing = env.get("LD_PRELOAD", "").strip()
    env["LD_PRELOAD"] = (
        shim_path if not existing else "{}:{}".format(shim_path, existing)
    )
    # Pin a known call-history mode for the init log line.
    env["CLEARML_SNUG_CALL_HISTORY"] = "off"
    return env


def _which(prog):
    """Best-effort 'which' that doesn't require shutil.which on py<3.3."""
    for d in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(d, prog)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# Pulled out as a helper because two tests want the same parsing logic.
_EVENT_LINE_RE = re.compile(r"^\[snug-event\] (.+)$")


def _parse_event_lines(stderr_text):
    """Walk stderr text, pull out every '[snug-event] {json}' line, parse
    each as JSON. Returns the list of decoded dicts in order. Any
    [snug-event] line that won't JSON-decode fails the test loudly."""
    events = []
    for line in stderr_text.splitlines():
        m = _EVENT_LINE_RE.match(line)
        if not m:
            continue
        payload = m.group(1)
        try:
            events.append(json.loads(payload))
        except ValueError as e:
            raise AssertionError(
                "shim emitted a [snug-event] line that's not valid JSON: {!r}\n"
                "parse error: {}".format(payload, e)
            )
    return events


def test_shim_loads_and_logs_init_for_curl(shimmed_env, tmp_path):
    """LD_PRELOAD=<shim> curl https://example.com succeeds AND:
       - the [snug] init line shows up (proves the ctor ran),
       - at least one structured event is emitted as valid JSON,
       - we see RequestStarted with host=example.com,
       - we see BytesObserved with the tx direction,
       - we see RequestCompleted on connection teardown."""
    if _which("curl") is None:
        pytest.skip("curl not on PATH")

    stderr_log = tmp_path / "err.log"
    result = subprocess.run(
        ["curl", "-sS", "--http1.1", "https://example.com", "-o", "/dev/null"],
        env=shimmed_env,
        stderr=open(str(stderr_log), "wb"),
        timeout=60,
    )

    log = stderr_log.read_text()
    assert result.returncode == 0, (
        "curl exited {} - shim broke the request.\nstderr:\n{}".format(
            result.returncode, log
        )
    )
    assert re.search(r"\[snug\] init pid=\d+", log), (
        "no init line in stderr:\n{}".format(log)
    )

    events = _parse_event_lines(log)
    assert events, "no [snug-event] lines in stderr:\n{}".format(log)

    request_started = [e for e in events if e.get("kind") == "RequestStarted"]
    assert request_started, (
        "no RequestStarted event in {} parsed events".format(len(events))
    )
    assert request_started[0]["host"] == "example.com"
    assert request_started[0]["method"] in ("GET", "HEAD")
    assert request_started[0]["path"].startswith("/")
    assert request_started[0]["whitelisted"] is False  # header injection turns this on
    assert request_started[0]["inject_headers"] is False

    bytes_observed = [e for e in events if e.get("kind") == "BytesObserved"]
    assert any(e["direction"] == "tx" for e in bytes_observed), (
        "no tx BytesObserved event"
    )
    # rx isn't strictly guaranteed (transient errors mid-handshake could
    # short-circuit), but on a healthy example.com fetch we should see it.
    assert any(e["direction"] == "rx" for e in bytes_observed), (
        "no rx BytesObserved event - did the response body never arrive?"
    )

    request_completed = [e for e in events if e.get("kind") == "RequestCompleted"]
    assert request_completed, (
        "no RequestCompleted on connection close - SSL_free hook didn't fire?"
    )
    assert request_completed[0]["bytes_tx"] > 0
    assert request_completed[0]["bytes_rx"] > 0


def test_shim_emits_diagnostic_on_http2(shimmed_env, tmp_path):
    """curl --http2 over a server that speaks h2 must produce exactly one
    http2_unsupported ShimDiagnostic for that connection, no RequestStarted,
    and the curl invocation still completes."""
    if _which("curl") is None:
        pytest.skip("curl not on PATH")

    stderr_log = tmp_path / "err.log"
    # nghttp2.org is a documented HTTP/2 reference; example.com may not
    # advertise h2. If --http2 fallbacks to h1, the diagnostic test is
    # vacuously skipped via the assertions below.
    result = subprocess.run(
        [
            "curl", "-sS", "--http2", "--max-time", "20",
            "https://nghttp2.org/", "-o", "/dev/null",
        ],
        env=shimmed_env,
        stderr=open(str(stderr_log), "wb"),
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(
            "curl --http2 to nghttp2.org failed in this CI env "
            "(network/proxy/curl-version) - non-shim issue"
        )

    log = stderr_log.read_text()
    events = _parse_event_lines(log)
    diagnostics = [
        e for e in events
        if e.get("kind") == "ShimDiagnostic"
        and e.get("kind_detail") == "http2_unsupported"
    ]
    if not diagnostics:
        # curl may have negotiated h1 despite --http2 (e.g. no ALPN
        # support in the local OpenSSL). Be lenient.
        pytest.skip("curl negotiated HTTP/1.1 despite --http2; cannot validate diagnostic")
    # At most one per connection by design.
    assert len(diagnostics) == 1, (
        "expected exactly one http2_unsupported diagnostic, got {}: {}".format(
            len(diagnostics), diagnostics
        )
    )
    # And no RequestStarted for the h2 connection.
    started = [e for e in events if e.get("kind") == "RequestStarted"]
    assert not started, (
        "RequestStarted emitted for an HTTP/2 connection - parser ran where "
        "it shouldn't: {}".format(started)
    )


def test_shim_doesnt_break_non_ssl_program(shimmed_env):
    """Loading the shim into a pure-bash program must not crash. The init
    line fires (the ctor runs), but no SSL hooks are exercised - so no
    [snug-event] lines, just the [snug] init line."""
    result = subprocess.run(
        ["bash", "-c", "echo hi"],
        env=shimmed_env,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        "bash exited {} - shim crashed a non-SSL program.\nstderr:\n{}".format(
            result.returncode, result.stderr.decode("utf-8", "replace")
        )
    )
    assert result.stdout.strip() == b"hi"
    # Stderr may contain [snug] init but nothing else from the shim.
    events = _parse_event_lines(result.stderr.decode("utf-8", "replace"))
    assert not events, (
        "shim emitted events for a non-SSL program (it shouldn't): {}".format(events)
    )


def test_inject_headers_reach_destination_via_httpbin(shimmed_env, shim_path, tmp_path):
    """When the whitelist matches with inject_headers=true, the destination
    server (httpbin.org/headers) echoes the received headers back in its
    JSON response - proving the splice produced a wire-correct request and
    the original byte count was returned to libssl without confusing it."""
    if _which("curl") is None:
        pytest.skip("curl not on PATH")

    # Build a per-test whitelist matching httpbin.org with injection on.
    wl = {
        "version": 1,
        "default_action": "meter",
        "rules": [{
            "host": "httpbin.org",
            "path_prefix": "/",
            "debug": False,
            "inject_headers": True,
            "tokenizer": "approx",
        }],
    }

    env = dict(shimmed_env)
    env["CLEARML_SNUG_WHITELIST"] = base64.b64encode(
        json.dumps(wl).encode("utf-8")
    ).decode("ascii")
    env["CLEARML_PROJECT_ID"] = "test-project-abc"
    env["CLEARML_TASK_ID"] = "test-task-xyz"

    stderr_log = tmp_path / "err.log"
    result = subprocess.run(
        ["curl", "-sS", "--http1.1", "--max-time", "20",
         "https://httpbin.org/headers"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=open(str(stderr_log), "wb"),
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(
            "httpbin.org unreachable from this env (network/proxy/etc.): "
            "exit={}, stderr=\n{}".format(
                result.returncode, stderr_log.read_text()
            )
        )

    try:
        response = json.loads(result.stdout.decode("utf-8"))
    except ValueError as e:
        pytest.skip(
            "httpbin.org didn't return JSON; non-shim issue. stdout:\n{}\n"
            "parse error: {}".format(result.stdout, e)
        )

    # httpbin returns {"headers": {"Header-Name": "value", ...}}; the
    # exact case of header keys depends on the server, so we normalize.
    received = {k.lower(): v for k, v in response.get("headers", {}).items()}
    assert received.get("project") == "test-project-abc", (
        "shim didn't inject the project header on the wire.\n"
        "httpbin received: {}\nshim stderr:\n{}".format(
            received, stderr_log.read_text()
        )
    )
    assert received.get("session") == "test-task-xyz", (
        "shim didn't inject the session header on the wire.\n"
        "httpbin received: {}".format(received)
    )

    # And the shim's own event stream agrees about what it did.
    log = stderr_log.read_text()
    events = _parse_event_lines(log)
    started = [e for e in events if e.get("kind") == "RequestStarted"]
    assert started, "no RequestStarted event in shim stderr"
    assert started[0]["whitelisted"] is True
    assert started[0]["inject_headers"] is True
    assert started[0]["host"] == "httpbin.org"


def test_request_started_unwhitelisted_when_no_rule_matches(shimmed_env, shim_path, tmp_path):
    """example.com isn't in our whitelist for this test -> whitelisted=False,
    inject_headers=False, no headers added on the wire."""
    if _which("curl") is None:
        pytest.skip("curl not on PATH")

    # Empty whitelist - example.com matches no rule.
    wl = {"version": 1, "default_action": "meter", "rules": []}

    env = dict(shimmed_env)
    env["CLEARML_SNUG_WHITELIST"] = base64.b64encode(
        json.dumps(wl).encode("utf-8")
    ).decode("ascii")
    env["CLEARML_PROJECT_ID"] = "test-project-abc"
    env["CLEARML_TASK_ID"] = "test-task-xyz"

    stderr_log = tmp_path / "err.log"
    result = subprocess.run(
        ["curl", "-sS", "--http1.1", "https://example.com", "-o", "/dev/null"],
        env=env,
        stderr=open(str(stderr_log), "wb"),
        timeout=60,
    )
    assert result.returncode == 0, stderr_log.read_text()

    events = _parse_event_lines(stderr_log.read_text())
    started = [e for e in events if e.get("kind") == "RequestStarted"]
    assert started
    assert started[0]["whitelisted"] is False
    assert started[0]["inject_headers"] is False


def test_python_https_request_emits_full_lifecycle(shimmed_env, tmp_path):
    """Real SSL traffic from a python3 process produces the full
    event lifecycle including RequestCompleted - via SSL_free during
    GC (the common path) or via the exit() hook flushing the state
    map at teardown (the safety net). One of them must land it."""
    if _which("python3") is None:
        pytest.skip("python3 not on PATH")

    stderr_log = tmp_path / "err.log"
    # No explicit close - teardown is driven by interpreter exit.
    script = (
        "import urllib.request, ssl;"
        " r = urllib.request.urlopen('https://example.com', timeout=20);"
        " _ = r.read();"
        " print('done')"
    )
    result = subprocess.run(
        ["python3", "-c", script],
        env=shimmed_env,
        stdout=subprocess.PIPE,
        stderr=open(str(stderr_log), "wb"),
        timeout=60,
    )
    log = stderr_log.read_text()
    if result.returncode != 0:
        pytest.skip(
            "python3 urllib request failed in this env (network/proxy/cert): "
            "exit={}, stderr=\n{}".format(result.returncode, log)
        )
    assert result.stdout.strip() == b"done"

    events = _parse_event_lines(log)
    assert events, "no [snug-event] lines from a python HTTPS request:\n{}".format(log)

    kinds = [e.get("kind") for e in events]
    assert "RequestStarted" in kinds, (
        "no RequestStarted - SSL_write hook didn't intercept python's request.\n"
        "events: {}\nstderr:\n{}".format(kinds, log)
    )
    assert "BytesObserved" in kinds, (
        "no BytesObserved - byte-counting path broken. events: {}".format(kinds)
    )
    assert "RequestCompleted" in kinds, (
        "no RequestCompleted - SSL_free didn't fire during interpreter "
        "shutdown AND our exit() hook's flush didn't emit either, so "
        "the trailing event was lost.\n"
        "events: {}\nstderr:\n{}".format(kinds, log)
    )


def test_libc_exit_hook_flushes_keepalive_connection(shimmed_env, tmp_path):
    """Verify the `exit(3)` hook emits the trailing RequestCompleted
    for a connection still in a keep-alive pool at process exit -
    the case where SSL_free never fires so observe_free can't.

    The script leaks a connection (keep response referenced, partial
    read, no close) then calls libc `exit(3)` via `ctypes.CDLL(None)`.
    `CDLL(None)` uses RTLD_DEFAULT semantics, so dlsym resolves `exit`
    through the LD_PRELOAD chain - our interposer wins over libc.

    Negative confirmation: a broken exit() hook would time out
    (process hangs on the real exit chain) or fail the RC assertion -
    no SSL_free fires for the leaked connection, so the hook is the
    only path that could produce the trailing event.
    """
    if _which("python3") is None:
        pytest.skip("python3 not on PATH")

    stderr_log = tmp_path / "err.log"
    script = (
        "import ctypes, urllib.request;"
        # Leak the connection: keep `r` referenced (no GC),
        # partial-read (no implicit close).
        " r = urllib.request.urlopen('https://example.com', timeout=20);"
        " _ = r.read(100);"
        # ctypes.CDLL(None) -> RTLD_DEFAULT; exit() resolves through
        # the LD_PRELOAD chain into our interposer.
        " ctypes.CDLL(None).exit(0)"
    )
    result = subprocess.run(
        ["python3", "-c", script],
        env=shimmed_env,
        stdout=subprocess.PIPE,
        stderr=open(str(stderr_log), "wb"),
        timeout=60,
    )
    log = stderr_log.read_text()
    if result.returncode != 0:
        pytest.skip(
            "python3 urllib request failed in this env (network/proxy/cert): "
            "exit={}, stderr=\n{}".format(result.returncode, log)
        )

    events = _parse_event_lines(log)
    assert events, (
        "no [snug-event] lines after libc exit() through LD_PRELOAD'd "
        "shim:\n{}".format(log)
    )

    kinds = [e.get("kind") for e in events]
    assert "RequestStarted" in kinds, (
        "no RequestStarted - SSL_write hook didn't fire on the urllib "
        "HTTPS request. events: {}\nstderr:\n{}".format(kinds, log)
    )
    # THE critical assertion. RequestCompleted in this scenario can
    # ONLY come from our exit() hook's flush_all_pending_requests
    # call - SSL_free didn't run (connection leaked into the keep-
    # alive pool), so observe_free has nothing to emit. The exit()
    # hook is the only remaining path that could produce the RC.
    assert "RequestCompleted" in kinds, (
        "no RequestCompleted after libc exit() - the exit() hook "
        "didn't flush the still-open keep-alive connection. This is "
        "the very case the exit() hook exists for; if it doesn't "
        "fire here, non-Python hosts will silently lose the trailing "
        "usage event.\nevents: {}\nstderr:\n{}".format(kinds, log)
    )


def test_only_allowed_symbols_globally_exported(shim_path):
    """Mirror of the CI nm guard - catches local-build regressions
    before they hit CI. Allowed global text symbols:

      * OpenSSL hooks: SSL_{write,read,free} + _ex variants
        (the _ex symbols are what Python 3.10's _ssl.so uses).
      * exit - libc exit(3); the exit-time flush. See hooks/exit.rs.
    """
    if _which("nm") is None:
        pytest.skip("nm not on PATH")

    out = subprocess.check_output(
        ["nm", "-D", "--defined-only", "--extern-only", shim_path],
        stderr=subprocess.STDOUT,
    ).decode("utf-8", "replace")

    allowed = re.compile(
        r"^[0-9a-f]+ T "
        r"(SSL_write|SSL_read|SSL_free|SSL_write_ex|SSL_read_ex|exit)$"
    )
    text_lines = [line for line in out.splitlines() if " T " in line]
    leaks = [line for line in text_lines if not allowed.match(line)]
    assert not leaks, "global text symbols leaked from shim:\n{}".format("\n".join(leaks))
