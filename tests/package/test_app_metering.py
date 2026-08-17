"""Tests for the generic desktop-app metering glue in
clearml_agent.helper.app_metering.

These cover the PURE functions the worker wires together: the SDK wrapper
content, the SDK-dir discovery (ELF-vs-wrapper distinction), the idempotent
wrapper install (CA placement + rename), the Electron launcher wrapper, the NSS
trust install/remove, and the setup wiring. The proxy launch is integration-side
and exercised via the devloop; here we unit-test the deterministic file/string
logic plus the watcher body (``_run_watcher``) and its subprocess spawn with the
real ``Popen`` stubbed out.

The module was generalized from a single hardcoded "Claude Desktop" app to a
data-driven ``AppProfile`` registry (``BUILTIN_PROFILES``). Beyond porting the
Claude behavior tests 1:1, the ``-- genericity --`` block below proves the
mechanism is parameterized by profile/``SdkBinary``/``Launcher`` and not
Claude-hardcoded.
"""
import base64
import json
import os
import socket
import stat
import subprocess
import time

import pytest

import clearml_agent.helper.app_metering as am
import clearml_agent.snug.whitelist as wl


# A minimal fake ELF: the 4-byte magic plus some bytes so it looks like a binary.
_FAKE_ELF = b"\x7fELF" + b"\x00" * 60

# The Claude profile + its single watched SDK binary drive the fixtures below so
# they aren't hardcoded to string literals where the new API takes params.
_CLAUDE_PROFILE = am.BUILTIN_PROFILES["claude_desktop"]
_CLAUDE_SDK = _CLAUDE_PROFILE.sdk_binaries[0]
_CLAUDE_MARKER = am._launcher_marker("claude_desktop")
_CLAUDE_LAUNCHER_BASENAME = os.path.basename(_CLAUDE_PROFILE.launchers[0].path)


class _FakeConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeSession(object):
    """Minimal session whose ``.config.get`` build_whitelist_env reads."""

    def __init__(self, whitelist):
        self.config = _FakeConfig({"agent.snug.whitelist": whitelist})


def _make_sdk_dir(tmp_path, sdk=_CLAUDE_SDK, version="1.2.3",
                  pkg="node_modules/@anthropic/claude-code"):
    """Create <home>/<pkg>/<version>/<sdk.binary_name> as a fake ELF matching
    ``sdk``'s home_glob shape; return (home, sdk_dir)."""
    home = tmp_path / "home"
    sdk_dir = home / pkg / version
    sdk_dir.mkdir(parents=True)
    (sdk_dir / sdk.binary_name).write_bytes(_FAKE_ELF)
    return str(home), str(sdk_dir)


def _write_ca(tmp_path, body=b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n"):
    ca = tmp_path / "snug_proxy_ca.pem"
    ca.write_bytes(body)
    return str(ca)


def _make_launcher(tmp_path, name=_CLAUDE_LAUNCHER_BASENAME,
                   body="#!/bin/sh\nexec /opt/claude/electron \"$@\"\n"):
    """Create a fake original Electron launcher (a #!/bin/sh script by default,
    like the real .deb installs, WITHOUT our marker); return its path."""
    d = tmp_path / "usr-bin"
    d.mkdir(exist_ok=True)
    launcher = d / name
    launcher.write_text(body)
    os.chmod(str(launcher), 0o755)
    return str(launcher)


# -- render_sdk_wrapper ------------------------------------------------------

def test_render_sdk_wrapper_content():
    w = am.render_sdk_wrapper("http://127.0.0.1:8888", "snug_ca.pem", "claude")
    # Shebang + CA/proxy exports + loopback bypass + exec of the real binary.
    assert w.startswith("#!/bin/sh\n")
    assert 'export NODE_EXTRA_CA_CERTS="$DIR/snug_ca.pem"' in w
    assert 'export HTTPS_PROXY="http://127.0.0.1:8888"' in w
    assert 'export HTTP_PROXY="http://127.0.0.1:8888"' in w
    # Loopback bypass so an app's own 127.0.0.1 proxy isn't double-proxied.
    assert 'export NO_PROXY="localhost,127.0.0.1,::1"' in w
    assert 'export no_proxy="localhost,127.0.0.1,::1"' in w
    assert 'exec "$DIR/claude.real" "$@"' in w
    # Must NOT set SSL_CERT_FILE (bun ignores it and it would REPLACE the store).
    assert "SSL_CERT_FILE" not in w


def test_render_sdk_wrapper_uses_dirname_for_relocatability():
    # Everything is resolved relative to the wrapper ($(dirname "$0")) so it works
    # under bwrap's ro-bind at the SDK's real path.
    w = am.render_sdk_wrapper("http://127.0.0.1:9999", "ca.pem", "claude")
    assert 'DIR="$(dirname "$0")"' in w


# -- find_sdk_dirs -----------------------------------------------------------

def test_find_sdk_dirs_matches_elf(tmp_path):
    home, sdk_dir = _make_sdk_dir(tmp_path)
    found = am.find_sdk_dirs(home, _CLAUDE_SDK)
    assert found == [sdk_dir]


def test_find_sdk_dirs_skips_already_wrapped(tmp_path):
    # A dir whose `claude` is our shell wrapper (not an ELF) is skipped -> the
    # discovery is idempotent and won't re-wrap.
    home, sdk_dir = _make_sdk_dir(tmp_path)
    with open(os.path.join(sdk_dir, _CLAUDE_SDK.binary_name), "w") as fh:
        fh.write("#!/bin/sh\nexec ./claude.real\n")
    assert am.find_sdk_dirs(home, _CLAUDE_SDK) == []


def test_find_sdk_dirs_ignores_non_claude_code_dirs(tmp_path):
    # A `claude` binary that isn't under a *claude-code* package is not ours
    # (container_substr must appear in the grandparent dir name).
    home = tmp_path / "home"
    other = home / "node_modules" / "somethingelse" / "1.0.0"
    other.mkdir(parents=True)
    (other / _CLAUDE_SDK.binary_name).write_bytes(_FAKE_ELF)
    assert am.find_sdk_dirs(str(home), _CLAUDE_SDK) == []


def test_find_sdk_dirs_missing_home(tmp_path):
    assert am.find_sdk_dirs(str(tmp_path / "does-not-exist"), _CLAUDE_SDK) == []
    assert am.find_sdk_dirs("", _CLAUDE_SDK) == []


# -- install_sdk_wrapper -----------------------------------------------------

def test_install_sdk_wrapper_wraps_and_places_ca(tmp_path):
    home, sdk_dir = _make_sdk_dir(tmp_path)
    ca_src = _write_ca(tmp_path)

    wrapped = am.install_sdk_wrapper(sdk_dir, ca_src, "http://127.0.0.1:8888", _CLAUDE_SDK)
    assert wrapped is True

    # Real binary preserved under claude.real (still the ELF).
    real = os.path.join(sdk_dir, "claude.real")
    with open(real, "rb") as fh:
        assert fh.read(4) == b"\x7fELF"
    # ...and executable — the wrapper execs it, and Cowork may hand us the freshly
    # downloaded binary before it has chmod'd +x, so install must force it.
    assert os.stat(real).st_mode & stat.S_IXUSR

    # claude is now our wrapper, executable, with the expected content.
    claude = os.path.join(sdk_dir, "claude")
    with open(claude) as fh:
        body = fh.read()
    assert body.startswith("#!/bin/sh")
    assert 'exec "$DIR/claude.real" "$@"' in body
    mode = os.stat(claude).st_mode
    assert mode & stat.S_IXUSR

    # CA copied BESIDE the wrapper (not to /tmp) so bwrap's ro-bind sees it.
    ca_dst = os.path.join(sdk_dir, "snug_ca.pem")
    assert os.path.isfile(ca_dst)
    with open(ca_dst, "rb") as fh:
        assert b"BEGIN CERTIFICATE" in fh.read()


def test_install_sdk_wrapper_forces_executable_on_mode_600_download(tmp_path):
    # Cowork writes the freshly-downloaded SDK binary mode 0600 and chmods +x only
    # just before running it; the watcher can rename it into place during that
    # window. The wrapper execs claude.real, so install must force +x regardless
    # of the inherited mode (else "Claude Code crashed" on the exec).
    home, sdk_dir = _make_sdk_dir(tmp_path)
    ca_src = _write_ca(tmp_path)
    os.chmod(os.path.join(sdk_dir, "claude"), 0o600)  # rw------- , no execute bit

    assert am.install_sdk_wrapper(sdk_dir, ca_src, "http://127.0.0.1:8888", _CLAUDE_SDK) is True
    real = os.path.join(sdk_dir, "claude.real")
    mode = os.stat(real).st_mode
    assert mode & stat.S_IXUSR and mode & stat.S_IXGRP and mode & stat.S_IXOTH


def test_install_sdk_wrapper_idempotent(tmp_path):
    home, sdk_dir = _make_sdk_dir(tmp_path)
    ca_src = _write_ca(tmp_path)

    assert am.install_sdk_wrapper(sdk_dir, ca_src, "http://127.0.0.1:8888", _CLAUDE_SDK) is True
    # Second call: claude is already the wrapper (not an ELF) -> no-op, and the
    # real binary must not be clobbered.
    assert am.install_sdk_wrapper(sdk_dir, ca_src, "http://127.0.0.1:8888", _CLAUDE_SDK) is False
    with open(os.path.join(sdk_dir, "claude.real"), "rb") as fh:
        assert fh.read(4) == b"\x7fELF"


def test_install_sdk_wrapper_bails_without_ca_leaving_sdk_intact(tmp_path):
    # If the proxy hasn't written its CA yet, the install must touch NOTHING: the
    # SDK binary stays a runnable ELF at `claude` so the watcher can retry.
    home, sdk_dir = _make_sdk_dir(tmp_path)
    missing_ca = str(tmp_path / "not-there.pem")

    assert am.install_sdk_wrapper(sdk_dir, missing_ca, "http://127.0.0.1:8888", _CLAUDE_SDK) is False
    claude = os.path.join(sdk_dir, "claude")
    with open(claude, "rb") as fh:
        assert fh.read(4) == b"\x7fELF", "claude must remain the real ELF"
    assert not os.path.exists(os.path.join(sdk_dir, "claude.real"))


def test_install_sdk_wrapper_chowns_created_files_to_dir_owner_as_root(tmp_path, monkeypatch):
    # The agent runs as root but the app owns its SDK dir/binary as the desktop
    # user; a root-owned wrapper/CA is exactly what gave cowork EPERM/EACCES on its
    # own binary. install must chown the wrapper + .real + CA back to the SDK dir's
    # owner. Simulate root + a non-root-owned dir and capture chown calls.
    home, sdk_dir = _make_sdk_dir(tmp_path)
    ca_src = _write_ca(tmp_path)

    monkeypatch.setattr(am.os, "geteuid", lambda: 0)  # pretend we're root
    real_stat = am.os.stat

    class _St:
        st_uid = 4242
        st_gid = 4243
        st_mode = 0o40755

    def _fake_stat(path):
        # sdk_dir is "owned" by uid 4242; everything else keeps its real stat so
        # the wrapper's own chmod/read logic still works.
        if os.path.abspath(path) == os.path.abspath(sdk_dir):
            return _St()
        return real_stat(path)

    monkeypatch.setattr(am.os, "stat", _fake_stat)
    chowned = []
    monkeypatch.setattr(am.os, "chown", lambda p, u, g: chowned.append((os.path.basename(p), u, g)))

    assert am.install_sdk_wrapper(sdk_dir, ca_src, "http://127.0.0.1:8888", _CLAUDE_SDK) is True
    # wrapper (claude), preserved binary (claude.real), and the CA all handed to 4242:4243.
    assert ("claude", 4242, 4243) in chowned
    assert ("claude.real", 4242, 4243) in chowned
    assert ("snug_ca.pem", 4242, 4243) in chowned


def test_install_sdk_wrapper_no_chown_when_not_root(tmp_path, monkeypatch):
    # When the agent is NOT root, there is no privilege split to correct -> never
    # chown (the files are already the current user's).
    home, sdk_dir = _make_sdk_dir(tmp_path)
    ca_src = _write_ca(tmp_path)
    monkeypatch.setattr(am.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(am.os, "chown",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not chown when not root")))
    assert am.install_sdk_wrapper(sdk_dir, ca_src, "http://127.0.0.1:8888", _CLAUDE_SDK) is True


# -- _sdk_binary_ready (the anti-clobber gate) -------------------------------

def _ready(sdk_dir, bin_path, prev, cur):
    return am._sdk_binary_ready(sdk_dir, bin_path, prev, cur)


def test_sdk_binary_ready_false_when_not_executable(tmp_path):
    # The app chmods +x only after its own size check, so a non-executable binary
    # is still mid-install -> never wrap it.
    d = tmp_path / "sdk"; d.mkdir()
    b = d / "claude"; b.write_bytes(_FAKE_ELF); os.chmod(str(b), 0o644)
    sig = am._sdk_binary_sig(str(b))
    assert _ready(str(d), str(b), sig, sig) is False


def test_sdk_binary_ready_false_when_size_changing(tmp_path):
    # Executable but (size,mtime) changed since last tick -> still being written
    # (e.g. a fresh re-download) -> defer.
    d = tmp_path / "sdk"; d.mkdir()
    b = d / "claude"; b.write_bytes(_FAKE_ELF); os.chmod(str(b), 0o755)
    prev = (10, 111.0)
    cur = am._sdk_binary_sig(str(b))
    assert cur != prev
    assert _ready(str(d), str(b), prev, cur) is False


def test_sdk_binary_ready_true_when_executable_and_stable(tmp_path):
    # Executable AND unchanged since the previous tick -> install settled -> wrap.
    d = tmp_path / "sdk"; d.mkdir()
    b = d / "claude"; b.write_bytes(_FAKE_ELF); os.chmod(str(b), 0o755)
    sig = am._sdk_binary_sig(str(b))
    assert _ready(str(d), str(b), sig, sig) is True


def test_sdk_binary_ready_true_on_verified_marker_even_if_sig_differs(tmp_path):
    # The app's own `.verified` completion marker means it's done -> ready now,
    # without waiting a second stability tick (as long as it's executable).
    d = tmp_path / "sdk"; d.mkdir()
    b = d / "claude"; b.write_bytes(_FAKE_ELF); os.chmod(str(b), 0o755)
    open(str(d / ".verified"), "w").close()
    cur = am._sdk_binary_sig(str(b))
    assert _ready(str(d), str(b), None, cur) is True  # prev None (first sighting) but verified


def test_sdk_binary_ready_verified_still_requires_executable(tmp_path):
    # A stale `.verified` next to a not-yet-executable (still-downloading) binary
    # must NOT trigger a wrap.
    d = tmp_path / "sdk"; d.mkdir()
    b = d / "claude"; b.write_bytes(_FAKE_ELF); os.chmod(str(b), 0o644)
    open(str(d / ".verified"), "w").close()
    cur = am._sdk_binary_sig(str(b))
    assert _ready(str(d), str(b), cur, cur) is False


# -- candidate_home_roots ----------------------------------------------------

def test_candidate_home_roots_includes_home_and_home_users(tmp_path, monkeypatch):
    # Agent home plus every existing /home/* user home, deduped, existing-only.
    home = tmp_path / "agent-home"
    home.mkdir()
    user_a = tmp_path / "home" / "userA"
    user_b = tmp_path / "home" / "userB"
    user_a.mkdir(parents=True)
    user_b.mkdir(parents=True)
    missing = tmp_path / "home" / "gone"  # globbed but does not exist -> dropped

    monkeypatch.setattr(
        am.glob, "glob",
        lambda pat: [str(missing), str(user_b), str(user_a)] if pat == "/home/*" else [],
    )

    roots = am.candidate_home_roots(str(home))
    # Agent home first, then the sorted existing /home/* users; missing dropped.
    assert roots == [str(home), str(user_a), str(user_b)]


def test_candidate_home_roots_empty_home_and_dedup(tmp_path, monkeypatch):
    user_a = tmp_path / "home" / "userA"
    user_a.mkdir(parents=True)
    # Passing userA as the agent home too must not list it twice (realpath dedup).
    monkeypatch.setattr(
        am.glob, "glob",
        lambda pat: [str(user_a)] if pat == "/home/*" else [],
    )
    assert am.candidate_home_roots(str(user_a)) == [str(user_a)]
    # Falsy home contributes nothing; only the /home/* dirs remain.
    assert am.candidate_home_roots("") == [str(user_a)]


# -- resolve_app_profile -----------------------------------------------------

def test_resolve_app_profile_default_none():
    assert am.resolve_app_profile(_FakeConfig()) is None


def test_resolve_app_profile_named_selector():
    profile = am.resolve_app_profile(_FakeConfig({"agent.snug.app_mode": "claude_desktop"}))
    assert profile is not None
    assert profile.app_id == "claude_desktop"
    # A different / unknown app name does not enable claude-desktop mode.
    assert am.resolve_app_profile(_FakeConfig({"agent.snug.app_mode": "something"})) is None


def test_cd_log_default_always_prints(capsys, monkeypatch):
    # Errors/skips (the default, debug=False) must land on the console even
    # without the debug flag, so a failed metering setup is diagnosable.
    monkeypatch.delenv("CLEARML_SNUG_DEBUG_LOG", raising=False)
    am._cd_log("wrap failed for x")
    assert "[snug-app] wrap failed for x" in capsys.readouterr().out


def test_cd_log_debug_gated_by_env(capsys, monkeypatch):
    # Routine progress lines (debug=True) stay silent unless CLEARML_SNUG_DEBUG_LOG
    # is truthy, then print like any other [snug-app] line.
    monkeypatch.delenv("CLEARML_SNUG_DEBUG_LOG", raising=False)
    am._cd_log("watcher poll ...", debug=True)
    assert "watcher poll" not in capsys.readouterr().out

    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("CLEARML_SNUG_DEBUG_LOG", truthy)
        am._cd_log("watcher poll ...", debug=True)
        assert "[snug-app] watcher poll ..." in capsys.readouterr().out

    monkeypatch.setenv("CLEARML_SNUG_DEBUG_LOG", "0")
    am._cd_log("watcher poll ...", debug=True)
    assert "watcher poll" not in capsys.readouterr().out


# -- watcher -----------------------------------------------------------------

def _break_after_tick(monkeypatch):
    # _run_watcher wraps its sleep in `except (KeyboardInterrupt, SystemExit)`, so
    # raise one of those to exit the loop cleanly after exactly one tick.
    def _sleep(_secs):
        raise KeyboardInterrupt
    monkeypatch.setattr(am.time, "sleep", _sleep)


def test_run_watcher_wraps_found_dirs_on_a_tick(tmp_path, monkeypatch):
    # One real discovery+install tick must leave the SDK dir wrapped, then the
    # patched sleep breaks the loop. The binary is executable + carries the app's
    # `.verified` marker so it counts as settled on the single tick (see
    # _sdk_binary_ready).
    home, sdk_dir = _make_sdk_dir(tmp_path)
    ca_src = _write_ca(tmp_path)
    claude = os.path.join(sdk_dir, "claude")
    os.chmod(claude, 0o755)
    open(os.path.join(sdk_dir, ".verified"), "w").close()
    _break_after_tick(monkeypatch)

    am._run_watcher(home, ca_src, "http://127.0.0.1:8888", [_CLAUDE_SDK], poll_sec=0.01)

    with open(claude, "rb") as fh:
        assert fh.read(2) == b"#!", "watcher tick should have wrapped the settled SDK dir"


def test_run_watcher_defers_wrap_until_binary_settled(tmp_path, monkeypatch):
    # A binary that is NOT settled (not executable, no .verified) must NOT be
    # wrapped on the tick -- wrapping mid-install is exactly what broke cowork.
    home, sdk_dir = _make_sdk_dir(tmp_path)  # _FAKE_ELF written mode 0644, no .verified
    ca_src = _write_ca(tmp_path)
    _break_after_tick(monkeypatch)

    am._run_watcher(home, ca_src, "http://127.0.0.1:8888", [_CLAUDE_SDK], poll_sec=0.01)

    claude = os.path.join(sdk_dir, "claude")
    with open(claude, "rb") as fh:
        assert fh.read(4) == am._ELF_MAGIC, "watcher must NOT wrap an unsettled binary"
    assert not os.path.exists(claude + ".real"), "no rename should have happened"


def test_run_watcher_installs_each_found_dir(monkeypatch):
    # With discovery + the readiness gate stubbed, a single tick must call
    # install_sdk_wrapper once per (dir, binary) find_sdk_dirs reports
    # (union-deduped across roots), forwarding the ca + proxy + sdk args, then
    # exit on the patched sleep.
    monkeypatch.setattr(am, "candidate_home_roots", lambda home: ["/r1", "/r2"])
    monkeypatch.setattr(
        am, "find_sdk_dirs",
        lambda root, sdk: ["/r1/sdk"] if root == "/r1" else ["/r1/sdk", "/r2/sdk"],
    )
    monkeypatch.setattr(am, "_sdk_binary_ready", lambda *a, **k: True)
    installed = []
    monkeypatch.setattr(
        am, "install_sdk_wrapper",
        lambda sdk_dir, ca, proxy, sdk: installed.append((sdk_dir, ca, proxy, sdk)),
    )
    _break_after_tick(monkeypatch)

    am._run_watcher("/home/x", "/ca.pem", "http://127.0.0.1:8888", [_CLAUDE_SDK], poll_sec=0.01)

    # /r1/sdk appears under both roots but is wrapped once (union dedup by
    # (dir, binary_name)).
    assert installed == [
        ("/r1/sdk", "/ca.pem", "http://127.0.0.1:8888", _CLAUDE_SDK),
        ("/r2/sdk", "/ca.pem", "http://127.0.0.1:8888", _CLAUDE_SDK),
    ]


class _FakePopen(object):
    """Records argv/kwargs and the terminate/kill/wait/poll calls made against it."""

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True

    def poll(self):
        # Real Popen returns None while the child is alive, else its exit code.
        # This fake is "alive" until terminate()/kill() is called on it, same as
        # the liveness checks in read_ca_spki/setup_app_metering expect.
        return 0 if (self.terminated or self.killed) else None


def test_start_sdk_watcher_spawns_detached_subprocess(monkeypatch):
    created = {}

    def _fake_popen(argv, **kwargs):
        proc = _FakePopen(argv, **kwargs)
        created["proc"] = proc
        return proc

    monkeypatch.setattr(am.subprocess, "Popen", _fake_popen)

    handle = am.start_sdk_watcher(
        "/home/x", "/ca.pem", "http://127.0.0.1:8888", "claude_desktop", poll_sec=0.25
    )
    proc = created["proc"]

    # Runs THIS interpreter as `-m clearml_agent.helper.app_metering` with args,
    # so it is a process that survives the agent's os.execv (not a thread), and
    # re-resolves the profile from --app-id.
    assert proc.argv[0] == am.sys.executable
    assert proc.argv[1:3] == ["-m", "clearml_agent.helper.app_metering"]
    assert proc.argv[proc.argv.index("--app-id") + 1] == "claude_desktop"
    assert proc.argv[proc.argv.index("--home") + 1] == "/home/x"
    assert proc.argv[proc.argv.index("--ca") + 1] == "/ca.pem"
    assert proc.argv[proc.argv.index("--proxy-url") + 1] == "http://127.0.0.1:8888"
    assert proc.argv[proc.argv.index("--poll-sec") + 1] == "0.25"
    # Detached from the parent's process group so execv / a group signal can't
    # take it down.
    assert proc.kwargs.get("start_new_session") is True

    # .stop() terminates the spawned process.
    handle.stop()
    assert proc.terminated is True


# -- render_launcher_wrapper -------------------------------------------------

def test_render_launcher_wrapper_content():
    w = am.render_launcher_wrapper(
        "/usr/bin/claude-desktop-unofficial.real",
        "http://127.0.0.1:8888",
        "/home/u/.clearml_snug/snug_proxy_ca.pem",
        "SPKIb64==",
        _CLAUDE_MARKER,
    )
    assert w.startswith("#!/bin/sh\n")
    # Distinctive marker so idempotency can detect an already-wrapped launcher
    # even though the ORIGINAL launcher is itself a #!/bin/sh script.
    assert _CLAUDE_MARKER in w
    # The proxy + CA-trust env for the launched app.
    assert 'export HTTP_PROXY="http://127.0.0.1:8888"' in w
    assert 'export HTTPS_PROXY="http://127.0.0.1:8888"' in w
    assert 'export NODE_EXTRA_CA_CERTS="/home/u/.clearml_snug/snug_proxy_ca.pem"' in w
    # The three Chromium launch switches (the mechanism proven on the hardened build).
    assert "--proxy-server=http://127.0.0.1:8888" in w
    assert "--proxy-bypass-list=<-loopback>" in w
    assert "--ignore-certificate-errors-spki-list=SPKIb64==" in w
    # Execs the preserved real launcher and forwards the caller's own args.
    assert 'exec "/usr/bin/claude-desktop-unofficial.real"' in w
    assert '"$@"' in w
    # bun-only trap: must not REPLACE the store via SSL_CERT_FILE.
    assert "SSL_CERT_FILE" not in w


def test_render_launcher_wrapper_h2_assumed_host():
    # When the profile carries an h2 assumed-host, the wrapper exports it for the
    # app's dynamically-linked (shim-hooked) children; when None, it is absent.
    with_host = am.render_launcher_wrapper(
        "/l.real", "http://127.0.0.1:9", "/ca.pem", "S", _CLAUDE_MARKER,
        h2_assumed_host="api.anthropic.com",
    )
    assert 'export CLEARML_SNUG_H2_ASSUMED_HOST="api.anthropic.com"' in with_host

    without_host = am.render_launcher_wrapper(
        "/l.real", "http://127.0.0.1:9", "/ca.pem", "S", _CLAUDE_MARKER,
        h2_assumed_host=None,
    )
    assert "CLEARML_SNUG_H2_ASSUMED_HOST" not in without_host


def test_render_launcher_wrapper_appends_switches_before_args():
    # The switches sit between the real binary and "$@" so the caller's args keep
    # their trailing position (both are forwarded to the real launcher).
    w = am.render_launcher_wrapper("/l.real", "http://127.0.0.1:9", "/ca.pem", "S", _CLAUDE_MARKER)
    exec_line = [ln for ln in w.splitlines() if ln.startswith("exec ")][0]
    assert exec_line.index("--proxy-server=") < exec_line.index('"$@"')
    assert exec_line.index("--ignore-certificate-errors-spki-list=S") < exec_line.index('"$@"')
    assert exec_line.rstrip().endswith('"$@"')


def test_render_launcher_wrapper_is_shell_safe(tmp_path):
    # The wrapper is #!/bin/sh; --proxy-bypass-list's ``<-loopback>`` value MUST be
    # quoted or /bin/sh parses ``<-loopback>`` as an input redirection and the
    # wrapper dies ("cannot open -loopback") before it ever execs the launcher.
    # Substring checks miss this (the redirect is valid syntax), so execute the
    # wrapper against a stub launcher and confirm the switches actually arrive.
    argfile = tmp_path / "args.txt"
    real = tmp_path / "real.sh"
    real.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > ' + str(argfile) + '\n')
    real.chmod(0o755)
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(am.render_launcher_wrapper(
        str(real), "http://127.0.0.1:8888", "/ca.pem", "sPkI+/Base64==", _CLAUDE_MARKER,
    ))
    wrapper.chmod(0o755)
    res = subprocess.run(
        ["/bin/sh", str(wrapper), "--caller-arg"], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    assert "cannot open" not in res.stderr
    got = argfile.read_text().splitlines()
    assert "--proxy-server=http://127.0.0.1:8888" in got
    assert "--proxy-bypass-list=<-loopback>" in got
    assert "--ignore-certificate-errors-spki-list=sPkI+/Base64==" in got
    assert "--caller-arg" in got  # caller's own args still forwarded


# -- install_launcher_wrapper ------------------------------------------------

def test_install_launcher_wrapper_wraps_and_preserves_real(tmp_path):
    launcher = _make_launcher(tmp_path)
    ca_src = _write_ca(tmp_path)

    wrapped = am.install_launcher_wrapper(
        launcher, "http://127.0.0.1:8888", ca_src, "SPKIb64==", _CLAUDE_MARKER
    )
    assert wrapped is True

    # Original launcher preserved under <name>.real, still executable.
    real = launcher + ".real"
    with open(real) as fh:
        assert "exec /opt/claude/electron" in fh.read()
    assert os.stat(real).st_mode & stat.S_IXUSR

    # launcher is now our wrapper, executable, with the switches + exec of .real.
    with open(launcher) as fh:
        body = fh.read()
    assert body.startswith("#!/bin/sh")
    assert _CLAUDE_MARKER in body
    assert "--proxy-server=http://127.0.0.1:8888" in body
    assert "--proxy-bypass-list=<-loopback>" in body
    assert "--ignore-certificate-errors-spki-list=SPKIb64==" in body
    assert 'exec "{}"'.format(real) in body
    assert os.stat(launcher).st_mode & stat.S_IXUSR


def test_install_launcher_wrapper_idempotent(tmp_path):
    launcher = _make_launcher(tmp_path)
    ca_src = _write_ca(tmp_path)

    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", ca_src, "S", _CLAUDE_MARKER) is True
    # Second call: launcher is already our wrapper (marker present) -> no-op, and
    # the preserved real launcher must not be clobbered.
    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", ca_src, "S", _CLAUDE_MARKER) is False
    with open(launcher + ".real") as fh:
        assert "exec /opt/claude/electron" in fh.read()


def test_stale_wrapper_is_repointed_via_uninstall_then_install(tmp_path):
    # A wrapper left by a prior run points at a now-dead proxy's OLD port. Because
    # install_launcher_wrapper is a no-op on an already-wrapped launcher, a fresh
    # run must uninstall (restore the real binary) FIRST, then install again to
    # re-point at the CURRENT proxy port. This is what setup_app_metering does per
    # launcher so a stale wrapper never bricks a session.
    launcher = _make_launcher(tmp_path)
    ca_src = _write_ca(tmp_path)

    # Prior run: wrapped pointing at the old port 8888.
    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", ca_src, "S", _CLAUDE_MARKER) is True
    assert "8888" in open(launcher).read()
    # A naive re-wrap is a no-op and leaves the stale port in place.
    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:9999", ca_src, "S", _CLAUDE_MARKER) is False
    assert "8888" in open(launcher).read() and "9999" not in open(launcher).read()
    # The fix: restore first, then wrap -> now points at the new port, real binary
    # preserved, no stale .real chain.
    assert am.uninstall_launcher_wrapper(launcher, _CLAUDE_MARKER) is True
    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:9999", ca_src, "S", _CLAUDE_MARKER) is True
    body = open(launcher).read()
    assert "9999" in body and "8888" not in body
    with open(launcher + ".real") as fh:
        assert "exec /opt/claude/electron" in fh.read()


def test_install_launcher_wrapper_bails_without_spki(tmp_path):
    # SPKI not ready yet (empty) -> touch NOTHING so a retry/one-shot has a clean
    # original launcher to wrap, and no half-applied .real is left behind.
    launcher = _make_launcher(tmp_path)
    ca_src = _write_ca(tmp_path)

    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", ca_src, "", _CLAUDE_MARKER) is False
    with open(launcher) as fh:
        assert _CLAUDE_MARKER not in fh.read()
    assert not os.path.exists(launcher + ".real")


def test_install_launcher_wrapper_bails_without_ca(tmp_path):
    # CA cert not written yet -> bail without disturbing the launcher.
    launcher = _make_launcher(tmp_path)
    missing_ca = str(tmp_path / "not-there.pem")

    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", missing_ca, "S", _CLAUDE_MARKER) is False
    with open(launcher) as fh:
        assert _CLAUDE_MARKER not in fh.read()
    assert not os.path.exists(launcher + ".real")


def test_install_launcher_wrapper_bails_when_launcher_absent(tmp_path):
    ca_src = _write_ca(tmp_path)
    missing = str(tmp_path / "usr-bin" / _CLAUDE_LAUNCHER_BASENAME)
    assert am.install_launcher_wrapper(missing, "http://127.0.0.1:8888", ca_src, "S", _CLAUDE_MARKER) is False
    assert not os.path.exists(missing + ".real")


# -- uninstall_launcher_wrapper / restore ------------------------------------

def test_uninstall_launcher_wrapper_restores_original(tmp_path):
    launcher = _make_launcher(tmp_path)
    ca_src = _write_ca(tmp_path)
    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", ca_src, "S", _CLAUDE_MARKER) is True

    assert am.uninstall_launcher_wrapper(launcher, _CLAUDE_MARKER) is True
    # Original launcher back in place, our wrapper gone, .real removed.
    with open(launcher) as fh:
        body = fh.read()
    assert "exec /opt/claude/electron" in body
    assert _CLAUDE_MARKER not in body
    assert not os.path.exists(launcher + ".real")


def test_uninstall_launcher_wrapper_noop_when_not_wrapped(tmp_path):
    # An un-wrapped launcher (no marker) must be left untouched, and a missing
    # .real must not raise -> returns False.
    launcher = _make_launcher(tmp_path)
    assert am.uninstall_launcher_wrapper(launcher, _CLAUDE_MARKER) is False
    with open(launcher) as fh:
        assert "exec /opt/claude/electron" in fh.read()


def test_metering_handle_teardown_restores_launcher(tmp_path):
    # teardown() must restore each launcher we recorded, even with a live proxy
    # proc + no watcher, and must be safe/idempotent on a second call.
    launcher = _make_launcher(tmp_path)
    ca_src = _write_ca(tmp_path)
    assert am.install_launcher_wrapper(launcher, "http://127.0.0.1:8888", ca_src, "S", _CLAUDE_MARKER) is True

    proxy = _FakePopen(["proxy"])
    handle = am.AppMeteringHandle(
        proxy_proc=proxy,
        watcher=None,
        proxy_url="http://127.0.0.1:8888",
        ca_path=ca_src,
        launchers=[(launcher, _CLAUDE_MARKER, None)],
    )
    handle.teardown()

    with open(launcher) as fh:
        assert _CLAUDE_MARKER not in fh.read()
    assert not os.path.exists(launcher + ".real")
    assert proxy.terminated is True
    # Idempotent: a second teardown does nothing and does not raise.
    handle.teardown()


# -- read_ca_spki ------------------------------------------------------------

def test_read_ca_spki_reads_written_value(tmp_path):
    spki = tmp_path / "snug_proxy_ca.spki"
    spki.write_text("  Zm9vYmFyc3BraQ==\n")  # surrounding whitespace stripped
    assert am.read_ca_spki(str(spki), timeout=1.0, poll_sec=0.01) == "Zm9vYmFyc3BraQ=="


def test_read_ca_spki_tolerates_spki_prefix(tmp_path):
    # Defensive: the stdout spelling is `SPKI=<value>`; if the file carries that
    # prefix it is stripped so the launcher pins the bare value.
    spki = tmp_path / "snug_proxy_ca.spki"
    spki.write_text("SPKI=abc123==\n")
    assert am.read_ca_spki(str(spki), timeout=1.0, poll_sec=0.01) == "abc123=="


def test_read_ca_spki_bails_when_not_ready(tmp_path):
    # The proxy never wrote the file -> return None within ~timeout (not forever),
    # so the caller skips the launcher wrap instead of pinning an empty SPKI.
    missing = str(tmp_path / "never-written.spki")
    assert am.read_ca_spki(missing, timeout=0.05, poll_sec=0.01) is None


def test_read_ca_spki_waits_then_reads(tmp_path, monkeypatch):
    # Race: file absent on the first poll, present on the next -> read_ca_spki must
    # wait (not bail early) and then return the value.
    spki = tmp_path / "snug_proxy_ca.spki"
    calls = {"n": 0}
    real_sleep = am.time.sleep

    def _sleep(_secs):
        calls["n"] += 1
        if calls["n"] == 1:
            spki.write_text("late==\n")  # appears just after the first poll miss

    monkeypatch.setattr(am.time, "sleep", _sleep)
    assert am.read_ca_spki(str(spki), timeout=1.0, poll_sec=0.01) == "late=="
    assert calls["n"] >= 1
    _ = real_sleep  # keep a reference; not used but documents the swap


# -- read_ca_spki: proxy-liveness gate (proc/settle_sec) ---------------------
# The proxy writes CA + SPKI and starts the reporter/whitelist BEFORE its final
# listen bind (main.rs), so a losing instance under a port collision still
# leaves a fully-formed but orphaned SPKI on disk moments before exiting. These
# cover read_ca_spki's proc/settle_sec gate that catches that case.

class _StaticPoll(_FakePopen):
    """Fake Popen handle whose poll() always returns the same fixed value
    (None = alive, an int = that exit code). Inherits terminate/kill/wait from
    _FakePopen so it survives AppMeteringHandle.teardown()."""

    def __init__(self, returncode):
        super().__init__(["proxy"])
        self._returncode = returncode

    def poll(self):
        return self._returncode


def test_read_ca_spki_proc_none_skips_liveness_check(tmp_path):
    # Existing/direct callers that don't pass proc get the exact old behavior:
    # no liveness gate at all.
    spki = tmp_path / "snug_proxy_ca.spki"
    spki.write_text("noProcCheckSpki==\n")
    assert am.read_ca_spki(str(spki), timeout=1.0, poll_sec=0.01) == "noProcCheckSpki=="


def test_read_ca_spki_returns_value_when_proc_stays_alive(tmp_path):
    spki = tmp_path / "snug_proxy_ca.spki"
    spki.write_text("aliveSpki==\n")
    proc = _StaticPoll(None)  # never exits
    assert am.read_ca_spki(
        str(spki), timeout=1.0, poll_sec=0.01, proc=proc, settle_sec=0.02
    ) == "aliveSpki=="


def test_read_ca_spki_returns_none_when_proc_already_dead(tmp_path):
    # SPKI was written (the proxy got that far before dying), but it has already
    # exited by the time we check -- e.g. it lost a port-bind race. Must NOT
    # return the orphaned value: the caller would pin/trust/route through a dead
    # proxy, or worse, whatever else is actually answering that port.
    spki = tmp_path / "snug_proxy_ca.spki"
    spki.write_text("orphanedSpki==\n")
    proc = _StaticPoll(1)  # already exited with rc=1
    assert am.read_ca_spki(
        str(spki), timeout=1.0, poll_sec=0.01, proc=proc, settle_sec=0.02
    ) is None


def test_read_ca_spki_returns_none_when_proc_dies_during_settle_window(tmp_path):
    # Alive at the moment the SPKI is found, but dies partway through the settle
    # window -- must still be caught. This is the actual bind-failure timing: the
    # proxy writes CA/SPKI/whitelist, THEN calls bind(), so there's a real gap
    # between "SPKI is on disk" and "we know whether the bind succeeded".
    calls = {"n": 0}

    class _DiesOnThirdPoll(object):
        def poll(self):
            calls["n"] += 1
            return None if calls["n"] < 3 else 1

    spki = tmp_path / "snug_proxy_ca.spki"
    spki.write_text("dyingSpki==\n")
    proc = _DiesOnThirdPoll()
    assert am.read_ca_spki(
        str(spki), timeout=1.0, poll_sec=0.01, proc=proc, settle_sec=0.2
    ) is None


def test_read_ca_spki_fast_bails_when_proc_dead_and_no_file(tmp_path):
    # The proxy died before writing any SPKI (e.g. CA-write failure): with a large
    # timeout, read_ca_spki must NOT wait it out -- it returns None promptly once
    # it sees the proc is dead. This keeps the caller's per-attempt retry cheap.
    missing = str(tmp_path / "never-written.spki")
    proc = _StaticPoll(1)  # dead on arrival, never wrote the file
    start = time.time()
    assert am.read_ca_spki(
        str(missing), timeout=30.0, poll_sec=0.01, proc=proc
    ) is None
    assert time.time() - start < 1.0, "must fast-bail on a dead proc, not wait out timeout"


# -- install_ca_into_nss / remove_ca_from_nss --------------------------------
#
# These cover baking the proxy CA into the desktop user's NSS trust stores so
# the EXTERNAL Chrome that Claude Desktop opens for Google OAuth trusts the proxy CA.
# certutil is never invoked for real -- shutil.which and subprocess.run are
# stubbed and the calls asserted -- and _resolve_desktop_ids is stubbed to None
# (no privilege drop / chown) so the tests are deterministic regardless of
# whether the suite runs as root.

class _FakeCompleted(object):
    """Stand-in for subprocess.CompletedProcess (returncode + captured stdout)."""

    def __init__(self, returncode=0, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


def _certutil_op(argv):
    """The certutil operation flag (argv[0] is the certutil path)."""
    return argv[1] if len(argv) > 1 else ""


def _stub_certutil(monkeypatch, calls, rc_for=None, raise_exc=None, which="/usr/bin/certutil"):
    """Point install/remove at a fake certutil: record every argv, return rc_for
    (default 0), and never drop privileges/chown (_resolve_desktop_ids -> None)."""
    monkeypatch.setattr(am.shutil, "which", lambda name: which if name == "certutil" else None)
    monkeypatch.setattr(am, "_resolve_desktop_ids", lambda home, user=None: None)

    def _run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if raise_exc is not None:
            raise raise_exc
        rc = rc_for(argv) if rc_for is not None else 0
        return _FakeCompleted(returncode=rc)

    monkeypatch.setattr(am.subprocess, "run", _run)


def test_install_ca_into_nss_noop_when_certutil_missing(tmp_path, monkeypatch):
    # No certutil on PATH -> log + return False WITHOUT invoking subprocess.
    home = tmp_path / "home"
    home.mkdir()
    ca = _write_ca(tmp_path)
    monkeypatch.setattr(am.shutil, "which", lambda name: None)

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called when certutil is missing")

    monkeypatch.setattr(am.subprocess, "run", _boom)
    assert am.install_ca_into_nss(ca, str(home)) is False


def test_install_ca_into_nss_missing_ca_is_noop(tmp_path, monkeypatch):
    # certutil present but the CA cert isn't written yet -> return False, no calls.
    home = tmp_path / "home"
    home.mkdir()
    calls = []
    _stub_certutil(monkeypatch, calls)
    assert am.install_ca_into_nss(str(tmp_path / "nope.pem"), str(home)) is False
    assert calls == []


def test_install_ca_into_nss_adds_to_both_db_paths(tmp_path, monkeypatch):
    # The CA must be created + trusted for TLS in BOTH NSS DBs Chromium consults;
    # with no pre-existing DB, each dir gets init (-N) then a trusted add (-A).
    home = tmp_path / "home"
    home.mkdir()
    ca = _write_ca(tmp_path)
    calls = []
    _stub_certutil(monkeypatch, calls)

    assert am.install_ca_into_nss(ca, str(home), user="desktop") is True

    db1 = os.path.join(str(home), ".pki", "nssdb")
    db2 = os.path.join(str(home), ".local", "share", "pki", "nssdb")
    # Both DB dirs were created on disk.
    assert os.path.isdir(db1) and os.path.isdir(db2)

    for db in (db1, db2):
        ops_for_db = [_certutil_op(a) for a, _k in calls if ("sql:" + db) in a]
        # init, idempotent predelete, then the trusted add -- in that order.
        assert ops_for_db == ["-N", "-D", "-A"], (db, ops_for_db)
        add = [a for a, _k in calls if ("sql:" + db) in a and _certutil_op(a) == "-A"][0]
        assert "-t" in add and add[add.index("-t") + 1] == "C,,"
        assert "-n" in add and add[add.index("-n") + 1] == "clearml-snug-proxy"
        assert "-i" in add and add[add.index("-i") + 1] == ca


def test_install_ca_into_nss_skips_init_when_db_initialized(tmp_path, monkeypatch):
    # If the sql: DB is already initialized (cert9.db present), the add must be
    # idempotent: no -N, and a -D (delete) before the -A (re-add) on the stable
    # nickname so a re-run refreshes rather than erroring on a duplicate.
    home = tmp_path / "home"
    for sub in (".pki/nssdb", ".local/share/pki/nssdb"):
        d = home / sub
        d.mkdir(parents=True)
        (d / "cert9.db").write_bytes(b"\x00")  # sentinel: already initialized
    ca = _write_ca(tmp_path)
    calls = []
    _stub_certutil(monkeypatch, calls)

    assert am.install_ca_into_nss(ca, str(home)) is True

    ops = [_certutil_op(a) for a, _k in calls]
    assert "-N" not in ops, "init must be skipped when the DB already exists"
    db1 = os.path.join(str(home), ".pki", "nssdb")
    ops_db1 = [_certutil_op(a) for a, _k in calls if ("sql:" + db1) in a]
    assert ops_db1 == ["-D", "-A"], ops_db1


def test_install_ca_into_nss_graceful_on_certutil_failure(tmp_path, monkeypatch):
    # certutil -A returning nonzero (e.g. bad cert) must not raise; the function
    # reports False so the caller knows trust wasn't installed.
    home = tmp_path / "home"
    home.mkdir()
    ca = _write_ca(tmp_path)
    calls = []
    _stub_certutil(monkeypatch, calls, rc_for=lambda argv: 1 if _certutil_op(argv) == "-A" else 0)
    assert am.install_ca_into_nss(ca, str(home)) is False


def test_install_ca_into_nss_graceful_when_certutil_raises(tmp_path, monkeypatch):
    # If spawning certutil raises (e.g. OSError), install must swallow it, log,
    # and return False -- NSS trust is best-effort and must never break setup.
    home = tmp_path / "home"
    home.mkdir()
    ca = _write_ca(tmp_path)
    calls = []
    _stub_certutil(monkeypatch, calls, raise_exc=OSError("boom"))
    assert am.install_ca_into_nss(ca, str(home)) is False


def test_remove_ca_from_nss_deletes_from_existing_dbs(tmp_path, monkeypatch):
    # Teardown removal: -D the stable nickname from each NSS DB that exists.
    home = tmp_path / "home"
    db1 = home / ".pki" / "nssdb"
    db2 = home / ".local" / "share" / "pki" / "nssdb"
    db1.mkdir(parents=True)
    db2.mkdir(parents=True)
    calls = []
    _stub_certutil(monkeypatch, calls)

    assert am.remove_ca_from_nss(str(home)) is True
    deleted = [a for a, _k in calls if _certutil_op(a) == "-D"]
    assert len(deleted) == 2
    for a in deleted:
        assert a[a.index("-n") + 1] == "clearml-snug-proxy"
    # Both DB dirs were targeted.
    dbs = {a[a.index("-d") + 1] for a in deleted}
    assert dbs == {"sql:" + str(db1), "sql:" + str(db2)}


def test_remove_ca_from_nss_noop_when_certutil_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".pki" / "nssdb").mkdir(parents=True)
    monkeypatch.setattr(am.shutil, "which", lambda name: None)

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called when certutil is missing")

    monkeypatch.setattr(am.subprocess, "run", _boom)
    assert am.remove_ca_from_nss(str(home)) is False


def test_remove_ca_from_nss_skips_missing_dbs(tmp_path, monkeypatch):
    # A home with no NSS DBs at all -> nothing to delete, no certutil calls.
    home = tmp_path / "home"
    home.mkdir()
    calls = []
    _stub_certutil(monkeypatch, calls)
    assert am.remove_ca_from_nss(str(home)) is False
    assert calls == []


def test_metering_handle_teardown_removes_nss_ca(tmp_path, monkeypatch):
    # teardown() must remove the CA from NSS for the recorded home/user, and be
    # idempotent (a second teardown does not call remove again / does not raise).
    removed = []
    monkeypatch.setattr(
        am, "remove_ca_from_nss",
        lambda home, user=None: removed.append((home, user)) or True,
    )
    proxy = _FakePopen(["proxy"])
    handle = am.AppMeteringHandle(
        proxy_proc=proxy,
        watcher=None,
        proxy_url="http://127.0.0.1:8888",
        ca_path="/ca.pem",
        nss_home="/home/desktop",
        nss_user="desktop",
    )
    handle.teardown()
    assert removed == [("/home/desktop", "desktop")]
    assert proxy.terminated is True
    handle.teardown()  # idempotent: no second removal, no raise
    assert removed == [("/home/desktop", "desktop")]


def test_metering_handle_teardown_skips_nss_when_not_installed(monkeypatch):
    # If NSS trust was never installed (nss_home is None), teardown must NOT try
    # to remove it.
    called = []
    monkeypatch.setattr(am, "remove_ca_from_nss", lambda home, user=None: called.append(1))
    handle = am.AppMeteringHandle(
        proxy_proc=None, watcher=None, proxy_url="u", ca_path="/ca.pem",
    )
    handle.teardown()
    assert called == []


# -- setup_app_metering wiring for NSS ---------------------------------------

def _stub_setup_deps(monkeypatch, spki="SPKIval", nss=None):
    """Stub every side-effecting dependency of setup_app_metering so only the
    NSS-install wiring is exercised. ``nss`` replaces install_ca_into_nss.

    The launcher wrap is stubbed to SUCCEED (return True): on a launcher-based
    profile a wrap failure sets metering_active False and short-circuits before
    the NSS step, so these NSS-focused tests need the wrap to pass first."""
    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "launch_proxy", lambda **k: (_FakePopen(["proxy"]), None))
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: spki)
    monkeypatch.setattr(am, "install_launcher_wrapper", lambda **k: True)
    monkeypatch.setattr(am, "start_sdk_watcher", lambda **k: object())
    if nss is not None:
        monkeypatch.setattr(am, "install_ca_into_nss", nss)


def test_setup_calls_nss_install_and_records_on_handle(tmp_path, monkeypatch):
    # setup must install the CA into NSS after the CA exists, forwarding
    # ca_path/home/user, and record home/user on the handle so teardown cleans up.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")  # CA present

    seen = {}

    def _nss(ca_path, home, user=None):
        seen["args"] = (ca_path, home, user)
        return True

    _stub_setup_deps(monkeypatch, nss=_nss)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop",
    )
    assert seen["args"] == (
        os.path.join(str(home), ".clearml_snug", "snug_proxy_ca.pem"),
        str(home),
        "desktop",
    )
    assert handle.nss_home == str(home)
    assert handle.nss_user == "desktop"
    assert handle.metering_active is True


def test_setup_nss_install_failure_does_not_abort_metering(tmp_path, monkeypatch):
    # An NSS install that RAISES must be swallowed inside setup: metering still
    # comes up (a handle is returned) and nss_home is left unset so teardown
    # won't try to remove trust that was never installed.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    def _nss_raises(ca_path, home, user=None):
        raise RuntimeError("certutil exploded")

    _stub_setup_deps(monkeypatch, nss=_nss_raises)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop",
    )
    assert handle is not None
    assert handle.nss_home is None


# -- launch_proxy whitelist env + setup wiring -------------------------------

def _capture_proxy_env(monkeypatch):
    """Stub Popen so launch_proxy's env dict is captured (no real proxy spawn)."""
    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakePopen(argv, **kwargs)

    monkeypatch.setattr(am.subprocess, "Popen", _fake_popen)
    return captured


def test_launch_proxy_sets_whitelist_env_when_given(monkeypatch):
    # When whitelist_b64 is passed, launch_proxy exports it as CLEARML_SNUG_WHITELIST.
    captured = _capture_proxy_env(monkeypatch)
    am.launch_proxy(
        proxy_bin="/proxy", cred_b64=None, ca_path="/ca.pem",
        ca_key_path="/ca.key.pem", whitelist_b64="WL_B64",
    )
    assert captured["env"]["CLEARML_SNUG_WHITELIST"] == "WL_B64"


def test_launch_proxy_omits_whitelist_env_when_none(monkeypatch):
    # With whitelist_b64=None the env var is absent, so the proxy falls back to its
    # meter-all default rather than an empty/garbage whitelist.
    captured = _capture_proxy_env(monkeypatch)
    am.launch_proxy(
        proxy_bin="/proxy", cred_b64=None, ca_path="/ca.pem",
        ca_key_path="/ca.key.pem", whitelist_b64=None,
    )
    assert "CLEARML_SNUG_WHITELIST" not in captured["env"]


def test_setup_builds_and_forwards_whitelist_to_proxy(tmp_path, monkeypatch):
    # setup must resolve the whitelist via build_whitelist_env(session,
    # extra_rules=profile.whitelist_contribution) and forward the result into
    # launch_proxy(whitelist_b64=...).
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    seen = {}

    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: None)
    monkeypatch.setattr(am, "install_launcher_wrapper", lambda **k: False)
    monkeypatch.setattr(am, "start_sdk_watcher", lambda **k: object())
    monkeypatch.setattr(am, "install_ca_into_nss", lambda *a, **k: False)

    def _fake_build_wl(session, extra_rules=None):
        seen["session"] = session
        seen["extra_rules"] = extra_rules
        return "WL_B64"

    def _fake_launch(**kwargs):
        seen["whitelist_b64"] = kwargs.get("whitelist_b64")
        return (_FakePopen(["proxy"]), None)

    monkeypatch.setattr(am, "build_whitelist_env", _fake_build_wl)
    monkeypatch.setattr(am, "launch_proxy", _fake_launch)

    sentinel_session = object()
    # launch_attempts=1: read_ca_spki is stubbed to None so no attempt succeeds;
    # one attempt is enough to prove the whitelist is forwarded to launch_proxy.
    am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=sentinel_session, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", launch_attempts=1,
    )
    assert seen["session"] is sentinel_session
    # The profile's own hosts are forwarded as additions.
    assert seen["extra_rules"] == _CLAUDE_PROFILE.whitelist_contribution
    assert seen["whitelist_b64"] == "WL_B64"


def test_setup_whitelist_build_failure_falls_back_to_none(tmp_path, monkeypatch):
    # A build_whitelist_env failure must never abort setup: launch_proxy still runs
    # with whitelist_b64=None (proxy meter-all default).
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    seen = {}

    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: None)
    monkeypatch.setattr(am, "install_launcher_wrapper", lambda **k: False)
    monkeypatch.setattr(am, "start_sdk_watcher", lambda **k: object())
    monkeypatch.setattr(am, "install_ca_into_nss", lambda *a, **k: False)

    def _boom(session, extra_rules=None):
        raise RuntimeError("whitelist build exploded")

    def _fake_launch(**kwargs):
        seen["whitelist_b64"] = kwargs.get("whitelist_b64")
        return (_FakePopen(["proxy"]), None)

    monkeypatch.setattr(am, "build_whitelist_env", _boom)
    monkeypatch.setattr(am, "launch_proxy", _fake_launch)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", launch_attempts=1,
    )
    assert handle is not None
    assert seen["whitelist_b64"] is None


# -- dynamic proxy port allocation --------------------------------------------
# Port 8888 used to be hardcoded end-to-end with no collision handling, unlike
# the app's own ports (clearml_apps' _find_free_port/_allocate_ports for
# nginx/KasmVNC/filebrowser/the relays) -- these cover the SNUG proxy's own
# free-port pick, which mirrors that pattern.

def test_find_free_port_returns_a_bindable_reservation():
    # The returned socket must actually be holding the port: binding a second
    # socket to the same port while the first is open must fail.
    port, reservation = am._find_free_port()
    try:
        assert 0 < port < 65536
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                other.bind(("127.0.0.1", port))
        finally:
            other.close()
    finally:
        reservation.close()


def _stub_setup_for_port_test(monkeypatch, launch_proxy_fn):
    # read_ca_spki + launcher wrap both SUCCEED so the first attempt comes up
    # clean and the test asserts on the port that was picked, not the retry path.
    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "build_whitelist_env", lambda *a, **k: None)
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: "SPKIval")
    monkeypatch.setattr(am, "install_launcher_wrapper", lambda **k: True)
    monkeypatch.setattr(am, "start_sdk_watcher", lambda **k: object())
    monkeypatch.setattr(am, "install_ca_into_nss", lambda *a, **k: False)
    monkeypatch.setattr(am, "launch_proxy", launch_proxy_fn)


def test_setup_app_metering_picks_free_port_when_none(tmp_path, monkeypatch):
    # port=None (the default) must auto-pick a free port via _find_free_port and
    # release the reservation before launch_proxy spawns the real proxy on it --
    # not the old hardcoded 8888.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    reservation_closed = []

    class _FakeReservation(object):
        def close(self):
            reservation_closed.append(True)

    monkeypatch.setattr(am, "_find_free_port", lambda: (54321, _FakeReservation()))

    seen = {}

    def _fake_launch(**kwargs):
        seen["port"] = kwargs.get("port")
        # Must be released BEFORE the proxy is spawned on it, not after.
        assert reservation_closed == [True]
        return (_FakePopen(["proxy"]), None)

    _stub_setup_for_port_test(monkeypatch, _fake_launch)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop",
    )
    assert seen["port"] == 54321
    assert handle.proxy_url == "http://127.0.0.1:54321"
    assert handle.metering_active is True


def test_setup_app_metering_uses_explicit_port_when_given(tmp_path, monkeypatch):
    # An explicit port must be honored as-is, with no free-port pick -- e.g. for
    # a caller that wants a specific port.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    def _must_not_be_called():
        raise AssertionError("_find_free_port must not be called when port is explicit")

    monkeypatch.setattr(am, "_find_free_port", _must_not_be_called)
    _stub_setup_for_port_test(monkeypatch, lambda **k: (_FakePopen(["proxy"]), None))

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", port=9999,
    )
    assert handle.proxy_url == "http://127.0.0.1:9999"


# -- setup_app_metering: end-to-end proxy-liveness gate -----------------------

def test_setup_app_metering_disables_metering_when_proxy_dies_on_bind(tmp_path, monkeypatch):
    # End-to-end: the proxy wrote its SPKI (it got that far) but has already
    # exited by the time setup checks -- e.g. it lost a port-bind race with
    # another session on the same --network=host worker. Nothing must be
    # wrapped/trusted/watched, and the handle must report metering as inactive so
    # the caller doesn't claim metering is active while the app is actually about
    # to run through a dead (or someone else's) proxy.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")
    (home / ".clearml_snug" / "snug_proxy_ca.spki").write_text("orphanedSpki==\n")

    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "build_whitelist_env", lambda *a, **k: None)
    monkeypatch.setattr(am, "launch_proxy", lambda **k: (_StaticPoll(1), None))

    def _must_not_wrap(**k):
        raise AssertionError("launcher must not be wrapped when the proxy is dead")

    def _must_not_trust(*a, **k):
        raise AssertionError("NSS trust must not be installed when the proxy is dead")

    def _must_not_watch(**k):
        raise AssertionError("SDK watcher must not start when the proxy is dead")

    monkeypatch.setattr(am, "install_launcher_wrapper", _must_not_wrap)
    monkeypatch.setattr(am, "install_ca_into_nss", _must_not_trust)
    monkeypatch.setattr(am, "start_sdk_watcher", _must_not_watch)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop",
        spki_timeout=0.5, launch_attempts=1,
    )
    assert handle.metering_active is False
    assert handle.launchers == []
    assert handle.nss_home is None
    assert handle.watcher is None


# -- setup_app_metering: retry with a fresh port per attempt ------------------

def test_setup_app_metering_retries_with_fresh_port_and_self_heals(tmp_path, monkeypatch):
    # A losing racer on attempt 1 (proxy dies -> read_ca_spki None) must NOT sink
    # the session: the loop retries with a FRESH port and, when the next attempt
    # comes up, metering is active. This is the whole point of dynamic-port +
    # retry -- two sessions on one --network=host host both end up metered.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    ports = []
    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "build_whitelist_env", lambda *a, **k: None)
    monkeypatch.setattr(am, "install_launcher_wrapper", lambda **k: True)
    monkeypatch.setattr(am, "start_sdk_watcher", lambda **k: object())
    monkeypatch.setattr(am, "install_ca_into_nss", lambda *a, **k: False)

    launches = {"n": 0}

    def _fake_launch(**kwargs):
        launches["n"] += 1
        ports.append(kwargs.get("port"))
        return (_FakePopen(["proxy"]), None)

    # First read_ca_spki returns None (lost the bind), second returns a value.
    spki_calls = {"n": 0}

    def _fake_read_spki(*a, **k):
        spki_calls["n"] += 1
        return None if spki_calls["n"] == 1 else "SPKIok=="

    monkeypatch.setattr(am, "launch_proxy", _fake_launch)
    monkeypatch.setattr(am, "read_ca_spki", _fake_read_spki)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", launch_attempts=3,
    )
    assert launches["n"] == 2, "should have retried exactly once then succeeded"
    assert len(ports) == 2 and ports[0] != ports[1], "each attempt must draw a FRESH port"
    assert handle.metering_active is True


def test_setup_app_metering_fails_after_exhausting_attempts(tmp_path, monkeypatch):
    # Every attempt loses the bind -> after launch_attempts tries the handle is
    # returned inactive (proxy torn down), so the caller fails the task.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    launches = {"n": 0}
    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "build_whitelist_env", lambda *a, **k: None)
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: None)

    def _fake_launch(**kwargs):
        launches["n"] += 1
        return (_FakePopen(["proxy"]), None)

    monkeypatch.setattr(am, "launch_proxy", _fake_launch)
    monkeypatch.setattr(am, "install_launcher_wrapper",
                        lambda **k: (_ for _ in ()).throw(AssertionError("no wrap on failure")))

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", launch_attempts=3,
    )
    assert launches["n"] == 3, "should try exactly launch_attempts times"
    assert handle.metering_active is False


def test_setup_app_metering_inactive_when_launcher_wrap_fails(tmp_path, monkeypatch):
    # The proxy is live (SPKI confirmed) but the launcher can't be wrapped -> the
    # Electron shell would bypass the proxy entirely, so this session would run
    # UN-metered. That must count as metering NOT active (a live proxy nothing
    # routes through is not metering), so the caller fails the task.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "build_whitelist_env", lambda *a, **k: None)
    monkeypatch.setattr(am, "launch_proxy", lambda **k: (_FakePopen(["proxy"]), None))
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: "SPKIok==")
    monkeypatch.setattr(am, "install_launcher_wrapper", lambda **k: False)  # wrap fails

    def _must_not_trust(*a, **k):
        raise AssertionError("NSS trust must not run when the launcher wasn't wrapped")

    def _must_not_watch(**k):
        raise AssertionError("SDK watcher must not run when the launcher wasn't wrapped")

    monkeypatch.setattr(am, "install_ca_into_nss", _must_not_trust)
    monkeypatch.setattr(am, "start_sdk_watcher", _must_not_watch)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", launch_attempts=1,
    )
    assert handle.metering_active is False
    assert handle.launchers == []  # nothing recorded as wrapped
    assert handle.nss_home is None


def test_setup_restores_stale_launcher_before_rewrap(tmp_path, monkeypatch):
    # Per launcher, setup must call uninstall_launcher_wrapper BEFORE
    # install_launcher_wrapper, so a stale wrapper from a prior run (pointing at a
    # dead proxy port) is restored and then re-pointed at THIS proxy -- never read
    # as a wrap failure that would fail the task forever.
    home = tmp_path / "home"
    (home / ".clearml_snug").mkdir(parents=True)
    (home / ".clearml_snug" / "snug_proxy_ca.pem").write_bytes(b"ca")

    calls = []
    monkeypatch.setattr(am, "build_shim_descriptor_b64", lambda **k: "cred")
    monkeypatch.setattr(am, "build_whitelist_env", lambda *a, **k: None)
    monkeypatch.setattr(am, "launch_proxy", lambda **k: (_FakePopen(["proxy"]), None))
    monkeypatch.setattr(am, "read_ca_spki", lambda *a, **k: "SPKIok==")
    monkeypatch.setattr(am, "install_ca_into_nss", lambda *a, **k: False)
    monkeypatch.setattr(am, "start_sdk_watcher", lambda **k: object())
    monkeypatch.setattr(am, "uninstall_launcher_wrapper",
                        lambda *a, **k: calls.append("uninstall") or True)
    monkeypatch.setattr(am, "install_launcher_wrapper",
                        lambda **k: calls.append("install") or True)

    handle = am.setup_app_metering(
        profile=_CLAUDE_PROFILE,
        session=None, task_id="t", project="p", home=str(home),
        proxy_bin="/proxy", config=_FakeConfig(), user="desktop", launch_attempts=1,
    )
    assert handle.metering_active is True
    # For each launcher: uninstall (restore any stale wrapper) then install.
    n = len(_CLAUDE_PROFILE.launchers)
    assert calls == ["uninstall", "install"] * n


def test_app_mode_requested_distinguishes_off_from_unknown():
    # app_mode_requested returns the raw configured name; resolve_app_profile
    # returns the profile-or-None. The worker uses the pair to fail CLOSED on a
    # set-but-unknown name (a typo) instead of silently running un-metered.
    off = _FakeConfig({})
    assert am.app_mode_requested(off) == ""
    assert am.resolve_app_profile(off) is None

    known = _FakeConfig({"agent.snug.app_mode": "claude_desktop"})
    assert am.app_mode_requested(known) == "claude_desktop"
    assert am.resolve_app_profile(known) is not None

    # Hyphen typo of the real underscore id: REQUESTED (non-empty) but does NOT
    # resolve -> the worker must treat this as a hard failure, not "off".
    typo = _FakeConfig({"agent.snug.app_mode": "claude-desktop"})
    assert am.app_mode_requested(typo) == "claude-desktop"
    assert am.resolve_app_profile(typo) is None


# ===========================================================================
# -- genericity -------------------------------------------------------------
# The tests above are the Claude behavior ported 1:1 to the generic API. The
# tests below prove the mechanism is DATA-DRIVEN (parameterized by profile /
# SdkBinary / Launcher), not hardcoded to Claude, so a second app is onboarded
# by adding a profile rather than editing the mechanism.
# ===========================================================================


def test_resolve_app_profile_variants():
    # (a) app_mode names the profile to enable.
    p = am.resolve_app_profile(_FakeConfig({"agent.snug.app_mode": "claude_desktop"}))
    assert p is not None and p.app_id == "claude_desktop"
    # (a2) opencode is a second registered profile.
    p2 = am.resolve_app_profile(_FakeConfig({"agent.snug.app_mode": "opencode"}))
    assert p2 is not None and p2.app_id == "opencode"
    # (b) an app_mode naming a profile that isn't registered -> None (not an error).
    assert am.resolve_app_profile(_FakeConfig({"agent.snug.app_mode": "cursor"})) is None
    # (c) empty / unset -> None so a plain agent is unaffected.
    assert am.resolve_app_profile(_FakeConfig({"agent.snug.app_mode": ""})) is None
    assert am.resolve_app_profile(_FakeConfig()) is None
    # (d) the old claude_desktop:true boolean is NOT a selector — app_mode is the
    # only gate (the alias was removed on this unreleased branch).
    assert am.resolve_app_profile(_FakeConfig({"agent.snug.claude_desktop": True})) is None


def test_builtin_claude_desktop_profile_golden():
    # GOLDEN VALUE: lock the exact Claude profile so a future refactor of the
    # generic mechanism can't silently drift Claude's behavior.
    expected = am.AppProfile(
        app_id="claude_desktop",
        launchers=(am.Launcher("/usr/bin/claude-desktop-unofficial", "electron_chromium"),),
        sdk_binaries=(
            am.SdkBinary(
                binary_name="claude",
                discovery="home_glob",
                container_substr="claude-code",
                expects_version_parent=True,
                watched=True,
                wrapper_kind="node_bun",
            ),
        ),
        default_tokenizer="claude",
        external_oauth_browser=True,
        decrypt_all=True,
        h2_assumed_host="api.anthropic.com",
        whitelist_contribution=(
            {
                "host": "claude.ai",
                "path_prefix": "/",
                "inject_headers": False,
                "tokenizer": "claude",
                "estimate_unmeasured": True,
                "completion_path": "*/completion",
                "provider": "anthropic",
            },
        ),
    )
    assert am.BUILTIN_PROFILES["claude_desktop"] == expected

    # Spell out the load-bearing pieces too, so a failure points at the field.
    prof = am.BUILTIN_PROFILES["claude_desktop"]
    assert prof.app_id == "claude_desktop"
    assert prof.launchers == (am.Launcher("/usr/bin/claude-desktop-unofficial", "electron_chromium"),)
    (sdk,) = prof.sdk_binaries
    assert sdk.binary_name == "claude"
    assert sdk.discovery == "home_glob"
    assert sdk.container_substr == "claude-code"
    assert sdk.expects_version_parent is True
    assert sdk.watched is True
    assert sdk.wrapper_kind == "node_bun"
    assert prof.default_tokenizer == "claude"
    assert prof.external_oauth_browser is True
    assert prof.decrypt_all is True
    assert prof.h2_assumed_host == "api.anthropic.com"
    (rule,) = prof.whitelist_contribution
    assert rule["host"] == "claude.ai"
    assert rule["tokenizer"] == "claude"
    assert rule["estimate_unmeasured"] is True
    assert rule["completion_path"] == "*/completion"
    assert rule["provider"] == "anthropic"


def test_synthetic_second_profile_is_generic(tmp_path):
    # Build a fake, non-Claude profile and prove the mechanism is parameterized by
    # its data rather than Claude literals.
    probe = am.AppProfile(
        app_id="probe",
        launchers=(),
        sdk_binaries=(am.SdkBinary("probecli", discovery="path_lookup"),),
        default_tokenizer="approx",
        external_oauth_browser=False,
        decrypt_all=True,
        h2_assumed_host=None,
        whitelist_contribution=(),
    )

    # The launcher marker is derived from the app_id.
    assert am._launcher_marker(probe.app_id) == "SNUG probe launcher wrapper"

    # The SDK wrapper execs <binary>.real for whatever binary the profile names.
    (probe_sdk,) = probe.sdk_binaries
    w = am.render_sdk_wrapper("http://127.0.0.1:8888", "ca.pem", probe_sdk.binary_name)
    assert 'exec "$DIR/probecli.real" "$@"' in w

    # No h2 assumed-host in this profile -> the export is absent from the wrapper.
    lw = am.render_launcher_wrapper(
        "/l.real", "http://127.0.0.1:8888", "/ca.pem", "S",
        am._launcher_marker(probe.app_id), h2_assumed_host=probe.h2_assumed_host,
    )
    assert "CLEARML_SNUG_H2_ASSUMED_HOST" not in lw

    # A discovery mode the module hasn't implemented returns [] (no crash), so an
    # app declaring path_lookup is simply not walked yet.
    assert am.find_sdk_dirs(str(tmp_path), probe_sdk) == []


def test_render_wrappers_reject_unsupported_kinds():
    # Unsupported injection strategies error loudly rather than silently emitting a
    # wrapper that wouldn't route/trust correctly.
    with pytest.raises(ValueError):
        am.render_sdk_wrapper("http://127.0.0.1:8888", "ca.pem", "x", wrapper_kind="generic_ssl")
    with pytest.raises(ValueError):
        am.render_launcher_wrapper(
            "/l.real", "http://127.0.0.1:8888", "/ca.pem", "S", _CLAUDE_MARKER,
            kind="something",
        )


def _decode_whitelist(b64):
    return json.loads(base64.b64decode(b64).decode("utf-8"))


def test_whitelist_union_adds_profile_hosts():
    # A profile's whitelist_contribution is the app-shipped BASE: build_whitelist_env
    # merges the config whitelist as an EXTENSION on top, so the profile rule WINS
    # on a host collision (it carries the app's functional per-host config), while
    # the config still owns default_action and contributes its non-colliding hosts.
    base_only_anthropic = {
        "version": 1,
        "default_action": "meter",
        "rules": [{"host": "api.anthropic.com", "path_prefix": "/"}],
    }

    # With the profile's rows, claude.ai is added.
    session = _FakeSession(base_only_anthropic)
    with_profile = _decode_whitelist(
        wl.build_whitelist_env(session, extra_rules=_CLAUDE_PROFILE.whitelist_contribution)
    )
    hosts_with = [r["host"] for r in with_profile["rules"]]
    assert "claude.ai" in hosts_with
    assert "api.anthropic.com" in hosts_with

    # Without them, claude.ai is absent (proves it comes from the profile, not base).
    session = _FakeSession(base_only_anthropic)
    without_profile = _decode_whitelist(wl.build_whitelist_env(session))
    hosts_without = [r["host"] for r in without_profile["rules"]]
    assert "claude.ai" not in hosts_without

    # If the config ALREADY defines a (stale) claude.ai rule, the PROFILE wins the
    # collision: exactly one claude.ai rule, and it is the profile's (carrying the
    # estimate predicates), NOT the config's — a stale user rule can't disable the
    # app's metering. The config's non-colliding host and default_action survive.
    config_with_stale_claude = {
        "version": 1,
        "default_action": "ignore",
        "rules": [
            {"host": "claude.ai", "tokenizer": "claude"},  # stale: no estimate fields
            {"host": "api.openai.com", "path_prefix": "/"},
        ],
    }
    session = _FakeSession(config_with_stale_claude)
    merged = _decode_whitelist(
        wl.build_whitelist_env(session, extra_rules=_CLAUDE_PROFILE.whitelist_contribution)
    )
    claude_rules = [r for r in merged["rules"] if r["host"] == "claude.ai"]
    assert len(claude_rules) == 1
    assert claude_rules[0].get("estimate_unmeasured") is True, "profile rule wins the collision"
    assert claude_rules[0].get("completion_path")
    assert claude_rules[0].get("provider") == "anthropic"
    # config's non-colliding host + its default_action are preserved.
    assert "api.openai.com" in [r["host"] for r in merged["rules"]]
    assert merged["default_action"] == "ignore"


# ===========================================================================
# -- OpenCode: explicit_path discovery + the second (launcher-less) profile --
# OpenCode is a bun/BoringSSL binary the LD_PRELOAD shim can't hook, so it is
# metered via the app-mode forward proxy like Claude Desktop — but with no
# Electron launcher and a fixed-path SDK on PATH (explicit_path discovery).
# ===========================================================================

_OPENCODE_PROFILE = am.BUILTIN_PROFILES["opencode"]
_OPENCODE_SDK = _OPENCODE_PROFILE.sdk_binaries[0]


def _explicit_sdk(dirs, binary_name="opencode"):
    return am.SdkBinary(
        binary_name=binary_name,
        discovery="explicit_path",
        explicit_dirs=tuple(dirs),
        watched=True,
        wrapper_kind="node_bun",
    )


def test_find_sdk_dirs_explicit_path_matches_elf(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "opencode").write_bytes(_FAKE_ELF)
    assert am.find_sdk_dirs(str(tmp_path), _explicit_sdk([str(d)])) == [str(d)]


def test_find_sdk_dirs_explicit_path_follows_symlink(tmp_path):
    # The real binary lives elsewhere; the on-PATH entry is a symlink to it.
    # find_sdk_dirs must return the SYMLINK's dir (that's where the wrapper +
    # `.real`/CA go and where $(dirname "$0") lands) — isfile/_is_elf follow it.
    real = tmp_path / "opt" / "opencode"
    real.parent.mkdir(parents=True)
    real.write_bytes(_FAKE_ELF)
    usrbin = tmp_path / "usr-local-bin"
    usrbin.mkdir()
    os.symlink(str(real), str(usrbin / "opencode"))
    assert am.find_sdk_dirs(str(tmp_path), _explicit_sdk([str(usrbin)])) == [str(usrbin)]


def test_find_sdk_dirs_explicit_path_skips_already_wrapped(tmp_path):
    # A dir whose `opencode` is our #!/bin/sh wrapper (not an ELF) is skipped ->
    # idempotent, no re-wrap.
    d = tmp_path / "bin"
    d.mkdir()
    (d / "opencode").write_text("#!/bin/sh\nexec ./opencode.real\n")
    assert am.find_sdk_dirs(str(tmp_path), _explicit_sdk([str(d)])) == []


def test_find_sdk_dirs_explicit_path_missing_dir(tmp_path):
    assert am.find_sdk_dirs(str(tmp_path), _explicit_sdk([str(tmp_path / "nope")])) == []


def test_find_sdk_dirs_explicit_path_ignores_home_arg(tmp_path):
    # `home` is irrelevant for explicit_path: a bogus home still finds the
    # absolute dir (the watcher passes each home root; only explicit_dirs matter).
    d = tmp_path / "bin"
    d.mkdir()
    (d / "opencode").write_bytes(_FAKE_ELF)
    assert am.find_sdk_dirs("/no/such/home", _explicit_sdk([str(d)])) == [str(d)]


def test_install_sdk_wrapper_on_symlinked_binary(tmp_path):
    # opencode shape: /usr/local/bin/opencode is a symlink to the real ELF.
    # Wrapping renames the SYMLINK to opencode.real (still -> the ELF) and drops
    # the wrapper + CA in the on-PATH dir so $(dirname "$0") resolves them.
    real = tmp_path / "opt" / "opencode"
    real.parent.mkdir(parents=True)
    real.write_bytes(_FAKE_ELF)
    os.chmod(str(real), 0o755)
    usrbin = tmp_path / "usr-local-bin"
    usrbin.mkdir()
    os.symlink(str(real), str(usrbin / "opencode"))
    ca_src = _write_ca(tmp_path)
    sdk = _explicit_sdk([str(usrbin)])

    assert am.install_sdk_wrapper(str(usrbin), ca_src, "http://127.0.0.1:8888", sdk) is True
    body = (usrbin / "opencode").read_text()
    assert body.startswith("#!/bin/sh\n")
    assert 'exec "$DIR/opencode.real" "$@"' in body
    # The preserved `.real` is the renamed symlink and still points at the ELF.
    assert os.path.islink(str(usrbin / "opencode.real"))
    assert os.path.realpath(str(usrbin / "opencode.real")) == str(real)
    assert (usrbin / "snug_ca.pem").exists()
    # Idempotent: the name is now our wrapper (not an ELF) -> second call no-ops.
    assert am.install_sdk_wrapper(str(usrbin), ca_src, "http://127.0.0.1:8888", sdk) is False


def test_builtin_opencode_profile_golden():
    # GOLDEN VALUE: lock the OpenCode profile so a refactor can't silently drift
    # the launcher-less / fixed-path / decrypt-only-providers shape.
    prof = am.BUILTIN_PROFILES["opencode"]
    assert prof.app_id == "opencode"
    assert prof.launchers == ()          # no Electron/Chromium leg
    (sdk,) = prof.sdk_binaries
    assert sdk.binary_name == "opencode"
    assert sdk.discovery == "explicit_path"
    assert sdk.explicit_dirs == ("/usr/local/bin",)
    assert sdk.watched is True           # setup wraps SDKs only via the watcher
    assert sdk.wrapper_kind == "node_bun"
    assert prof.default_tokenizer == "approx"
    assert prof.external_oauth_browser is False   # BYO API key, no browser OAuth
    assert prof.decrypt_all is False              # decrypt only known providers
    assert prof.h2_assumed_host is None
    assert prof.whitelist_contribution == ()      # 3 providers are known hosts


def test_opencode_profile_starts_watcher_not_launcher():
    # A launcher-less profile with a WATCHED SDK: setup must start the SDK
    # watcher (the only thing that wraps SDKs) and not gate on any launcher.
    prof = am.BUILTIN_PROFILES["opencode"]
    assert prof.launchers == ()
    assert any(getattr(s, "watched", False) for s in prof.sdk_binaries)
