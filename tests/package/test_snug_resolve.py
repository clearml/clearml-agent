"""Tests for clearml_agent.helper.snug.

Locks in that resolve_shim_path(), injection_env_var(), snug_enabled(), and
macos_dyld_injection_supported() behave consistently across Linux + macOS -
regardless of whether a shim happens to be built into this checkout.
"""
import os
import platform

import pytest

from clearml_agent.helper import snug


# --- resolve_shim_path() ---------------------------------------------------
def test_resolve_shim_path_returns_none_on_unsupported_os(monkeypatch):
    """Windows (and anything not in _SUPPORTED_OS_EXT) -> None: no preload
    mechanism we support, so SNUG is a no-op there."""
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert snug.resolve_shim_path() is None


def test_resolve_shim_path_returns_none_on_unsupported_arch(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    assert snug.resolve_shim_path() is None


def test_resolve_shim_path_returns_none_on_macos_unsupported_arch(monkeypatch):
    # macOS IS supported, but only on x86_64 / arm64. A bogus arch -> None.
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    assert snug.resolve_shim_path() is None


def test_resolve_shim_path_honors_env_override(tmp_path, monkeypatch):
    """When CLEARML_SNUG_SHIM_PATH points at an existing file, that path is
    returned even on platforms / arches where the in-wheel lookup would
    otherwise return None. This is how the outer agent in --docker mode points
    the in-container agent at the mounted .so."""
    fake_so = tmp_path / "libclearml_snug.so"
    fake_so.write_bytes(b"\x7fELF\x02\x01\x01")  # ELF magic so it looks real
    monkeypatch.setenv("CLEARML_SNUG_SHIM_PATH", str(fake_so))
    # Pretend we're on an UNSUPPORTED platform to prove the override bypasses
    # the OS/arch gate entirely.
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert snug.resolve_shim_path() == str(fake_so)


def test_resolve_shim_path_falls_back_when_override_file_missing(tmp_path, monkeypatch):
    """A misconfigured CLEARML_SNUG_SHIM_PATH (set, but pointing at a nonexistent
    file) falls through to the normal pkg_resources lookup so the agent doesn't
    silently break. We pin an UNSUPPORTED OS so the fallback is deterministically
    None regardless of whether a shim is built into this checkout."""
    monkeypatch.setenv("CLEARML_SNUG_SHIM_PATH", str(tmp_path / "does-not-exist.so"))
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert snug.resolve_shim_path() is None


def test_resolve_shim_path_ignores_empty_override(monkeypatch):
    """Empty / whitespace CLEARML_SNUG_SHIM_PATH falls through to the normal
    lookup (pinned to an unsupported OS -> None)."""
    monkeypatch.setenv("CLEARML_SNUG_SHIM_PATH", "   ")
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert snug.resolve_shim_path() is None


def _patch_resource_filename(monkeypatch, tmp_path, create=True):
    """Patch pkg_resources.resource_filename to record the requested relative
    path and return a real file under tmp_path (so os.path.isfile passes).
    Returns a one-element list that will hold the captured relative path."""
    captured = []

    def _fake(pkg, rel):
        captured.append(rel)
        dest = tmp_path / rel
        if create:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"")
        return str(dest)

    monkeypatch.setattr("pkg_resources.resource_filename", _fake)
    return captured


def test_resolve_shim_path_picks_dylib_ext_on_macos(tmp_path, monkeypatch):
    """On macOS the resolver looks for the .dylib (DYLD_INSERT_LIBRARIES), in the
    same per-arch dir as the Linux .so."""
    monkeypatch.delenv("CLEARML_SNUG_SHIM_PATH", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    captured = _patch_resource_filename(monkeypatch, tmp_path)
    path = snug.resolve_shim_path()
    assert path is not None
    assert captured == ["snug/lib/aarch64/libclearml_snug.dylib"]
    assert path.endswith("/snug/lib/aarch64/libclearml_snug.dylib")


def test_resolve_shim_path_picks_so_ext_on_linux(tmp_path, monkeypatch):
    """On Linux the resolver looks for the .so (LD_PRELOAD)."""
    monkeypatch.delenv("CLEARML_SNUG_SHIM_PATH", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    captured = _patch_resource_filename(monkeypatch, tmp_path)
    path = snug.resolve_shim_path()
    assert path is not None
    assert captured == ["snug/lib/x86_64/libclearml_snug.so"]


def test_resolve_shim_path_force_linux_so_on_macos_host(tmp_path, monkeypatch):
    """--docker mode on a macOS agent: force_system='Linux' resolves the LINUX
    .so for the container (NOT the host .dylib), keyed by the host CPU arch."""
    monkeypatch.delenv("CLEARML_SNUG_SHIM_PATH", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")   # macOS agent host
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    captured = _patch_resource_filename(monkeypatch, tmp_path)
    path = snug.resolve_shim_path(force_system="Linux")
    assert path is not None
    assert captured == ["snug/lib/aarch64/libclearml_snug.so"]
    assert path.endswith("/snug/lib/aarch64/libclearml_snug.so")


def test_resolve_shim_path_force_arch_override(tmp_path, monkeypatch):
    """force_arch overrides the host CPU arch — e.g. resolving the x86_64 Linux
    .so for an amd64 container on an arm64 Mac (a forced --platform image)."""
    monkeypatch.delenv("CLEARML_SNUG_SHIM_PATH", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    captured = _patch_resource_filename(monkeypatch, tmp_path)
    path = snug.resolve_shim_path(force_system="Linux", force_arch="x86_64")
    assert path is not None
    assert captured == ["snug/lib/x86_64/libclearml_snug.so"]


def test_resolve_shim_path_consistent_with_filesystem():
    """The resolver returns a path iff the shim is actually on disk.

    On a fresh checkout with no Rust build, this returns None. On a CI-built
    install where the wheel shipped the shim, this returns an absolute path that
    os.path.isfile(...) verifies. Either way the invariant holds: None -> nothing
    at the expected path; non-None -> real file there.
    """
    result = snug.resolve_shim_path()
    if result is None:
        # Either we're on an unsupported OS/arch host, OR the shim wasn't built.
        # Both are valid states.
        return
    assert os.path.isfile(result), "resolver returned {!r} but no file there".format(result)


# --- injection_env_var() ---------------------------------------------------
def test_injection_env_var_macos(monkeypatch):
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: True)
    assert snug.injection_env_var() == "DYLD_INSERT_LIBRARIES"


def test_injection_env_var_linux(monkeypatch):
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: False)
    assert snug.injection_env_var() == "LD_PRELOAD"


# --- snug_enabled() --------------------------------------------------------
class _FakeConfig(object):
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, key, default=None):
        return self._mapping.get(key, default)


class _FakeSession(object):
    def __init__(self, mapping):
        self.config = _FakeConfig(mapping)


def test_snug_enabled_false_when_config_false():
    """The operator hasn't opted in."""
    session = _FakeSession({"agent.snug.enabled": False})
    assert snug.snug_enabled(session) is False


def test_snug_enabled_false_when_resolver_returns_none(monkeypatch):
    """No shim on disk -> can't load anything, so the flag effectively flips
    itself off. Matches what happens on Windows / unsupported arches."""
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: None)
    session = _FakeSession({"agent.snug.enabled": True})
    assert snug.snug_enabled(session) is False


def test_snug_enabled_true_when_all_conditions_met_linux(monkeypatch, tmp_path):
    """Operator opted in, we're on Linux, and a real shim exists. Then yes."""
    fake_so = tmp_path / "libclearml_snug.so"
    fake_so.write_bytes(b"")
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: str(fake_so))
    monkeypatch.setattr("clearml_agent.helper.base.is_linux_platform", lambda: True)
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: False)
    session = _FakeSession({"agent.snug.enabled": True})
    assert snug.snug_enabled(session) is True


def test_snug_enabled_true_on_macos(monkeypatch, tmp_path):
    """macOS is now a supported platform: opted in + a real .dylib -> enabled."""
    fake_dylib = tmp_path / "libclearml_snug.dylib"
    fake_dylib.write_bytes(b"")
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: str(fake_dylib))
    monkeypatch.setattr("clearml_agent.helper.base.is_linux_platform", lambda: False)
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: True)
    session = _FakeSession({"agent.snug.enabled": True})
    assert snug.snug_enabled(session) is True


def test_snug_enabled_false_on_unsupported_os(monkeypatch, tmp_path):
    """Even with config on and a shim on disk, an unsupported OS (neither Linux
    nor macOS, e.g. Windows) always disables."""
    fake_so = tmp_path / "libclearml_snug.so"
    fake_so.write_bytes(b"")
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: str(fake_so))
    monkeypatch.setattr("clearml_agent.helper.base.is_linux_platform", lambda: False)
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: False)
    session = _FakeSession({"agent.snug.enabled": True})
    assert snug.snug_enabled(session) is False


def test_snug_enabled_false_when_session_get_raises():
    """Defensive: a broken Session must not crash the executioner."""
    class Boom(object):
        @property
        def config(self):
            raise RuntimeError("boom")
    assert snug.snug_enabled(Boom()) is False


# --- macos_dyld_injection_supported() --------------------------------------
def test_macos_dyld_injection_supported_true_on_non_macos(monkeypatch):
    """On non-macOS the SIP probe is irrelevant (LD_PRELOAD is used), so it
    short-circuits True without spawning any subprocess."""
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: False)
    # If this tried to spawn, it would need a real interpreter; the short-circuit
    # means the bogus path is never executed.
    assert snug.macos_dyld_injection_supported("/nonexistent/python") is True


def test_macos_dyld_injection_supported_false_when_no_shim(monkeypatch):
    """On macOS with no built shim, there is nothing to inject -> False (and no
    probe spawn)."""
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: True)
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: None)
    snug._DYLD_SUPPORT_CACHE.clear()
    assert snug.macos_dyld_injection_supported("/usr/bin/python3") is False


def test_macos_dyld_injection_definitive_result_is_cached(monkeypatch):
    """A completed probe is a definitive verdict (the interpreter's SIP/hardened
    status is stable), so it's cached and probed only once."""
    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: True)
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: "/fake/shim.dylib")
    snug._DYLD_SUPPORT_CACHE.clear()
    calls = {"n": 0}

    class _Res(object):
        returncode = 0
        stderr = b"[snug] init pid=1 call_history=off rules=0 reporter=stderr\n"

    def _ok(*a, **k):
        calls["n"] += 1
        return _Res()

    monkeypatch.setattr(snug.subprocess, "run", _ok)
    assert snug.macos_dyld_injection_supported("/opt/homebrew/bin/python3") is True
    assert snug.macos_dyld_injection_supported("/opt/homebrew/bin/python3") is True
    assert calls["n"] == 1, "a definitive verdict must be cached (probe once)"


def test_macos_dyld_injection_transient_failure_not_cached(monkeypatch):
    """A transient probe failure (TimeoutExpired / fork OSError) is NOT a real
    injectability verdict and must NOT be cached — otherwise one blip would
    permanently disable SNUG for that interpreter on a long-lived agent. The
    next task must re-probe."""
    import subprocess as _sp

    monkeypatch.setattr("clearml_agent.helper.base.is_macos_platform", lambda: True)
    monkeypatch.setattr(snug, "resolve_shim_path", lambda: "/fake/shim.dylib")
    snug._DYLD_SUPPORT_CACHE.clear()
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise _sp.TimeoutExpired(cmd="probe", timeout=15)

    monkeypatch.setattr(snug.subprocess, "run", _boom)
    assert snug.macos_dyld_injection_supported("/opt/homebrew/bin/python3") is False
    # Not cached -> a second call re-probes (re-invokes subprocess.run).
    assert snug.macos_dyld_injection_supported("/opt/homebrew/bin/python3") is False
    assert calls["n"] == 2, "a transient failure must not be cached (must re-probe)"
    assert "/opt/homebrew/bin/python3" not in [
        k for k in snug._DYLD_SUPPORT_CACHE
    ] and not any("python3" in str(k) for k in snug._DYLD_SUPPORT_CACHE)
