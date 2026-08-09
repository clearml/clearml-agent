"""Tests for the SNUG-related env vars in Worker._get_job_os_envs.

The executioner wires LD_PRELOAD along with CLEARML_SNUG_WHITELIST
(base64 inline whitelist content), the call-history env vars
(CLEARML_SNUG_CALL_HISTORY[_BUFFER|_CAP_BYTES], from HOCON), and
CLEARML_PROJECT_ID (from the task object). All locked here so a future
refactor that drops one of them fails loud.

We construct a minimal Worker instance via ``__new__`` and set only the
attributes _get_job_os_envs touches; full Worker initialization would
pull in the whole agent (Session, Config, etc.).
"""
import pytest

from clearml_agent.commands import worker as worker_mod
from clearml_agent.commands.worker import Worker


class _FakeConfig(object):
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, key, default=None):
        return self._mapping.get(key, default)


class _FakeSession(object):
    def __init__(self, mapping=None, config_file="/tmp/test-config.conf", user_properties=None):
        self.config = _FakeConfig(mapping or {})
        self.config_file = config_file
        self._user_properties = dict(user_properties or {})

    def get(self, service, action, tasks=None, **kw):
        # Mirrors tasks.get_hyper_params: the "properties" section as name/value
        # entries. The per-task predefine read in _get_job_os_envs goes through
        # here; empty by default → get_task_user_property returns None.
        hyperparams = [
            {"section": "properties", "name": n, "value": v}
            for n, v in self._user_properties.items()
        ]
        return {"params": [{"hyperparams": hyperparams}]}


class _FakeTask(object):
    def __init__(self, task_id="task-abc123", project=""):
        self.id = task_id
        self.project = project


def _stub_worker(session=None):
    """A Worker instance we can call _get_job_os_envs on without
    running __init__."""
    w = Worker.__new__(Worker)
    w._session = session or _FakeSession()
    return w


# --- LD_PRELOAD env tests --------------------------------------------


def test_no_ld_preload_when_snug_disabled(monkeypatch):
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: False)
    monkeypatch.setenv("LD_PRELOAD", "/some/other.so")

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert "LD_PRELOAD" not in envs


def test_ld_preload_set_when_snug_enabled(monkeypatch):
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda: "/opt/clearml/libclearml_snug.so"
    )
    # Pin the preload var to LD_PRELOAD so this test is deterministic on a macOS
    # dev host (where injection_env_var() would otherwise return
    # DYLD_INSERT_LIBRARIES); the prepend/preserve logic is identical for both.
    monkeypatch.setattr(worker_mod, "injection_env_var", lambda: "LD_PRELOAD")
    # build_whitelist_env must not be the real one - we're testing env
    # plumbing here, not config resolution. Monkeypatching the source module
    # so the function-local import inside _get_job_os_envs picks up the stub.
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("LD_PRELOAD") == "/opt/clearml/libclearml_snug.so"


def test_ld_preload_preserves_existing(monkeypatch):
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda: "/opt/clearml/libclearml_snug.so"
    )
    monkeypatch.setattr(worker_mod, "injection_env_var", lambda: "LD_PRELOAD")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    monkeypatch.setenv("LD_PRELOAD", "/opt/other.so")

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("LD_PRELOAD") == "/opt/clearml/libclearml_snug.so:/opt/other.so"


def test_ld_preload_handles_empty_existing(monkeypatch):
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda: "/opt/clearml/libclearml_snug.so"
    )
    monkeypatch.setattr(worker_mod, "injection_env_var", lambda: "LD_PRELOAD")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    monkeypatch.setenv("LD_PRELOAD", "   ")

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("LD_PRELOAD") == "/opt/clearml/libclearml_snug.so"


def test_dyld_insert_libraries_used_on_macos(monkeypatch):
    """On macOS the preload var is DYLD_INSERT_LIBRARIES (not LD_PRELOAD).
    injection_env_var() picks it; the prepend logic is otherwise identical."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda: "/opt/clearml/libclearml_snug.dylib"
    )
    monkeypatch.setattr(worker_mod, "injection_env_var", lambda: "DYLD_INSERT_LIBRARIES")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    monkeypatch.delenv("DYLD_INSERT_LIBRARIES", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("DYLD_INSERT_LIBRARIES") == "/opt/clearml/libclearml_snug.dylib"
    assert "LD_PRELOAD" not in envs


def test_snug_active_false_forces_no_preload(monkeypatch):
    """The macOS SIP preflight passes snug_active=False to force-disable SNUG
    for a launch even though the worker-level snug_enabled() is True. No preload
    var (and none of the SNUG-config env vars) must be emitted."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.dylib")
    monkeypatch.setattr(worker_mod, "injection_env_var", lambda: "DYLD_INSERT_LIBRARIES")

    def _fail_build(*a, **k):
        raise AssertionError("build_whitelist_env must not run when snug_active is False")

    monkeypatch.setattr("clearml_agent.snug.whitelist.build_whitelist_env", _fail_build)

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO", snug_active=False)
    assert "DYLD_INSERT_LIBRARIES" not in envs
    assert "LD_PRELOAD" not in envs
    assert "CLEARML_SNUG_WHITELIST" not in envs


def test_no_ld_preload_when_resolver_returns_none(monkeypatch):
    """Edge: snug_enabled True but resolver returned None mid-stream.
    Must not emit a truthy LD_PRELOAD pointing at nothing."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: None)

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert "LD_PRELOAD" not in envs
    # And none of the SNUG-config env vars should leak either.
    assert "CLEARML_SNUG_WHITELIST" not in envs


# --- SNUG-config env vars ----------------------------------------


def test_snug_config_env_vars_set_when_snug_enabled(monkeypatch):
    """When snug_enabled, _get_job_os_envs sets CLEARML_SNUG_WHITELIST
    (base64 content), the call-history env vars (from HOCON), and
    CLEARML_PROJECT_ID (from task)."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda: "/opt/x.so"
    )
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "d2hpdGVsaXN0LWI2NA==",
    )

    session = _FakeSession(
        {
            "agent.snug.call_history": "continuous",
            "agent.snug.call_history_buffer": 20,
        }
    )
    w = _stub_worker(session)
    task = _FakeTask(task_id="task-abc", project="project-xyz")
    envs = w._get_job_os_envs(task, "INFO")

    assert envs.get("CLEARML_SNUG_WHITELIST") == "d2hpdGVsaXN0LWI2NA=="
    assert envs.get("CLEARML_SNUG_CALL_HISTORY") == "continuous"
    assert envs.get("CLEARML_SNUG_CALL_HISTORY_BUFFER") == "20"
    assert envs.get("CLEARML_PROJECT_ID") == "project-xyz"


def test_call_history_defaults(monkeypatch):
    """Missing call-history keys -> CLEARML_SNUG_CALL_HISTORY='off' plus the
    shipped buffer/cap defaults (the agent.conf snug block defaults)."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )

    # No call-history keys in the config -> defaults kick in.
    w = _stub_worker(_FakeSession({}))
    envs = w._get_job_os_envs(_FakeTask(project="proj"), "INFO")
    assert envs.get("CLEARML_SNUG_CALL_HISTORY") == "off"
    assert envs.get("CLEARML_SNUG_CALL_HISTORY_BUFFER") == "50"
    assert envs.get("CLEARML_SNUG_CALL_HISTORY_CAP_BYTES") == "262144"


def test_empty_project_id_passed_through(monkeypatch):
    """A task with no .project attribute or with project='' -> the env var
    is set to '' and the shim omits the project: header."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )

    task = _FakeTask(project="")
    w = _stub_worker()
    envs = w._get_job_os_envs(task, "INFO")
    assert envs.get("CLEARML_PROJECT_ID") == ""


def test_no_snug_config_env_vars_when_snug_disabled(monkeypatch):
    """The default path: when SNUG is off, none of the SNUG-config env vars
    appear and build_whitelist_env is never called."""
    called = {"build_whitelist_env": 0}

    def _fail_build(*a, **k):
        called["build_whitelist_env"] += 1
        raise AssertionError("build_whitelist_env must not run when SNUG is off")

    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: False)
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env", _fail_build
    )

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    for key in (
        "CLEARML_SNUG_WHITELIST",
        "CLEARML_SNUG_CALL_HISTORY",
        "CLEARML_SNUG_CALL_HISTORY_BUFFER",
        "CLEARML_SNUG_CALL_HISTORY_CAP_BYTES",
        "CLEARML_PROJECT_ID",
    ):
        assert key not in envs, "{} unexpectedly present: {}".format(key, envs)
    assert called["build_whitelist_env"] == 0


def test_no_ipc_socket_env_var(monkeypatch):
    """Reporting is in-process (no IPC socket), so _get_job_os_envs never emits
    CLEARML_SNUG_IPC. The credential handoff (CLEARML_SNUG_CRED_FD) is set in
    execute_task at launch time (when the memfd is created), not here."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    w = _stub_worker(_FakeSession({}))
    envs = w._get_job_os_envs(_FakeTask(task_id="task-abc"), "INFO")
    assert "CLEARML_SNUG_IPC" not in envs
    assert "CLEARML_SNUG_CRED_FD" not in envs  # set in execute_task, not here


# --- per-task predefine (CLEARML_SNUG_WHITELIST_ADDITIONS) --------------


def test_predefined_whitelist_property_becomes_additions_env(monkeypatch):
    """A task that already carries a _snug_whitelist User Property (set before
    launch) gets its raw value passed to the shim as
    CLEARML_SNUG_WHITELIST_ADDITIONS, so the additions apply from the FIRST
    request (before the reporter's first poll)."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    session = _FakeSession(
        {}, user_properties={"_snug_whitelist": "api.foo.com, api.bar.com"}
    )
    w = _stub_worker(session)
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("CLEARML_SNUG_WHITELIST_ADDITIONS") == "api.foo.com, api.bar.com"


def test_no_additions_env_when_no_predefined_whitelist(monkeypatch):
    """No (or empty) _snug_whitelist property → no CLEARML_SNUG_WHITELIST_ADDITIONS
    (the common case; additions then arrive only via the live poll)."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    # absent property
    w = _stub_worker(_FakeSession({}))
    assert "CLEARML_SNUG_WHITELIST_ADDITIONS" not in w._get_job_os_envs(_FakeTask(), "INFO")
    # present-but-empty property (operator cleared the field) also yields no env
    w2 = _stub_worker(_FakeSession({}, user_properties={"_snug_whitelist": ""}))
    assert "CLEARML_SNUG_WHITELIST_ADDITIONS" not in w2._get_job_os_envs(_FakeTask(), "INFO")


# --- per-task predefine (_snug_call_history) ----------------------------


def test_predefined_call_history_property_overrides_config(monkeypatch):
    """A _snug_call_history User Property set before launch wins over the agent
    config default, so a mode set in advance applies from the first request."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    # Config says "off", but the task carries a pre-set "continuous" -> the
    # property wins.
    session = _FakeSession(
        {"agent.snug.call_history": "off"},
        user_properties={"_snug_call_history": "continuous"},
    )
    envs = _stub_worker(session)._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("CLEARML_SNUG_CALL_HISTORY") == "continuous"


def test_predefined_call_history_normalized_and_invalid_falls_back(monkeypatch):
    """A pre-set mode is normalized (trim/lowercase); an unrecognized value
    falls back to the agent config default."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    # mixed case + whitespace -> normalized to "collect"
    s1 = _FakeSession(
        {"agent.snug.call_history": "off"},
        user_properties={"_snug_call_history": "  Collect "},
    )
    assert _stub_worker(s1)._get_job_os_envs(_FakeTask(), "INFO").get(
        "CLEARML_SNUG_CALL_HISTORY"
    ) == "collect"
    # bogus value -> ignored, falls back to the config default
    s2 = _FakeSession(
        {"agent.snug.call_history": "continuous"},
        user_properties={"_snug_call_history": "bogus"},
    )
    assert _stub_worker(s2)._get_job_os_envs(_FakeTask(), "INFO").get(
        "CLEARML_SNUG_CALL_HISTORY"
    ) == "continuous"


# --- parse-usage gating env var ----------------------------------


def _snug_enabled_worker(monkeypatch, config):
    """A stub Worker with SNUG enabled, a resolvable .so, and a stubbed
    whitelist renderer - parametrized by the session config mapping."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )
    return _stub_worker(_FakeSession(config))


def test_parse_usage_unset_when_no_sink(monkeypatch):
    """With both sinks off there is nothing to consume parsed usage, so
    CLEARML_SNUG_PARSE_USAGE is NOT set - the shim then does no body parsing
    (the zero-overhead default path)."""
    w = _snug_enabled_worker(monkeypatch, {})
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert "CLEARML_SNUG_PARSE_USAGE" not in envs


def test_parse_usage_set_when_task_metrics_on(monkeypatch):
    """A reporting sink turns usage parsing on; the shim then parses usage for
    the known providers."""
    w = _snug_enabled_worker(monkeypatch, {"agent.snug.report_task_metrics": True})
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("CLEARML_SNUG_PARSE_USAGE") == "1"


def test_parse_usage_set_when_usage_on(monkeypatch):
    w = _snug_enabled_worker(monkeypatch, {"agent.snug.report_usage_events": True})
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("CLEARML_SNUG_PARSE_USAGE") == "1"


def test_debug_log_unset_by_default(monkeypatch):
    """agent.snug.debug_log defaults off, so CLEARML_SNUG_DEBUG_LOG is NOT set
    and the shim logs errors only (plus the single init line where it reports)."""
    w = _snug_enabled_worker(monkeypatch, {})
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert "CLEARML_SNUG_DEBUG_LOG" not in envs


def test_debug_log_set_when_enabled(monkeypatch):
    """agent.snug.debug_log truthy exports CLEARML_SNUG_DEBUG_LOG=1 so the shim
    emits its verbose per-process diagnostics."""
    w = _snug_enabled_worker(monkeypatch, {"agent.snug.debug_log": True})
    envs = w._get_job_os_envs(_FakeTask(), "INFO")
    assert envs.get("CLEARML_SNUG_DEBUG_LOG") == "1"


# --- cred-fd forwarding to the task subprocess (both launch paths) ----------


def test_execute_forwards_cred_fd_on_both_launch_paths():
    """Regression guard for the in-process reporter handoff.

    The task subprocess is launched two ways in execute(): the
    disable_monitoring branch (command.check_call) and the monitoring branch
    (_log_command_output -> subprocess.Popen). BOTH spawn with the POSIX default
    close_fds=True, which drops every inherited fd except 0/1/2 unless it's in
    pass_fds. So BOTH must forward the SNUG credential fd via
    pass_fds=[snug_cred_fd]; otherwise the shim in the task finds
    CLEARML_SNUG_CRED_FD pointing at a closed/reused number and silently falls
    back to reporter=stderr (losing all metering). On macOS use_execv is always
    False, so the non-execv path is the ONLY one a task can take.

    execute() is far too large to unit-drive, so this is a source-level guard:
    the cred-fd pass_fds forwarding must appear on BOTH launch sites.
    """
    import inspect
    src = inspect.getsource(Worker.execute)
    n = src.count('"pass_fds": [snug_cred_fd]')
    assert n >= 2, (
        "expected the SNUG cred fd to be forwarded via pass_fds on BOTH the "
        "disable_monitoring (check_call) and the monitoring (_log_command_output) "
        "launch paths; found {} forwarding site(s). The monitoring path "
        "(subprocess.Popen close_fds=True) drops the fd otherwise.".format(n)
    )


# --- docker-mode args injector --------------------------------


def test_get_snug_docker_args_empty_when_snug_disabled(monkeypatch):
    """When SNUG is off, the docker-args injector returns [] - the docker
    invocation proceeds with no SNUG-specific changes."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: False)
    w = _stub_worker()
    args = w._get_snug_docker_args(task_id="some-task")
    assert args == []


def test_get_snug_docker_args_empty_when_resolver_returns_none(monkeypatch):
    """Snug enabled but resolver can't find a .so (missing file, etc.) -> []
    rather than emitting -v with a bogus path."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda *a, **k: None)
    w = _stub_worker()
    args = w._get_snug_docker_args(task_id="some-task")
    assert args == []


def test_get_snug_docker_args_resolves_linux_so_on_macos_host(monkeypatch):
    """Docker-from-macOS-agent: even on a Darwin host, the task container is
    Linux, so _get_snug_docker_args must resolve + mount the LINUX .so (via
    resolve_shim_path(force_system="Linux")), NOT the host .dylib. We capture the
    force args to prove it asks for Linux, and confirm the mount + enable env."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "is_linux_platform", lambda: False)  # pretend a macOS host

    captured = {}

    def _resolve(force_system=None, force_arch=None):
        captured["force_system"] = force_system
        captured["force_arch"] = force_arch
        # Mimic resolving the LINUX .so for the container arch.
        return "/Users/me/repo/clearml_agent/snug/lib/aarch64/libclearml_snug.so"

    monkeypatch.setattr(worker_mod, "resolve_shim_path", _resolve)
    w = _stub_worker()
    args = w._get_snug_docker_args(task_id="some-task")

    assert captured["force_system"] == "Linux", (
        "the container is Linux, so the injector must resolve the Linux .so "
        "(force_system='Linux'), not the host .dylib"
    )
    assert args[0] == "-v"
    mount = args[1]
    assert mount.endswith("libclearml_snug.so:/opt/clearml-snug/libclearml_snug.so:ro")
    assert "CLEARML_AGENT_SNUG_ENABLED=true" in args


def test_get_snug_docker_args_returns_volume_and_env(monkeypatch):
    """Snug enabled + resolver returns a host path -> the injector
    returns -v <host>:<container>:ro plus two -e flags: CLEARML_SNUG_SHIM_PATH
    (so resolve_shim_path() inside the task container picks the mounted .so)
    and CLEARML_AGENT_SNUG_ENABLED=true (so the inner agent's snug_enabled()
    returns True and actually sets LD_PRELOAD)."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda *a, **k: "/path/to/libclearml_snug.so"
    )
    w = _stub_worker()
    args = w._get_snug_docker_args(task_id="some-task")

    # Shape: -v <mount> -e <shim-path-env> -e <enabled-env>. We don't pin
    # the exact container path string - just that host and shim-path env
    # agree, and that the enabled flag is also propagated.
    assert args[0] == "-v"
    assert args[2] == "-e"
    assert args[4] == "-e"
    mount = args[1]
    assert mount.startswith("/path/to/libclearml_snug.so:")
    assert mount.endswith(":ro")
    container_path = mount.split(":")[1]
    assert args[3] == "CLEARML_SNUG_SHIM_PATH={}".format(container_path)
    assert args[5] == "CLEARML_AGENT_SNUG_ENABLED=true"


def test_get_snug_docker_args_no_whitelist_mount_or_env(monkeypatch):
    """The whitelist is inline config now (agent.snug.whitelist) - the
    inner agent re-derives CLEARML_SNUG_WHITELIST from its own config. So
    the docker-args injector emits NO whitelist bind-mount (only the .so
    mount) and NO CLEARML_SNUG_WHITELIST -e flag (that's the inner agent's
    job, via _get_job_os_envs)."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda *a, **k: "/opt/x.so"
    )
    w = _stub_worker()
    args = w._get_snug_docker_args(task_id="some-task")

    # Only the .so bind-mount; no whitelist -v.
    v_count = sum(1 for a in args if a == "-v")
    assert v_count == 1, (
        "expected only the .so bind-mount, got {} -v entries: {}"
    ).format(v_count, args)
    # The whitelist content is NOT injected by the docker-args path.
    assert not any("CLEARML_SNUG_WHITELIST" in str(a) for a in args), (
        "docker-args must not carry the whitelist; the inner agent derives "
        "it from config: {}".format(args)
    )


def test_get_snug_docker_args_propagates_report_usage_events(monkeypatch):
    """When ``agent.snug.report_usage_events`` is true on
    the outer agent, the docker-args injector forwards
    ``CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS=true`` into the spawned
    task container so the inner agent's reporter also enables the
    usage sink.

    Without this propagation, the inner agent would only see
    ``CLEARML_AGENT_SNUG_ENABLED=true`` and the usage flag would
    default to False inside the spawned container - usage POSTs
    would never fire even though the operator opted in on the outer
    config.
    """
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda *a, **k: "/opt/x.so"
    )
    # Make sure the env-var fallback path is NOT what's triggering -
    # we want to confirm the config-tree path works in isolation.
    monkeypatch.delenv("CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS", raising=False)

    session = _FakeSession({"agent.snug.report_usage_events": True})
    w = _stub_worker(session=session)
    args = w._get_snug_docker_args(task_id="some-task")

    # Concretely: an -e flag with CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS=true.
    assert "CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS=true" in args, (
        "expected usage env var to be propagated; got args: {}"
    ).format(args)


def test_get_snug_docker_args_propagates_via_env_var_fallback(monkeypatch):
    """Defense-in-depth: even if the env-config override
    didn't materialize ``agent.snug.report_usage_events`` into the
    in-memory session config tree (e.g. due to a reload-ordering
    issue), reading the env var directly still propagates the flag.

    Mirrors the ``CLEARML_AGENT_SNUG_ENABLED=true`` hardcoded forward
    above, which doesn't depend on the config tree either."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda *a, **k: "/opt/x.so"
    )
    monkeypatch.setenv("CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS", "true")

    # Config doesn't have the key - env var fallback is the only signal.
    w = _stub_worker()
    args = w._get_snug_docker_args(task_id="some-task")

    assert "CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS=true" in args, (
        "env-var fallback should propagate the flag even without config; "
        "got args: {}"
    ).format(args)


def test_get_snug_docker_args_skips_usage_env_when_disabled(monkeypatch):
    """When ``agent.snug.report_usage_events`` is false (the default)
    AND the env var is unset, the injector does NOT forward the usage
    env var into the task container. Keeps the docker-run command
    minimal for the common (usage-disabled) case."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(
        worker_mod, "resolve_shim_path", lambda *a, **k: "/opt/x.so"
    )
    monkeypatch.delenv("CLEARML_AGENT_SNUG_REPORT_USAGE_EVENTS", raising=False)
    w = _stub_worker()  # empty config -> report_usage_events defaults to False
    args = w._get_snug_docker_args(task_id="some-task")

    assert not any(
        "REPORT_USAGE_EVENTS" in str(a) for a in args
    ), "usage env var should not be propagated when flag is off: {}".format(args)


def test_sdk_env_vars_still_present(monkeypatch):
    """Regression: the SNUG logic must not stomp the existing SDK env
    vars that _get_job_os_envs has emitted since forever."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )

    w = _stub_worker()
    envs = w._get_job_os_envs(_FakeTask("the-task-id"), "DEBUG")
    assert any(v == "the-task-id" for v in envs.values())
    assert any(v == "DEBUG" for v in envs.values())


def test_snug_deferred_env_keys_cover_every_snug_env_returned(monkeypatch):
    """Regression guard for the deferred-env fix in execute_task.

    The fix relies on a partition of _get_job_os_envs's output into
    deferred (must wait until after the credential descriptor fd is set) and
    non-deferred (safe to export immediately). If someone adds a new
    SNUG-specific env to _get_job_os_envs (e.g. CLEARML_SNUG_<NEW>) and
    forgets to also add it to _SNUG_DEFERRED_OS_ENVIRON_KEYS, helper
    subprocesses forked between the os.environ.update calls would load the
    shim before the descriptor exists (reporter=stderr) - regressing the fix.

    Catch that here: every SNUG-prefixed key plus LD_PRELOAD that the
    function emits MUST appear in the deferred set.
    """
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )

    # A reporting sink is on so CLEARML_SNUG_PARSE_USAGE is emitted and its
    # deferral is actually exercised by this guard.
    w = _stub_worker(_FakeSession({"agent.snug.report_task_metrics": True}))
    envs = w._get_job_os_envs(_FakeTask("the-task-id"), "INFO")
    snug_keys = {
        k for k in envs
        if k in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES")
        or k.startswith("CLEARML_SNUG_")
        or k == "CLEARML_PROJECT_ID"
    }
    missing = snug_keys - worker_mod._SNUG_DEFERRED_OS_ENVIRON_KEYS
    assert not missing, (
        "These env vars look SNUG-related but are NOT in "
        "_SNUG_DEFERRED_OS_ENVIRON_KEYS: {}. Either add them to the set "
        "(if their export must wait until the descriptor fd is set) or update "
        "this test if a new env genuinely is safe to export pre-bind."
    ).format(sorted(missing))


def test_snug_deferred_env_keys_only_contains_real_keys(monkeypatch):
    """Symmetric guard for the previous test: every key in the deferred
    set must actually be emitted by _get_job_os_envs when SNUG is on.
    Catches typos (e.g. CLEARMl_SNUG_*) or stale entries left behind by
    a refactor that dropped one of the env vars."""
    monkeypatch.setattr(worker_mod, "snug_enabled", lambda session: True)
    monkeypatch.setattr(worker_mod, "resolve_shim_path", lambda: "/opt/x.so")
    monkeypatch.setattr(
        "clearml_agent.snug.whitelist.build_whitelist_env",
        lambda session: "c3R1Yi1iNjQ=",
    )

    # A reporting sink is on so CLEARML_SNUG_PARSE_USAGE (conditionally emitted)
    # is present; without it the env would look stale in the deferred set.
    w = _stub_worker(_FakeSession({"agent.snug.report_task_metrics": True}))
    emitted_keys = set(w._get_job_os_envs(_FakeTask(), "INFO").keys())
    # Some deferred keys are only CONDITIONALLY emitted by _get_job_os_envs, so
    # exclude them from the symmetry check:
    #  * CLEARML_SNUG_CRED_FD — set in execute_task (when the cred fd is created).
    #  * CLEARML_SNUG_WHITELIST_ADDITIONS — set only when the task predefined a
    #    "_snug_whitelist" User Property (none in this stub session).
    #  * CLEARML_SNUG_DEBUG_LOG — set only when agent.snug.debug_log is truthy
    #    (off in this stub session).
    #  * LD_PRELOAD / DYLD_INSERT_LIBRARIES — exactly ONE is emitted per OS (the
    #    other is the other platform's preload var). The host's one is asserted
    #    emitted separately below.
    deferred = worker_mod._SNUG_DEFERRED_OS_ENVIRON_KEYS - {
        "CLEARML_SNUG_CRED_FD",
        "CLEARML_SNUG_WHITELIST_ADDITIONS",
        "CLEARML_SNUG_DEBUG_LOG",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    }
    stale = deferred - emitted_keys
    assert not stale, (
        "These keys are in _SNUG_DEFERRED_OS_ENVIRON_KEYS but are not emitted "
        "by _get_job_os_envs (excluding the execute_task-set CLEARML_SNUG_CRED_FD): "
        "{}. Probably a typo or a stale entry."
    ).format(sorted(stale))
    # The host's preload var (LD_PRELOAD on Linux, DYLD_INSERT_LIBRARIES on
    # macOS) must be emitted, and must itself be in the deferred set.
    inj = worker_mod.injection_env_var()
    assert inj in emitted_keys, "preload var {} not emitted".format(inj)
    assert inj in worker_mod._SNUG_DEFERRED_OS_ENVIRON_KEYS
