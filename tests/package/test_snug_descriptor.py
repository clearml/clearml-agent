"""Tests for the in-process reporter handoff descriptor builder in
clearml_agent.helper.snug.

The descriptor is written into an inheritable fd whose number the agent passes
to the shim via CLEARML_SNUG_CRED_FD: an anonymous ``memfd`` on Linux (no
on-disk file) and an immediately-unlinked 0600 temp file on macOS (no
``memfd_create``). The Rust reporter itself is tested in clearml_snug/reporter
(cargo test); these cover the agent-side wiring that feeds it.
"""
import json
import os

import clearml_agent.helper.snug as snug


class _FakeConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeSession(object):
    """Stand-in for the agent Session: exposes host/access_key/secret_key/token
    and a .config.get(...) like the real one."""

    def __init__(
        self,
        host="https://api.example.com",
        access_key="AK",
        secret_key="SK",
        token=None,
        **cfg
    ):
        self.host = host
        self.access_key = access_key
        self.secret_key = secret_key
        self.token = token
        self.config = _FakeConfig(cfg)

    # Mirror the real Session's host resolvers (classmethods there, instance
    # methods here): read the files/web servers from config, None when unset.
    def get_files_server_host(self, config):
        return config.get("api.files_server", None)

    def get_app_server_host(self, config):
        return config.get("api.web_server", None)


def _read_descriptor(fd):
    """Read + parse the descriptor JSON the agent wrote into the memfd."""
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        b = os.read(fd, 65536)
        if not b:
            break
        chunks.append(b)
    return json.loads(b"".join(chunks).decode("utf-8"))


def test_build_descriptor_shape_and_inheritable():
    s = _FakeSession(**{"api.verify_certificate": True})
    fd = snug.build_shim_descriptor_fd(s, "task1", worker_id="w1")
    try:
        # The fd must be inheritable so it survives execv / can be passed to the
        # task subprocess via pass_fds.
        assert os.get_inheritable(fd) is True, "descriptor fd must be inheritable"
        d = _read_descriptor(fd)
        assert d["api_server"] == "https://api.example.com"
        assert d["access_key"] == "AK" and d["secret_key"] == "SK"
        assert d["task_id"] == "task1" and d["worker_id"] == "w1"
        # user/project default empty when the caller doesn't supply them (the
        # backend then derives them from the task for report_llm_usage).
        assert d["user"] == "" and d["project"] == ""
        assert d["verify_certificate"] is True and d["ca_cert_path"] is None
        # No socket_path: reporting is in-process, not over a socket.
        assert "socket_path" not in d
        # Sinks default off / empty when nothing is configured.
        assert d["report_usage_events"] is False
        assert d["report_task_metrics"] is False
        assert d["task_metrics_fields"] == []
        assert d["aggregator_url"] is None
        # With no files/web config the fake resolves only the api server host
        # (port-stripped, lowercased) — the task's own backend, excluded.
        assert d["self_hosts"] == ["api.example.com"]
    finally:
        os.close(fd)


def test_descriptor_self_hosts_api_files_web():
    # SaaS-shaped: distinct api/files/web hostnames all land in the exclusion
    # list, api first.
    s = _FakeSession(
        host="https://api.clear.ml",
        **{
            "api.files_server": "https://files.clear.ml",
            "api.web_server": "https://app.clear.ml",
        }
    )
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["self_hosts"] == ["api.clear.ml", "files.clear.ml", "app.clear.ml"]
    finally:
        os.close(fd)


def test_self_hosts_self_hosted_ports_collapse_to_one_host():
    # Self-hosted: api/files/web share a hostname, differing only by port. They
    # dedup to a single bare host (the shim matches port-insensitively, so one
    # entry covers all three services).
    s = _FakeSession(
        host="http://localhost:8008",
        **{
            "api.files_server": "http://localhost:8081",
            "api.web_server": "http://localhost:8080",
        }
    )
    assert snug._self_hosts(s) == ["localhost"]


def test_hostname_extraction_edge_cases():
    assert snug._hostname("https://API.Clear.ML:443/api") == "api.clear.ml"
    assert snug._hostname("api.clear.ml") == "api.clear.ml"  # bare host, no scheme
    assert snug._hostname("http://localhost:8008") == "localhost"
    assert snug._hostname(None) is None
    assert snug._hostname("") is None


def test_descriptor_sink_fields_from_config():
    s = _FakeSession(**{
        "agent.snug.report_usage_events": True,
        "agent.snug.report_task_metrics": True,
        "agent.snug.task_metrics_fields": ["tokens_in", "requests"],
        "agent.snug.aggregator_url": "https://agg.example/ingest",
    })
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["report_usage_events"] is True
        assert d["report_task_metrics"] is True
        assert d["task_metrics_fields"] == ["tokens_in", "requests"]
        assert d["aggregator_url"] == "https://agg.example/ingest"
    finally:
        os.close(fd)


def test_descriptor_task_metrics_fields_env_override(monkeypatch):
    # CLEARML_AGENT_SNUG_TASK_METRICS_FIELDS (comma list) overrides the config.
    monkeypatch.setenv("CLEARML_AGENT_SNUG_TASK_METRICS_FIELDS", "bytes_tx, bytes_rx")
    s = _FakeSession(**{"agent.snug.task_metrics_fields": ["tokens_in"]})
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["task_metrics_fields"] == ["bytes_tx", "bytes_rx"]
    finally:
        os.close(fd)


def test_descriptor_verify_string_becomes_ca_path():
    # ClearML allows api.verify_certificate to be a CA-bundle path string.
    s = _FakeSession(**{"api.verify_certificate": "/etc/ssl/custom-ca.pem"})
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["ca_cert_path"] == "/etc/ssl/custom-ca.pem"
        assert d["verify_certificate"] is True
    finally:
        os.close(fd)


def test_descriptor_carries_user_and_project():
    # The agent passes the task's owning user + project for usage attribution on
    # report_llm_usage; they ride the descriptor verbatim.
    s = _FakeSession()
    fd = snug.build_shim_descriptor_fd(s, "t", user="u-1", project="p-2")
    try:
        d = _read_descriptor(fd)
        assert d["user"] == "u-1"
        assert d["project"] == "p-2"
    finally:
        os.close(fd)


def test_descriptor_user_falls_back_to_session_user_id():
    # A launcher that omits `user` still attributes usage: the descriptor falls
    # back to the session's authenticated user id (Session.user_id, decoded from
    # the token). Without this the usage event carries no user -> "Unattributed".
    s = _FakeSession()
    s.user_id = "u-session"
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["user"] == "u-session"
    finally:
        os.close(fd)


def test_descriptor_explicit_user_beats_session_user_id():
    # An explicitly-passed user (the worker passes the task owner) takes
    # precedence over the session's own user id.
    s = _FakeSession()
    s.user_id = "u-session"
    fd = snug.build_shim_descriptor_fd(s, "t", user="u-owner")
    try:
        d = _read_descriptor(fd)
        assert d["user"] == "u-owner"
    finally:
        os.close(fd)


def test_descriptor_user_empty_when_no_owner_and_no_session_user():
    # No explicit user and no session user id -> empty (backend derives from
    # the task). The fake session has no user_id attribute.
    s = _FakeSession()
    assert not hasattr(s, "user_id")
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["user"] == ""
    finally:
        os.close(fd)


def test_descriptor_carries_token():
    # Token-primary path: the session's current token rides along so the reporter
    # can use it immediately and Bearer-renew it.
    s = _FakeSession(token="jwt-abc")
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["auth_token"] == "jwt-abc"
    finally:
        os.close(fd)


def test_descriptor_token_only_no_key_secret():
    # Many deployments have only a token (no access/secret). The descriptor still
    # builds: empty creds + the token (reporter Bearer-renews; see api.rs).
    s = _FakeSession(access_key="", secret_key="", token="only-token")
    fd = snug.build_shim_descriptor_fd(s, "t")
    try:
        d = _read_descriptor(fd)
        assert d["access_key"] == "" and d["secret_key"] == ""
        assert d["auth_token"] == "only-token"
    finally:
        os.close(fd)


def test_build_descriptor_uses_tmpfile_when_no_memfd(monkeypatch):
    """macOS has no os.memfd_create, so build_shim_descriptor_fd falls to the
    unlinked-tempfile path. Force that branch (even on Linux) and assert the
    resulting fd is an inheritable, seekable, readable regular file carrying the
    descriptor — and that nothing is left behind by name (it was unlinked)."""
    # Force the tmpfile branch: hide memfd_create if this host has it (Linux).
    if hasattr(os, "memfd_create"):
        monkeypatch.delattr(os, "memfd_create")
    assert not hasattr(os, "memfd_create")

    import stat as _stat
    s = _FakeSession(token="jwt-xyz")
    fd = snug.build_shim_descriptor_fd(s, "task-tmp", worker_id="w")
    try:
        assert os.get_inheritable(fd) is True, "descriptor fd must be inheritable"
        st = os.fstat(fd)
        # The shim's fstat guard requires a REGULAR file (not a pipe/socket).
        assert _stat.S_ISREG(st.st_mode), "tmpfile fd must be a regular file"
        # Unlinked: only the open fd references the inode now.
        assert st.st_nlink == 0, "tmpfile must be unlinked (st_nlink == 0)"
        d = _read_descriptor(fd)
        assert d["task_id"] == "task-tmp" and d["worker_id"] == "w"
        assert d["auth_token"] == "jwt-xyz"
    finally:
        os.close(fd)
