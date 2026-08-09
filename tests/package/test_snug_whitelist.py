"""Tests for clearml_agent.snug.whitelist.

build_whitelist_env() resolves the inline ``agent.snug.whitelist`` config
block and returns it base64-encoded for the shim to read from
CLEARML_SNUG_WHITELIST. These tests lock the builder's defaulting,
pass-through, fallback, and encoding behaviors.
"""
import base64
import json

from clearml_agent.snug import whitelist as wl
from clearml_agent.external.pyhocon import ConfigFactory


class _FakeConfig(object):
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, key, default=None):
        return self._mapping.get(key, default)


class _FakeSession(object):
    def __init__(self, mapping=None):
        self.config = _FakeConfig(mapping or {})


def _decode(env_value):
    """Decode what build_whitelist_env produced back into a dict."""
    return json.loads(base64.b64decode(env_value).decode("utf-8"))


def test_empty_skeleton_when_whitelist_unset():
    session = _FakeSession({})
    data = _decode(wl.build_whitelist_env(session))
    assert data == {"version": 1, "default_action": "meter", "rules": []}


def test_inline_whitelist_passed_through():
    content = {
        "version": 1,
        "default_action": "meter",
        "rules": [
            {
                "host": "api.anthropic.com",
                "path_prefix": "/v1/",
                "debug": False,
                "inject_headers": True,
                "tokenizer": "claude",
            }
        ],
    }
    session = _FakeSession({"agent.snug.whitelist": content})
    data = _decode(wl.build_whitelist_env(session))
    assert data == content


def test_wildcard_host_passes_through_verbatim():
    """A wildcard host pattern is an ordinary string; the builder hands it to
    the shim untouched (the shim is the matcher). Locks in that '*' isn't
    stripped or rejected agent-side."""
    content = {
        "version": 1,
        "default_action": "meter",
        "rules": [
            {"host": "*.anthropic.com", "path_prefix": "/v1/"},
            {"host": "api.openai.*"},
            {"host": "*"},
        ],
    }
    session = _FakeSession({"agent.snug.whitelist": content})
    data = _decode(wl.build_whitelist_env(session))
    assert [r["host"] for r in data["rules"]] == [
        "*.anthropic.com",
        "api.openai.*",
        "*",
    ]


def test_non_dict_whitelist_falls_back_to_skeleton():
    """A non-dict value (misconfiguration) must NOT be passed to the shim;
    coerce to the v1 empty skeleton."""
    session = _FakeSession({"agent.snug.whitelist": "garbage"})
    data = _decode(wl.build_whitelist_env(session))
    assert data == {"version": 1, "default_action": "meter", "rules": []}


def test_non_v1_whitelist_falls_back_to_skeleton():
    """A v2 block (or anything we don't recognize) should NOT be passed
    through - the shim's parser is v1. Coerce to the v1 empty skeleton."""
    session = _FakeSession({"agent.snug.whitelist": {"version": 99, "rules": []}})
    data = _decode(wl.build_whitelist_env(session))
    assert data["version"] == 1
    assert data["rules"] == []


def test_output_is_base64_with_no_newlines():
    """The value travels in an env var (and across docker -e), so it must
    be a single line of standard base64 - no embedded newlines."""
    session = _FakeSession({"agent.snug.whitelist": {"version": 1, "rules": []}})
    env_value = wl.build_whitelist_env(session)
    assert "\n" not in env_value
    # Round-trips cleanly back to JSON.
    assert _decode(env_value)["version"] == 1


def test_configtree_like_input_is_coerced():
    """A pyhocon ConfigTree exposes as_plain_ordered_dict(); the builder
    must use it so json.dumps gets a plain JSON-serializable structure."""

    class _FakeTree(dict):
        def as_plain_ordered_dict(self):
            return {"version": 1, "default_action": "meter",
                    "rules": [{"host": "x.example"}]}

    session = _FakeSession({"agent.snug.whitelist": _FakeTree()})
    data = _decode(wl.build_whitelist_env(session))
    assert data["rules"] == [{"host": "x.example"}]


# --- union_whitelist (admin-protected merge of vault base + file adds) ------


def test_union_user_adds_hosts_on_top_of_vault_base():
    """The reported scenario: admin/config-vault whitelist has only Gemini; the
    user's file adds openai + anthropic + gemini-again. Result keeps the admin
    rule and ADDS the two new hosts; the duplicate gemini is dropped (covered by
    the admin rule), and default_action/version stay the admin's."""
    base = {
        "version": 1,
        "default_action": "meter",
        "rules": [{"host": "generativelanguage.googleapis.com", "tokenizer": "approx"}],
    }
    file_rules = [
        {"host": "api.openai.com", "tokenizer": "cl100k", "inject_headers": True},
        {"host": "api.anthropic.com", "tokenizer": "claude", "inject_headers": True},
        {"host": "generativelanguage.googleapis.com", "tokenizer": "approx"},
    ]
    merged = wl.union_whitelist(base, file_rules)
    assert [r["host"] for r in merged["rules"]] == [
        "generativelanguage.googleapis.com",  # admin base, first (authoritative)
        "api.openai.com",
        "api.anthropic.com",
    ]
    assert merged["default_action"] == "meter"
    assert merged["version"] == 1
    # The user's added hosts keep their own fields.
    anthropic = merged["rules"][2]
    assert anthropic["tokenizer"] == "claude" and anthropic["inject_headers"] is True


def test_union_user_cannot_override_or_remove_admin_rule():
    """A file rule whose host collides with an admin rule is dropped wholesale —
    the user can't flip the admin rule's tokenizer/inject_headers, and can't
    remove it. The admin rule is returned untouched."""
    base = {
        "version": 1,
        "default_action": "ignore",
        "rules": [{"host": "api.anthropic.com", "tokenizer": "approx", "inject_headers": False}],
    }
    file_rules = [{"host": "api.anthropic.com", "tokenizer": "claude", "inject_headers": True}]
    merged = wl.union_whitelist(base, file_rules)
    assert len(merged["rules"]) == 1
    assert merged["rules"][0]["tokenizer"] == "approx"      # admin's, not the file's
    assert merged["rules"][0]["inject_headers"] is False
    assert merged["default_action"] == "ignore"             # policy stays admin's


def test_union_wildcard_covered_host_survives_but_base_stays_first():
    """Coverage is EXACT-host only — a concrete host a base WILDCARD also matches
    is NOT dropped (matching is the shim's job). It survives as a shadowed entry,
    but the base wildcard rule is emitted FIRST so the shim's first-match-wins
    keeps the admin rule authoritative at runtime."""
    base = {"version": 1, "default_action": "meter", "rules": [{"host": "*.acme.com"}]}
    merged = wl.union_whitelist(base, [{"host": "api.acme.com"}, {"host": "api.other.com"}])
    assert [r["host"] for r in merged["rules"]] == ["*.acme.com", "api.acme.com", "api.other.com"]
    assert merged["rules"][0]["host"] == "*.acme.com"  # admin wildcard authoritative (first)


def test_union_dedups_additions_case_insensitively():
    base = {"version": 1, "default_action": "meter", "rules": []}
    merged = wl.union_whitelist(base, [{"host": "Api.X.com"}, {"host": "api.x.com"}])
    assert [r["host"] for r in merged["rules"]] == ["Api.X.com"]


def test_union_empty_additions_returns_base_rules():
    base = {"version": 1, "default_action": "meter", "rules": [{"host": "a.com"}]}
    assert wl.union_whitelist(base, [])["rules"] == [{"host": "a.com"}]
    assert wl.union_whitelist(base, None)["rules"] == [{"host": "a.com"}]


# --- resolve_effective_whitelist (admin vault base + file additions) --------


def _ov(rules):
    """An override-config list (as Config stores a vault layer) for a whitelist."""
    return [ConfigFactory.from_dict({"agent": {"snug": {"whitelist": {
        "version": 1, "default_action": "meter", "rules": rules,
    }}}})]


def test_resolve_none_without_any_vault():
    """No vault whitelist -> None (the caller leaves the resolved config as the
    file's)."""
    file_wl = {"version": 1, "default_action": "meter", "rules": [{"host": "openai.example"}]}
    assert wl.resolve_effective_whitelist(file_wl, [], []) is None


def test_resolve_monitoring_base_plus_file_adds():
    """Monitoring (priority) vault is the base; the file adds its non-colliding
    hosts (the duplicate gemini is dropped)."""
    mon = _ov([{"host": "gemini.example"}])
    file_wl = {"version": 1, "default_action": "meter",
               "rules": [{"host": "openai.example"}, {"host": "gemini.example"}]}
    eff = wl.resolve_effective_whitelist(file_wl, [], mon)
    assert [r["host"] for r in eff["rules"]] == ["gemini.example", "openai.example"]


def test_resolve_combines_both_vault_tiers_then_file():
    """Both vault tiers form the base (monitoring authoritative/first, config
    vault beneath); the file adds on top."""
    vault = _ov([{"host": "config-vault.example"}])
    mon = _ov([{"host": "monitoring.example"}])
    file_wl = {"version": 1, "default_action": "meter", "rules": [{"host": "file.example"}]}
    eff = wl.resolve_effective_whitelist(file_wl, vault, mon)
    assert [r["host"] for r in eff["rules"]] == [
        "monitoring.example", "config-vault.example", "file.example",
    ]


def test_resolve_config_vault_only_base_plus_file():
    """Config vault alone is the base; the file adds on top."""
    vault = _ov([{"host": "gemini.example"}])
    file_wl = {"version": 1, "default_action": "meter", "rules": [{"host": "openai.example"}]}
    eff = wl.resolve_effective_whitelist(file_wl, vault, [])
    assert [r["host"] for r in eff["rules"]] == ["gemini.example", "openai.example"]
