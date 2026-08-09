"""Tests for the SNUG whitelist layering across config-precedence layers
(``Config._apply_snug_whitelist`` ->
``clearml_agent.snug.whitelist.resolve_effective_whitelist``).

A config-vault whitelist is a PROTECTED BASE the user's file whitelist ADDS to:
admin rules always win and policy (``default_action``/``version``) stays the
admin's, but the user can extend the whitelist with new hosts. Without the
reconciliation the generic list-merge would REPLACE the file's rules with the
vault's, silently dropping every host the user added in their own config (the
reported "rules=1, only Gemini" symptom).

The monitoring (priority) vault is also a protected base: it is authoritative
over the config vault, but the two combine into the admin base the file adds to
(not a hard replace).
"""
from __future__ import absolute_import

import pytest

from clearml_agent.backend_config.config import Config
from clearml_agent.external.pyhocon import ConfigFactory


def _hosts(config):
    """Resolved ``agent.snug.whitelist`` rule hosts, in order."""
    wl = config.get("agent.snug.whitelist", None)
    if wl is None:
        return None
    return [r["host"] for r in wl.get("rules", [])]


@pytest.fixture
def file_config(tmp_path, monkeypatch):
    """A ``Config`` whose FILE layer carries a three-provider whitelist - the
    user's own clearml.conf. Isolated from ~/clearml.conf and the shipped
    defaults so only this whitelist is the file layer."""
    conf_root = tmp_path / "cfg"
    default_dir = conf_root / "default"
    default_dir.mkdir(parents=True)
    # ``_read_recursive`` namespaces by filename (agent.conf -> agent.*), so the
    # file body is the contents of the implicit ``agent`` block.
    (default_dir / "agent.conf").write_text(
        'snug {\n'
        '  enabled: true\n'
        '  whitelist {\n'
        '    version: 1\n'
        '    default_action: "meter"\n'
        '    rules: [\n'
        '      { host: "api.openai.com", path_prefix: "/v1/", tokenizer: "cl100k", inject_headers: true }\n'
        '      { host: "api.anthropic.com", path_prefix: "/v1/", tokenizer: "claude", inject_headers: true }\n'
        '      { host: "generativelanguage.googleapis.com", path_prefix: "/", tokenizer: "approx" }\n'
        '    ]\n'
        '  }\n'
        '}\n'
    )
    anchor = tmp_path / "anchor.py"
    anchor.write_text("")
    monkeypatch.setattr("clearml_agent.backend_config.config.LOCAL_CONFIG_FILES", [])
    monkeypatch.setattr("clearml_agent.backend_config.config.LOCAL_CONFIG_PATHS", [])

    config = Config(config_folder="cfg", verbose=False)
    config.load_relative_to(str(anchor))
    return config


def _vault(rules, default_action="meter"):
    return ConfigFactory.from_dict(
        {"agent": {"snug": {"whitelist": {
            "version": 1, "default_action": default_action, "rules": rules,
        }}}}
    )


def test_no_vault_leaves_file_whitelist_intact(file_config):
    """Sanity: with no vault, the user's file whitelist is used verbatim."""
    assert _hosts(file_config) == [
        "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
    ]


def test_config_vault_is_base_and_user_file_adds(file_config):
    """The reported bug: a config-vault whitelist with only Gemini must NOT wipe
    the user's openai/anthropic. After reconciliation the effective whitelist is
    the admin base PLUS the user's added hosts (the duplicate Gemini is dropped,
    covered by the admin rule)."""
    file_config.set_overrides(
        _vault([{"host": "generativelanguage.googleapis.com", "tokenizer": "approx"}])
    )
    assert _hosts(file_config) == [
        "generativelanguage.googleapis.com",  # admin base, authoritative (first)
        "api.openai.com",
        "api.anthropic.com",
    ]


def test_user_cannot_override_admin_rule_via_file(file_config):
    """A file rule colliding with an admin host is dropped wholesale: the admin
    rule's settings and the admin ``default_action`` are preserved. The file's
    anthropic (inject_headers/claude) cannot change the admin's metering-only
    anthropic, and the file's other hosts can't change default_action."""
    file_config.set_overrides(
        _vault(
            [{"host": "api.anthropic.com", "tokenizer": "approx", "inject_headers": False}],
            default_action="ignore",
        )
    )
    wl = file_config.get("agent.snug.whitelist")
    rules = list(wl.get("rules"))
    anthropic = next(r for r in rules if r["host"] == "api.anthropic.com")
    assert anthropic["tokenizer"] == "approx"        # admin's, not file's "claude"
    assert anthropic.get("inject_headers", False) is False
    assert wl.get("default_action") == "ignore"      # policy stays admin's
    # openai is not covered by the admin rule, so it's still added.
    assert "api.openai.com" in [r["host"] for r in rules]


def test_monitoring_vault_is_protected_base_and_file_adds(file_config):
    """The monitoring (priority) vault is the admin BASE the user's file whitelist
    ADDS to - NOT a hard replace. Admin rules stay authoritative (emitted first);
    the file's non-colliding hosts are appended, the colliding one (gemini) is
    dropped."""
    file_config.set_priority_overrides(
        _vault([{"host": "generativelanguage.googleapis.com", "tokenizer": "approx"}])
    )
    assert _hosts(file_config) == [
        "generativelanguage.googleapis.com",  # admin base, authoritative (first)
        "api.openai.com",
        "api.anthropic.com",
    ]


def test_both_vault_tiers_combine_as_base_then_file_adds(file_config):
    """Both vault tiers form the admin base: the monitoring vault is authoritative
    (first), the config vault's hosts come beneath it, and the file adds on top."""
    file_config.set_overrides(_vault([{"host": "config-vault.example"}]))
    file_config.set_priority_overrides(_vault([{"host": "monitoring.example"}]))
    assert _hosts(file_config) == [
        "monitoring.example",     # monitoring vault — authoritative
        "config-vault.example",   # config vault — beneath monitoring
        "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",  # file adds
    ]


def test_config_vault_without_whitelist_leaves_file_intact(file_config):
    """A vault that sets OTHER keys but no whitelist (e.g. an API-key vault) must
    not disturb the user's file whitelist."""
    file_config.set_overrides(
        ConfigFactory.from_dict({"agent": {"snug": {"enabled": True}}})
    )
    assert _hosts(file_config) == [
        "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
    ]
