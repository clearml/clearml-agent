"""Regression tests for ENVIRONMENT_CONFIG-registered env vars surviving
``Config.reload()`` / ``Config.set_overrides()``.

Without care, ``load_vaults()`` -> ``set_overrides()`` -> ``reload()`` rebuilds
the config tree by merging the on-disk defaults on top of the in-memory state,
which would wipe any value ``Session.__init__`` applied from an
ENVIRONMENT_CONFIG env var wherever the file specifies a default.

``Config._reload()`` therefore re-applies ENVIRONMENT_CONFIG overrides at the
very end, after file merges AND ``_overrides_configs`` (vaults), so env wins
everywhere - matching the precedence the agent already enforces manually for a
subset of keys in ``_apply_extra_configuration``.
"""
from __future__ import absolute_import

import os
from unittest.mock import MagicMock, patch

import pytest

from clearml_agent.backend_config.config import Config
from clearml_agent.definitions import EnvironmentConfig
from clearml_agent.external.pyhocon import ConfigFactory, ConfigTree


@pytest.fixture
def loaded_config(tmp_path, monkeypatch):
    """A ``Config`` initialized from a fixture HOCON file with
    ``agent.snug.enabled = false`` baked in - matches the production
    default that triggered the original bug.
    """
    conf_root = tmp_path / "cfg"
    default_dir = conf_root / "default"
    default_dir.mkdir(parents=True)
    # ``_read_recursive`` namespaces by filename (``agent.conf`` ->
    # ``agent.*``), so we write the contents of the implicit ``agent``
    # block directly.
    (default_dir / "agent.conf").write_text(
        'snug { enabled: false }\n'
        'worker_name: "from-file"\n'
    )

    # Anchor the config so ``load_relative_to`` walks the fixture tree
    # instead of pulling in ~/clearml.conf or the shipped agent.conf. The
    # Config loader expects a file path here whose ``with_name(folder)``
    # points at the config root.
    anchor = tmp_path / "anchor.py"
    anchor.write_text("")

    # Disable LOCAL_CONFIG_FILES (~/clearml.conf etc.) for the duration
    # of the test so user-installed configs don't bleed in.
    monkeypatch.setattr(
        "clearml_agent.backend_config.config.LOCAL_CONFIG_FILES", []
    )
    monkeypatch.setattr(
        "clearml_agent.backend_config.config.LOCAL_CONFIG_PATHS", []
    )

    config = Config(config_folder="cfg", verbose=False)
    config.load_relative_to(str(anchor))
    return config


def test_env_override_survives_reload(loaded_config, monkeypatch):
    """The simplest reproduction: env var set, file says false, after
    reload env wins."""
    monkeypatch.setenv("CLEARML_AGENT_SNUG_ENABLED", "true")

    # Sanity: pre-reload, file default is still in effect because the env
    # var wasn't set when load_relative_to first read the file. We
    # explicitly call reload now with the env var present.
    loaded_config.reload()

    assert loaded_config.get("agent.snug.enabled") is True


def test_env_override_survives_set_overrides(loaded_config, monkeypatch):
    """The actual production path: ``load_vaults()`` calls
    ``set_overrides()`` with vault data. Env vars must still win after."""
    monkeypatch.setenv("CLEARML_AGENT_SNUG_ENABLED", "true")

    # Vault payload that disagrees with the env var.
    vault = ConfigFactory.from_dict({"agent": {"snug": {"enabled": False}}})
    loaded_config.set_overrides(vault)

    assert loaded_config.get("agent.snug.enabled") is True, (
        "Env var should beat vault override; got vault value instead."
    )


def test_env_override_beats_vault_for_arbitrary_registered_key(
    loaded_config, monkeypatch
):
    """Not just agent.snug.enabled - the fix applies to every key
    registered in ENVIRONMENT_CONFIG. Use ``agent.worker_name`` here,
    which the file fixture sets to 'from-file' and is registered to the
    ``CLEARML_WORKER_NAME`` env var."""
    monkeypatch.setenv("CLEARML_WORKER_NAME", "from-env")

    vault = ConfigFactory.from_dict({"agent": {"worker_name": "from-vault"}})
    loaded_config.set_overrides(vault)

    assert loaded_config.get("agent.worker_name") == "from-env"


def test_snug_app_mode_env_selects_profile(loaded_config, monkeypatch):
    """``agent.snug.app_mode`` must be env-overridable. The shipped default is
    ``""`` (SNUG app-mode is opt-in), so the ``CLEARML_AGENT_SNUG_APP_MODE`` env
    binding is the only thing that can select a profile from an image; without
    it registered the value is silently dropped and the app-mode proxy never
    runs. Env must also beat a vault that leaves the profile empty."""
    monkeypatch.setenv("CLEARML_AGENT_SNUG_APP_MODE", "claude_desktop")

    vault = ConfigFactory.from_dict({"agent": {"snug": {"app_mode": ""}}})
    loaded_config.set_overrides(vault)

    assert loaded_config.get("agent.snug.app_mode") == "claude_desktop"


def test_no_env_var_leaves_file_default_intact(loaded_config, monkeypatch):
    """If the env var isn't set, the file default must stay.

    Guards against the re-apply loop accidentally writing empty strings or
    None into the config tree."""
    monkeypatch.delenv("CLEARML_AGENT_SNUG_ENABLED", raising=False)

    loaded_config.reload()

    assert loaded_config.get("agent.snug.enabled") is False
    assert loaded_config.get("agent.worker_name") == "from-file"


def test_apply_environment_overrides_skips_list_keys():
    """``.0``-style append-semantics keys (sdk.azure.storage.containers.0)
    must not be re-applied; otherwise reloads would accumulate duplicate
    list entries.

    We invoke ``_apply_environment_overrides`` directly with an injected
    ENVIRONMENT_CONFIG that contains a ``.0`` entry plus a scalar, then
    confirm only the scalar landed.
    """
    fake_env = {
        "agent.snug.enabled": EnvironmentConfig(
            "_TEST_BOGUS_SNUG_ENABLED", type=bool
        ),
        "sdk.fake.list.0": {
            "field": EnvironmentConfig("_TEST_BOGUS_LIST_FIELD"),
        },
    }

    with patch.dict(
        os.environ,
        {"_TEST_BOGUS_SNUG_ENABLED": "true",
         "_TEST_BOGUS_LIST_FIELD": "shouldnotappear"},
    ), patch(
        "clearml_agent.definitions.ENVIRONMENT_CONFIG", fake_env
    ):
        result = Config._apply_environment_overrides(ConfigTree())

    assert result.get("agent.snug.enabled") is True
    assert "sdk" not in result, (
        "list-style .0 key was re-applied even though that path is "
        "expected to be skipped."
    )


def test_apply_environment_overrides_skips_empty_value():
    """Empty-string env value (env var set to ``""``) is treated as 'no
    override', matching how ``EnvironmentConfig.get()`` behaves when the
    var is unset."""
    fake_env = {
        "agent.worker_name": EnvironmentConfig("_TEST_BOGUS_WORKER"),
    }
    initial = ConfigFactory.from_dict({"agent": {"worker_name": "kept"}})

    with patch.dict(os.environ, {"_TEST_BOGUS_WORKER": ""}), patch(
        "clearml_agent.definitions.ENVIRONMENT_CONFIG", fake_env
    ):
        result = Config._apply_environment_overrides(initial)

    assert result.get("agent.worker_name") == "kept"


# --- Monitoring vault: highest-priority overlay (admin beats env+file) -------


def test_monitoring_priority_override_beats_env(loaded_config, monkeypatch):
    """The monitoring vault INVERTS the normal env>vault rule: an admin
    value set via ``set_priority_overrides`` beats even an env var. This is
    the headline 'admin wins, a user can't override' behaviour."""
    monkeypatch.setenv("CLEARML_AGENT_SNUG_ENABLED", "true")

    # Config vault still loses to env (the existing rule, unchanged).
    loaded_config.set_overrides(
        ConfigFactory.from_dict({"agent": {"snug": {"enabled": False}}})
    )
    assert loaded_config.get("agent.snug.enabled") is True

    # The monitoring (priority) overlay flips it: admin false beats env true.
    loaded_config.set_priority_overrides(
        ConfigFactory.from_dict({"agent": {"snug": {"enabled": False}}})
    )
    assert loaded_config.get("agent.snug.enabled") is False, (
        "Monitoring vault must beat env var; got the env value instead."
    )


def test_monitoring_and_config_vault_whitelists_combine(loaded_config):
    """Both vault tiers are an admin base: the monitoring (priority) vault
    outranks the config vault (authoritative on a host collision) and the config
    vault's other hosts are added beneath it. With no file whitelist in this
    fixture, the effective whitelist is exactly the combined admin rules."""
    loaded_config.set_overrides(ConfigFactory.from_dict(
        {"agent": {"snug": {"whitelist": {"version": 1,
                                          "rules": [{"host": "user.example"}]}}}}
    ))
    loaded_config.set_priority_overrides(ConfigFactory.from_dict(
        {"agent": {"snug": {"whitelist": {"version": 1,
                                          "rules": [{"host": "admin.example"}]}}}}
    ))
    rules = loaded_config.get("agent.snug.whitelist.rules")
    hosts = [r["host"] for r in rules]
    assert hosts == ["admin.example", "user.example"], (
        "Monitoring must be authoritative (first) with the config vault beneath "
        "it; got {}".format(hosts)
    )


def test_empty_priority_overlay_leaves_env_winning(loaded_config, monkeypatch):
    """Guard: an empty priority slot must not clobber anything - env still
    beats the config vault (the monitoring tier doesn't change that)."""
    monkeypatch.setenv("CLEARML_AGENT_SNUG_ENABLED", "true")
    loaded_config.set_overrides(
        ConfigFactory.from_dict({"agent": {"snug": {"enabled": False}}})
    )
    loaded_config.set_priority_overrides()  # empty slot -> no-op
    assert loaded_config.get("agent.snug.enabled") is True


def test_load_monitoring_vault_applies_priority(monkeypatch):
    """``Session.load_monitoring_vault`` fetches the ``monitoring`` vault type
    and applies it via ``set_priority_overrides``."""
    from clearml_agent.backend_api.session import session as session_mod
    from clearml_agent.backend_api.session.session import Session

    monkeypatch.setattr(
        session_mod.ENV_DISABLE_VAULT_SUPPORT, "get", lambda *a, **k: False
    )
    s = MagicMock()
    s.check_min_api_version.return_value = True
    s.feature_set = "advanced"
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {
        "data": {"vaults": [{"data": "agent { snug { enabled: false } }"}]}
    }
    s.send_request.return_value = resp

    result = Session.load_monitoring_vault(s)

    assert result is True
    # The new monitoring type was requested.
    _, kwargs = s.send_request.call_args
    assert "types=monitoring" in kwargs.get("params", "")
    s.config.set_priority_overrides.assert_called_once()


def test_load_monitoring_vault_unknown_type_is_noop(monkeypatch):
    """Forward-compat: a backend that doesn't support ``types=monitoring``
    (404 / non-ok) must be a clean no-op - never raise, never apply."""
    from clearml_agent.backend_api.session import session as session_mod
    from clearml_agent.backend_api.session.session import Session

    monkeypatch.setattr(
        session_mod.ENV_DISABLE_VAULT_SUPPORT, "get", lambda *a, **k: False
    )
    s = MagicMock()
    s.check_min_api_version.return_value = True
    s.feature_set = "advanced"
    resp = MagicMock()
    resp.ok = False
    resp.status_code = 404
    s.send_request.return_value = resp

    result = Session.load_monitoring_vault(s)  # must not raise
    assert not result
    s.config.set_priority_overrides.assert_not_called()
