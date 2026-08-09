"""Test that locks the ``agent.snug`` HOCON defaults.

These are the keys the shipped agent.conf snug block provides with the
listed defaults. The rest of the feature reads them; renaming or removing
any of them is a contract bump.
"""
import os

import pytest

# pyhocon is vendored inside the agent under clearml_agent.external.
from clearml_agent.external.pyhocon import ConfigFactory


AGENT_CONF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "clearml_agent",
    "backend_api",
    "config",
    "default",
    "agent.conf",
)


@pytest.fixture(scope="module")
def agent_conf():
    """Parse the default agent.conf once per module.

    The file is wrapped in an implicit ``agent { ... }`` block at runtime by
    the agent's config loader; the on-disk file is the contents of that
    block. We parse it as-is here, so keys are accessed without an ``agent.``
    prefix.
    """
    return ConfigFactory.parse_file(AGENT_CONF_PATH)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("snug.enabled", False),
        # call-history capture (4-mode request/response logging)
        ("snug.call_history", "off"),
        ("snug.call_history_buffer", 50),
        ("snug.call_history_cap_bytes", 262144),
        ("snug.poll_interval_sec", 10),
        # fallback tokenizer for non-whitelisted connections so
        # every metered request gets a tokens_est value.
        ("snug.default_tokenizer", "approx"),
        # whitelist is an inline block now (see test_snug_default_whitelist)
        ("snug.whitelist.version", 1),
        ("snug.whitelist.default_action", "meter"),
        # null in HOCON -> Python None
        ("snug.aggregator_url", None),
        ("snug.report_usage_events", False),
        ("snug.report_task_metrics", False),
        # verbose per-process [snug] shim diagnostics (default off)
        ("snug.debug_log", False),
        (
            "snug.task_metrics_fields",
            [
                "tokens_in",
                "tokens_out",
                "cache_read_tokens",
                "cache_write_tokens",
                "requests",
                "latency_ms",
                "bytes_tx",
                "bytes_rx",
                "tool_calls",
                "tool_call_errors",
            ],
        ),
    ],
)
def test_snug_default(agent_conf, key, expected):
    """Every documented default is present with the documented value."""
    # ConfigTree.get returns the sentinel default we pass in if missing; we
    # use a distinctive sentinel so a "key missing" failure looks different
    # from a "value wrong" failure.
    sentinel = object()
    actual = agent_conf.get(key, sentinel)
    assert actual is not sentinel, "missing key: {}".format(key)
    assert actual == expected, "expected {!r} for {}, got {!r}".format(
        expected, key, actual
    )


def test_snug_block_has_no_unknown_keys(agent_conf):
    """Catch typos / accidental additions to the snug block early.

    Adding a new HOCON key to the contract requires updating the shipped
    agent.conf and this test. Failing here means one was changed without
    the other.
    """
    expected_keys = {
        "enabled",
        "app_mode",                 # app-mode profile selector (opt-in)
        "call_history",
        "call_history_buffer",
        "call_history_cap_bytes",
        "poll_interval_sec",
        "default_tokenizer",
        "whitelist",
        "aggregator_url",
        "report_usage_events",    # usage-events sink
        "report_task_metrics",      # task-metrics (SCALARS) sink
        "task_metrics_fields",      # task-metrics field selection
        "debug_log",                # verbose per-process [snug] diagnostics
    }
    actual_keys = set(agent_conf["snug"].keys())
    assert actual_keys == expected_keys, (
        "snug block keys drifted from the shipped agent.conf snug block.\n"
        "added: {added}\nremoved: {removed}".format(
            added=sorted(actual_keys - expected_keys),
            removed=sorted(expected_keys - actual_keys),
        )
    )


def test_snug_default_whitelist(agent_conf):
    """Lock the inlined default whitelist: a v1 block with rules for the
    common LLM providers and their per-host tokenizers. If a host is
    dropped or a tokenizer changed, update this test deliberately -
    operators who enable SNUG see the behaviour change out of the box.

    Every shipped rule meters WITHOUT header injection: the project:/session:
    headers carry ClearML identifiers to a third-party provider, so opting a
    host in is the operator's call, never a default.
    """
    rules = agent_conf.get("snug.whitelist.rules")
    assert isinstance(rules, list) and len(rules) > 0
    by_host = {r["host"]: r for r in rules}

    assert by_host["api.openai.com"]["tokenizer"] == "cl100k"

    assert by_host["api.anthropic.com"]["tokenizer"] == "claude"

    # Gemini has no first-class tokenizer in the shim's enum
    # (claude/cl100k/approx); fall back to approx.
    assert by_host["generativelanguage.googleapis.com"]["tokenizer"] == "approx"

    assert all(r["inject_headers"] is False for r in rules), (
        "shipped whitelist must not inject attribution headers by default: {}".format(
            [r["host"] for r in rules if r.get("inject_headers")]
        )
    )
