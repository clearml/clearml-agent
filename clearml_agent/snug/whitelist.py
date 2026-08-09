"""Build the SNUG whitelist payload the shim reads via CLEARML_SNUG_WHITELIST.

The whitelist is a normal config block (``agent.snug.whitelist``). The
executioner resolves it from the session config, serializes it to compact
JSON, and base64-encodes it into the ``CLEARML_SNUG_WHITELIST`` env var.
Because the whitelist is an ordinary config key it flows through the 
agent's config-precedence layers (file < config vault < env < monitoring vault)
so an admin can push a whitelist a user cannot override.

The v1 schema is locked in ``clearml_agent/snug/whitelist.schema.json``.
Any change requires a schema version bump. The shipped default (in
``backend_api/config/default/agent.conf``) instruments the common LLM
providers (OpenAI, Anthropic, Gemini); ``rules: []`` is the empty-skeleton
behaviour (no rules, no header injection).
"""
import base64
import functools
import json

from clearml_agent.external.pyhocon import ConfigTree


# Empty skeleton matches the documented v1 default ("meter all hosts, no
# rules"). Built from a constant rather than parsed at import time so the
# fallback path is allocation-free.
_EMPTY_SKELETON = {
    "version": 1,
    "default_action": "meter",
    "rules": [],
}


def build_whitelist_env(session, extra_rules=None):
    # type: (object, object) -> str
    """Resolve ``agent.snug.whitelist`` from config and return it as a
    base64-encoded compact-JSON string for transport in the
    ``CLEARML_SNUG_WHITELIST`` env var.

    Falls back to the empty skeleton (meter-only, no injection) when the
    config is absent or not v1 - the same safe default, sourced from config.
    The shim applies its own per-rule field defaults, so a partial rule in
    config is fine.

    ``extra_rules`` (an iterable of rule dicts, e.g. an app profile's
    ``whitelist_contribution``) is the app-shipped BASE: the resolved config
    whitelist EXTENDS it. The profile rules therefore WIN on a host collision —
    they carry the per-host config the app needs to meter (e.g. the consumer-wire
    estimate predicates), so a stale user/admin rule for the same app host can't
    silently disable the app's metering. The config whitelist's non-colliding
    hosts are still added, and it still owns ``default_action``/``version`` (the
    admin-protected knobs). This lets an app carry the provider hosts it needs
    without shipping them in every agent's base whitelist.
    """
    raw = session.config.get("agent.snug.whitelist", None)
    content = _coerce_or_default(raw)
    if extra_rules:
        base = {
            "version": content.get("version", 1),
            "default_action": content.get("default_action", "meter"),
            "rules": [_to_plain(r) for r in extra_rules],
        }
        content = union_whitelist(base, content.get("rules", []))
    payload = json.dumps(content, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _coerce_or_default(raw):
    # type: (object) -> dict
    """Normalize a config value into a plain v1 whitelist dict, or the
    empty skeleton when it is missing/malformed."""
    if raw is None:
        # Copy so callers can't accidentally mutate the constant.
        return dict(_EMPTY_SKELETON)
    # HOCON hands us a ConfigTree (and a list of ConfigTree for `rules`);
    # normalize to plain dict/list so json.dumps emits clean JSON.
    data = _to_plain(raw)
    # Light shape validation. Don't enforce the full schema here - the
    # shim's parser is the source of truth - just catch obvious
    # misconfigurations.
    if not isinstance(data, dict) or data.get("version") != 1:
        print(
            "WARNING: clearml SNUG agent.snug.whitelist is not v1 format; "
            "using empty default"
        )
        return dict(_EMPTY_SKELETON)
    return data


def _to_plain(obj):
    # type: (object) -> object
    """Recursively unwrap a pyhocon ConfigTree (and nested trees/lists)
    into plain Python dict/list/scalars so json.dumps can serialize it.

    ConfigTree exposes ``as_plain_ordered_dict()`` for exactly this; the
    manual dict/list recursion covers plain inputs (e.g. test fakes)."""
    if hasattr(obj, "as_plain_ordered_dict"):
        return obj.as_plain_ordered_dict()
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def union_whitelist(base, additions_rules):
    # type: (object, object) -> dict
    """Admin-protected union of an admin/config-vault whitelist (``base``) with a
    user's added rules (``additions_rules``, the file whitelist's ``rules``).

    Keeps every ``base`` rule, then appends each addition whose host is not
    already a base host. ``version`` and ``default_action`` always come from
    ``base`` — a user can ADD hosts but can never override/remove an admin rule
    or change policy. Base rules are emitted FIRST, so even if an addition slips
    past the dedup (e.g. a concrete host a base WILDCARD also matches), the
    shim's first-match-wins keeps the admin rule authoritative on any overlap.

    Coverage is tested by EXACT, case-insensitive host — deliberately not
    wildcard-aware. Matching is the shim's job
    (``clearml_snug/shim/src/whitelist.rs``); duplicating its matcher here would
    only invite drift. A user rule a base wildcard covers therefore survives as a
    harmless, shadowed list entry instead of being dropped.

    Returns a plain v1 whitelist dict. An addition without a usable host is
    skipped; additions are deduped against the base hosts and each other
    case-insensitively."""
    base = _to_plain(base)
    if not isinstance(base, dict):
        base = dict(_EMPTY_SKELETON)
    base_rules = list(base.get("rules", []) or [])
    result_rules = list(base_rules)
    # Exact, case-insensitive base hosts: a user can't re-specify (and thus
    # alter) one of these, and we won't add the same host twice. Grows as
    # additions are accepted so intra-list duplicates are dropped too.
    seen = {str(r.get("host", "")).strip().lower() for r in base_rules}
    seen.discard("")
    for rule in (additions_rules or []):
        rule = _to_plain(rule)
        if not isinstance(rule, dict):
            continue
        host = str(rule.get("host", "")).strip()
        if not host:
            continue
        key = host.lower()
        if key in seen:
            continue  # already a base host, or an intra-list duplicate
        seen.add(key)
        result_rules.append(rule)
    return {
        "version": base.get("version", 1),
        "default_action": base.get("default_action", "meter"),
        "rules": result_rules,
    }


def resolve_effective_whitelist(file_whitelist, config_vault_overrides, priority_overrides):
    # type: (object, object, object) -> object
    """Compose the admin-protected EFFECTIVE whitelist: the admin (vault) base
    the user's file whitelist ADDS to.

    The admin base is all vault rules combined — the monitoring (priority) vault
    outranks the config vault (authoritative on a host collision), with the
    config vault's other hosts beneath it. The file whitelist then adds its hosts
    on top (admin-protected: a file host already in the admin base is dropped;
    ``default_action``/``version`` stay the admin's). Neither vault tier is a
    hard replace — both are a base the user extends.

    ``file_whitelist`` is the file-layer whitelist captured before the vault
    overrides collapsed it. ``config_vault_overrides`` / ``priority_overrides``
    are the agent's override-config lists (``Config._overrides_configs`` /
    ``_priority_overrides_configs``). Returns a plain v1 whitelist dict, or
    ``None`` when no vault defines a whitelist (the caller should then leave the
    resolved config — already the file's — untouched)."""
    mon_wl = _whitelist_of_overrides(priority_overrides)
    vault_wl = _whitelist_of_overrides(config_vault_overrides)
    if mon_wl is None and vault_wl is None:
        return None
    if mon_wl is None:
        admin_base = vault_wl
    elif vault_wl is None:
        admin_base = mon_wl
    else:
        admin_base = union_whitelist(mon_wl, _whitelist_rules(vault_wl))
    return union_whitelist(admin_base, _whitelist_rules(file_whitelist))


def _whitelist_of_overrides(override_configs):
    # type: (object) -> object
    """The ``agent.snug.whitelist`` carried by a list of override ConfigTrees (a
    vault or monitoring-vault layer), or ``None`` when absent/empty. Merges the
    trees the way the config override-resolvers do, then reads the key — so the
    vault layer's whitelist is seen independently of the file layer's."""
    if not override_configs:
        return None
    merged = functools.reduce(
        lambda cfg, override: ConfigTree.merge_configs(cfg, override, copy_trees=True),
        override_configs,
        ConfigTree(),
    )
    return merged.get("agent.snug.whitelist", None)


def _whitelist_rules(whitelist):
    # type: (object) -> list
    """The ``rules`` list of a whitelist dict/ConfigTree, or ``[]``."""
    if not whitelist:
        return []
    try:
        return list(whitelist.get("rules", []) or [])
    except Exception:
        return []
