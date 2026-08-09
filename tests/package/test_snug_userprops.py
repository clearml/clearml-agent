"""Tests for the SNUG call-history / whitelist runtime-control helpers in
``clearml_agent.helper.snug``:

* the User-Property NAME + VALUE constants — the cross-language contract with
  the Rust poll thread (``clearml_snug/reporter/src/poll.rs`` ``PROP_*`` and
  ``clearml_snug/messages/src/lib.rs`` ``CallHistoryMode``); and
* ``get_task_user_property`` — reads the task's User Properties (hyperparams
  section ``properties``) via ``tasks.get_hyper_params``, the same channel the
  in-process reporter polls.
"""

from clearml_agent.helper import snug


# --- Cross-language name/value contract -----------------------------------


def test_userprop_name_constants():
    # These strings are the wire contract with poll.rs PROP_*; keep in lockstep.
    assert snug.SNUG_USERPROP_CALL_HISTORY == "_snug_call_history"
    assert snug.SNUG_USERPROP_SECTION == "properties"


def test_call_history_value_constants_match_rust_serde():
    # Must match the Rust CallHistoryMode serde lowercase strings.
    assert snug.SNUG_CALL_HISTORY_OFF == "off"
    assert snug.SNUG_CALL_HISTORY_COLLECT == "collect"
    assert snug.SNUG_CALL_HISTORY_DUMP == "dump"
    assert snug.SNUG_CALL_HISTORY_CONTINUOUS == "continuous"
    assert snug.SNUG_CALL_HISTORY_MODES == ("off", "collect", "dump", "continuous")


def test_userprop_whitelist_name_constant():
    # Cross-language contract with poll.rs PROP_WHITELIST; keep in lockstep.
    assert snug.SNUG_USERPROP_WHITELIST == "_snug_whitelist"


# --- Read helper ----------------------------------------------------------


class _FakeSessionWithProps:
    """Returns a get_hyper_params-shaped response so ``get_task_user_property``
    can be exercised. Carries an initial {name: value} map for the "properties"
    section."""

    def __init__(self, properties=None):
        self._properties = dict(properties or {})

    def get(self, service, action, tasks=None, **kw):
        hyperparams = [
            {"section": "properties", "name": n, "value": v}
            for n, v in self._properties.items()
        ]
        return {"params": [{"hyperparams": hyperparams}]}


def test_get_task_user_property_reads_value_or_none():
    s = _FakeSessionWithProps({"_snug_whitelist": "api.foo.com"})
    assert snug.get_task_user_property(s, "t", "_snug_whitelist") == "api.foo.com"
    # absent property → None
    s2 = _FakeSessionWithProps({})
    assert snug.get_task_user_property(s2, "t", "_snug_whitelist") is None


def test_get_task_user_property_returns_none_on_error():
    class _Boom:
        def get(self, *a, **k):
            raise RuntimeError("backend down")

    assert snug.get_task_user_property(_Boom(), "t", "_snug_whitelist") is None
