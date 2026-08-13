"""
Bug 2 (reporting side): the per-task fractional container's ResourceMonitor reports
``gpu_fraction_<i>`` from ``GpuFractionsHandler``, which reads the allocated fraction
from the ``CLEARML_AGENT_GPU_FRACTIONS`` env var and DEFAULTS to 1.0 when it is unset.

These tests pin the contract the worker-side fix relies on:
  * env unset  -> fraction reported as 1.0 (the observed bug when the env is not injected)
  * env "0.5"  -> fraction reported as 0.5 (what the fix injects)

They also cover the T4 edge case noted in the bug report: a GPU whose name is not in
``_gpu_name_to_memory_gb`` yields ``_total_memory_gb == [0]``, which must NOT prevent the
plain comma-separated-float fractions path from reporting the injected value.
"""
import logging

from clearml_agent.definitions import ENV_GPU_FRACTIONS
from clearml_agent.helper.resource_monitor import GpuFractionsHandler

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)

_GPU_FRACTIONS_VAR = ENV_GPU_FRACTIONS.vars[0]  # "CLEARML_AGENT_GPU_FRACTIONS"


def _handler_with_gpu_names(monkeypatch, names):
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: names))
    return GpuFractionsHandler()


def test_env_var_name_is_stable():
    """The worker injects using ENV_GPU_FRACTIONS.vars[0]; keep that name pinned."""
    assert _GPU_FRACTIONS_VAR == "CLEARML_AGENT_GPU_FRACTIONS"


def test_fraction_defaults_to_one_when_env_unset(monkeypatch):
    """Reproduces the reported symptom: without the env var the container reports a full GPU."""
    monkeypatch.delenv(_GPU_FRACTIONS_VAR, raising=False)
    handler = _handler_with_gpu_names(monkeypatch, ["Tesla T4"])
    assert handler.fractions == [1.0]


def test_injected_fraction_is_reported(monkeypatch):
    """With the env var injected by the fix, the handler reports the allocated 0.5 fraction."""
    monkeypatch.setenv(_GPU_FRACTIONS_VAR, "0.5")
    handler = _handler_with_gpu_names(monkeypatch, ["Tesla T4"])
    assert handler.fractions == [0.5]


def test_t4_zero_total_memory_does_not_block_float_fractions(monkeypatch):
    """
    A T4 is absent from _gpu_name_to_memory_gb, so _total_memory_gb == [0]. This must not
    short-circuit the plain float-fraction path (that path does not depend on total memory).
    """
    monkeypatch.setenv(_GPU_FRACTIONS_VAR, "0.5")
    handler = _handler_with_gpu_names(monkeypatch, ["Tesla T4"])
    assert handler._total_memory_gb == [0]
    assert handler.fractions == [0.5]


# --------------------------------------------------------------------------------------
# Explicit gpu_fractions override (used by the dynamic-GPU manager daemon to report 0.0
# without mutating the process environment).
# --------------------------------------------------------------------------------------
def test_explicit_zero_override_reports_zero(monkeypatch):
    """
    "0" is an explicit, non-empty value: it must parse to [0.0] (a real zero), NOT be treated
    as falsy and fall back to [1.0]. This is what makes the manager daemon report 0 GPUs.
    """
    monkeypatch.delenv(_GPU_FRACTIONS_VAR, raising=False)
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["Tesla T4"]))
    assert GpuFractionsHandler(gpu_fractions="0").fractions == [0.0]


def test_override_takes_precedence_over_env(monkeypatch):
    """The daemon's explicit '0' must win even if CLEARML_AGENT_GPU_FRACTIONS happens to be set."""
    monkeypatch.setenv(_GPU_FRACTIONS_VAR, "0.5")
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["Tesla T4"]))
    assert GpuFractionsHandler(gpu_fractions="0").fractions == [0.0]


def test_none_override_falls_back_to_env(monkeypatch):
    """With override=None (the slot containers), the value comes from the env var as before."""
    monkeypatch.setenv(_GPU_FRACTIONS_VAR, "0.5")
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["Tesla T4"]))
    assert GpuFractionsHandler(gpu_fractions=None).fractions == [0.5]


def test_whole_gpu_multi_value_override(monkeypatch):
    """Whole-GPU dynamic slots report 1.0 per GPU; "1,1" must decode to [1.0, 1.0]."""
    monkeypatch.delenv(_GPU_FRACTIONS_VAR, raising=False)
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["Tesla T4", "Tesla T4"]))
    assert GpuFractionsHandler(gpu_fractions="1,1").fractions == [1.0, 1.0]
