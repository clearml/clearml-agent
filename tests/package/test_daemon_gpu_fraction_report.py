"""
Server-facing contract for the dynamic-GPU "daemon-half" fix.

On a 1-GPU box running the dynamic-GPU poller there are two workers: the poller DAEMON
(a manager that consumes no GPU) and the per-task SLOT container (the real consumer). The
daemon used to report gpu_fraction=1.0 for its physical GPU, so the dashboard counted
daemon(1.0) + slot(0.5) = 1.5 GPUs in use instead of 0.5.

The fix makes the daemon's ResourceMonitor report an EXPLICIT gpu_fraction of 0.0 (one per
visible GPU) while keeping gpu_usage/temperature. Explicit 0.0 matters: the allegro-engine
server does `if gpu_usage and not gpu_fraction: gpu_fraction = [1.0]*len(gpu_usage)`, so an
omitted/empty fraction would be defaulted straight back to 1.0 (the bug).

These tests drive ResourceMonitor._machine_stats() with a fake GPU query and assert the
emitted gpu_fraction_<i> values.
"""
import logging

from clearml_agent.helper import resource_monitor as rm
from clearml_agent.helper.resource_monitor import ResourceMonitor, GpuFractionsHandler
from clearml_agent.definitions import ENV_GPU_FRACTIONS

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)

_GPU_FRACTIONS_VAR = ENV_GPU_FRACTIONS.vars[0]


class _FakeConfig:
    def get(self, key, default=None):
        return default


class _FakeSession:
    feature_set = "advanced"  # fractional GPU support is enterprise (non-basic)
    config = _FakeConfig()


class _FakeGpu(dict):
    """gpustat GPUStat is dict-like (g["memory.used"]) with an .index attribute."""
    @property
    def index(self):
        return self.get("_index", 0)


class _FakeQuery:
    def __init__(self, gpus):
        self.gpus = gpus


class _FakeGpustat:
    def __init__(self, gpus):
        self._gpus = gpus

    def new_query(self):
        return _FakeQuery(self._gpus)


def _make_gpu(index=0):
    return _FakeGpu({
        "temperature.gpu": 45,
        "utilization.gpu": 10,
        "memory.used": 1000,
        "memory.total": 16000,
        "_index": index,
    })


def _build_monitor(monkeypatch, gpu_fractions, num_gpus=1):
    # the daemon box has a physical GPU; avoid touching real hardware
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["Tesla T4"] * num_gpus))
    monitor = ResourceMonitor(session=_FakeSession(), worker_id="w", gpu_fractions=gpu_fractions)
    monitor._gpustat = _FakeGpustat([_make_gpu(i) for i in range(num_gpus)])
    monitor._active_gpus = None  # report all GPUs, no filtering
    return monitor


def test_dynamic_daemon_reports_explicit_zero_but_keeps_usage(monkeypatch):
    """Daemon (gpu_fractions='0') emits gpu_fraction_0 == 0.0, and still reports util/temperature."""
    monkeypatch.delenv(_GPU_FRACTIONS_VAR, raising=False)
    monitor = _build_monitor(monkeypatch, gpu_fractions="0")

    stats = monitor._machine_stats()

    assert stats["gpu_fraction_0"] == 0.0
    # gpu_usage must remain non-empty (the server only defaults the fraction when gpu_usage is present)
    assert stats["gpu_utilization_0"] == 10
    assert stats["gpu_temperature_0"] == 45


def test_slot_reports_env_injected_fraction(monkeypatch):
    """A slot container (no override) reads its injected CLEARML_AGENT_GPU_FRACTIONS=0.5 -> 0.5."""
    monkeypatch.setenv(_GPU_FRACTIONS_VAR, "0.5")
    monitor = _build_monitor(monkeypatch, gpu_fractions=None)

    stats = monitor._machine_stats()

    assert stats["gpu_fraction_0"] == 0.5


def test_plain_daemon_still_reports_full_gpu(monkeypatch):
    """Non-dynamic (plain) daemon: no override and no env -> still reports 1.0 (no regression)."""
    monkeypatch.delenv(_GPU_FRACTIONS_VAR, raising=False)
    monitor = _build_monitor(monkeypatch, gpu_fractions=None)

    stats = monitor._machine_stats()

    assert stats["gpu_fraction_0"] == 1.0


def test_multi_gpu_daemon_reports_zero_for_each_gpu(monkeypatch):
    """With N visible GPUs the daemon reports 0.0 for every one of them (one per visible GPU)."""
    monkeypatch.delenv(_GPU_FRACTIONS_VAR, raising=False)
    monitor = _build_monitor(monkeypatch, gpu_fractions="0", num_gpus=2)

    stats = monitor._machine_stats()

    assert stats["gpu_fraction_0"] == 0.0
    assert stats["gpu_fraction_1"] == 0.0


def test_whole_gpu_slot_env_value_roundtrips_to_full_fraction(monkeypatch):
    """
    Constraint #4: whole-GPU dynamic slots report their fraction explicitly. The worker builds
    the env value as ",".join(str(f) for f in fractions); for whole GPUs fractions == [1] (or
    [1, 1]). Confirm that value decodes back to 1.0 per GPU (so a whole-GPU slot reports 1.0).
    """
    whole_gpu_fractions = [1, 1]
    env_value = ",".join(str(f) for f in whole_gpu_fractions)  # "1,1" — mirrors the worker
    assert env_value == "1,1"

    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["Tesla T4", "Tesla T4"]))
    assert GpuFractionsHandler(gpu_fractions=env_value).fractions == [1.0, 1.0]
