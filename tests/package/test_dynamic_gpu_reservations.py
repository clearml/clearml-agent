"""
Tests for dynamic-GPU (fractional GPU) daemon-mode accounting in
``clearml_agent.commands.worker.Worker``.

Covers two bugs:

* Bug 1 (concurrency over-pull): in ``--dynamic-gpus`` mode the daemon gates
  concurrency purely on ``_dynamic_gpu_get_available()``, which reads the list of
  registered workers from the *server*. ``run_one_task()`` returns as soon as the
  task container starts (before the child registers as a worker), so without local
  bookkeeping the daemon re-polls, sees the GPU as free, and over-pulls many tasks
  onto a single GPU. The fix adds a local reservation ledger that is subtracted from
  the computed availability until the child registers (released then) or a TTL
  elapses. The relevant tests below fail on the pre-fix code (which returns the GPU
  as fully free while a launch is still coming up).

* Bug 2 (env injection, worker side): the fractional task container must receive
  ``CLEARML_AGENT_GPU_FRACTIONS=<fraction>`` so its ResourceMonitor reports the
  allocated fraction. ``Worker._docker_cmd_set_env`` performs the injection into the
  docker command; ``test_docker_cmd_set_env_*`` exercise it directly.
"""
import logging
from time import time

from clearml_agent.commands.worker import Worker

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Minimal fakes so we can exercise Worker methods without a real ClearML session/server.
# --------------------------------------------------------------------------------------
class _FakeWorkerEntry:
    def __init__(self, worker_id):
        self.id = worker_id


class _FakeGetAllResponse:
    def __init__(self, workers):
        self.workers = workers


class _FakeConfig:
    def __init__(self, overrides=None):
        self._overrides = overrides or {}

    def get(self, key, default=None):
        return self._overrides.get(key, default)


class _FakeSession:
    """Returns a fixed list of registered workers for every GetAll request."""
    def __init__(self, registered_worker_ids, worker_name):
        self._workers = [_FakeWorkerEntry(w) for w in (registered_worker_ids or [])]
        self.config = _FakeConfig({"agent.worker_name": worker_name})

    def send_api(self, request):  # noqa: ARG002 - request shape is irrelevant to the fake
        return _FakeGetAllResponse(self._workers)


def _make_worker(registered_worker_ids=None, ttl=600.0, worker_name="test-worker"):
    """Build a Worker instance with only the attributes the tested methods touch."""
    worker = Worker.__new__(Worker)
    worker._session = _FakeSession(registered_worker_ids, worker_name)
    # the daemon's own worker-id; excluded from the "our workers" scan
    worker.worker_id = "{}:daemon".format(worker_name)
    worker._dynamic_gpus_reservations = {}
    worker._dynamic_gpus_reservation_ttl = ttl
    return worker


# --------------------------------------------------------------------------------------
# Bug 1: reservation accounting
# --------------------------------------------------------------------------------------
def test_pending_reservation_reduces_availability_before_child_registers():
    """
    A 0.5 fraction was just launched but the child has not registered yet (server
    reports no workers). Availability on the single GPU must reflect the reservation
    (0.5 free), NOT report the GPU as fully free.

    Fails on pre-fix code: without the reservation ledger the GPU shows as {"0": 1}.
    """
    worker = _make_worker(registered_worker_ids=[])
    worker._reserve_dynamic_gpus(["0.5a"], [0.5])

    available_gpus, allocated_gpus = worker._dynamic_gpu_get_available([0])

    assert available_gpus == {"0": 0.5}
    assert allocated_gpus.get("0.5a") == 0.5


def test_two_half_reservations_fill_single_gpu_and_block_further_pulls():
    """
    floor(1 / 0.5) == 2: two 0.5 tasks fill the only GPU. While neither child has
    registered yet, availability must be empty so the daemon stops pulling (this is
    exactly the over-pull the fix prevents).

    Fails on pre-fix code: the GPU would still report as fully free.
    """
    worker = _make_worker(registered_worker_ids=[])
    worker._reserve_dynamic_gpus(["0.5a", "0.5b"], [0.5, 0.5])

    available_gpus, _ = worker._dynamic_gpu_get_available([0])

    assert available_gpus == {}


def test_reservation_released_once_child_registers_without_double_counting():
    """
    Once the child registers on the server (worker id ends with 'gpu0.5a'), the
    server accounting covers the fraction, so the local reservation must be dropped
    and must NOT be counted a second time (availability stays 0.5, not 0.0).
    """
    worker = _make_worker(registered_worker_ids=["test-worker:gpu0.5a"])
    worker._reserve_dynamic_gpus(["0.5a"], [0.5])

    available_gpus, allocated_gpus = worker._dynamic_gpu_get_available([0])

    assert available_gpus == {"0": 0.5}
    assert allocated_gpus.get("0.5a") == 0.5
    # reservation released now that the server reflects the child
    assert worker._dynamic_gpus_reservations == {}


def test_reservation_expires_after_ttl():
    """
    A launch that never registers must not hold its reservation forever: once the
    TTL elapses the reservation is dropped and the GPU becomes available again.
    """
    worker = _make_worker(registered_worker_ids=[], ttl=600.0)
    # stale reservation (reserved well beyond the TTL, child never registered)
    worker._dynamic_gpus_reservations["0.5a"] = ("0", 0.5, time() - 10_000)

    available_gpus, _ = worker._dynamic_gpu_get_available([0])

    assert available_gpus == {"0": 1}
    assert worker._dynamic_gpus_reservations == {}


def test_whole_gpu_reservation_blocks_second_whole_gpu_pull():
    """
    The race also affects whole-GPU allocations. Reserving whole GPU '0' must remove
    it from availability until the child registers (multi-GPU case: only GPU '1'
    remains free).
    """
    worker = _make_worker(registered_worker_ids=[])
    worker._reserve_dynamic_gpus(["0"], [1])

    available_gpus, allocated_gpus = worker._dynamic_gpu_get_available([0, 1])

    assert available_gpus == {"1": 1}
    assert allocated_gpus.get("0") == 1


def test_reserve_dynamic_gpus_records_index_fraction_and_timestamp():
    """_reserve_dynamic_gpus parses the GPU index out of the suffix and stores the fraction."""
    worker = _make_worker()
    before = time()
    worker._reserve_dynamic_gpus(["0.5a"], [0.5])
    after = time()

    assert set(worker._dynamic_gpus_reservations.keys()) == {"0.5a"}
    gpu_idx, fraction, reserved_ts = worker._dynamic_gpus_reservations["0.5a"]
    assert gpu_idx == "0"
    assert fraction == 0.5
    assert before <= reserved_ts <= after


# --------------------------------------------------------------------------------------
# Bug 2 (worker side): CLEARML_AGENT_GPU_FRACTIONS injection into the container command
# --------------------------------------------------------------------------------------
_BASE_DOCKER_CMD = [
    "docker", "run", "-t", "--gpus", "0",
    "nvidia/cuda:12.0-runtime", "bash", "-c", "clearml-agent execute --id abc123",
]


def test_docker_cmd_set_env_injects_before_image():
    """The env is injected right after 'run' and before the image / in-container command."""
    out = Worker._docker_cmd_set_env(_BASE_DOCKER_CMD, "CLEARML_AGENT_GPU_FRACTIONS", 0.5)

    assert "-e" in out
    e_index = out.index("-e")
    assert out[e_index + 1] == "CLEARML_AGENT_GPU_FRACTIONS=0.5"
    # inserted after the 'run' subcommand ...
    assert out.index("run") < e_index
    # ... and before the image name (so docker treats it as a run option)
    assert e_index < out.index("nvidia/cuda:12.0-runtime")


def test_docker_cmd_set_env_does_not_mutate_input():
    """The helper returns a new list; the caller's command is left untouched."""
    original = list(_BASE_DOCKER_CMD)
    _ = Worker._docker_cmd_set_env(_BASE_DOCKER_CMD, "CLEARML_AGENT_GPU_FRACTIONS", 0.5)
    assert _BASE_DOCKER_CMD == original


def test_docker_cmd_set_env_falls_back_when_no_run_token():
    """If there is no 'run' token, inject right after the executable rather than crash."""
    out = Worker._docker_cmd_set_env(["podman", "image:tag"], "CLEARML_AGENT_GPU_FRACTIONS", 0.25)
    assert out == ["podman", "-e", "CLEARML_AGENT_GPU_FRACTIONS=0.25", "image:tag"]
