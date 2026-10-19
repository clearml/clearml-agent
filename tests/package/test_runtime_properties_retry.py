"""
Tests for how ``Worker`` handles failed runtime-properties calls.

A failed call is cached as "not supported" so a server that rejects runtime properties is not
called on every poll. The ``--dynamic-gpus`` setup ignores that cache and asks the server on every
attempt, so a single transient error (e.g. a gateway 504) cannot leave the daemon stuck on
"Dynamic GPU allocation is not supported by your ClearML-server" for the life of the process.
"""
import json

import pytest
import requests

from clearml_agent.commands.worker import Worker
from clearml_agent.errors import APIError


class _FakeSession:
    api_version = "2.36"

    def check_min_api_version(self, min_api_version):
        return True


class _FakeMonitor:
    def setup_dynamic_gpu_cluster_report(self, **kwargs):
        pass


def _error_response(status_code, content):
    response = requests.Response()
    response.status_code = status_code
    response._content = content
    return response


def _gateway_timeout_error():
    """A non-JSON error page, e.g. from a load balancer in front of the apiserver."""
    return APIError(response=_error_response(status_code=504, content=b"<html>504 Gateway Time-out</html>"))


def _apiserver_error():
    """A JSON error returned by the ClearML apiserver itself."""
    body = {"meta": {"id": "trx-123", "trx": "trx-123", "result_code": 500, "result_subcode": 0,
                     "result_msg": "General data error"}, "data": {}}
    return APIError(response=_error_response(status_code=500, content=json.dumps(body).encode()))


class _FakeServer:
    """Stands in for Worker.get / Worker.post: raises ``error`` while it is set and counts calls."""
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self, endpoint, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return {"runtime_properties": [{"key": "available_gpus", "value": "0_0"}]}


def _make_worker(server):
    """Build a Worker instance with only the attributes the tested methods touch."""
    worker = Worker.__new__(Worker)
    worker._session = _FakeSession()
    worker.worker_id = "test-worker:dgpu0-7"
    worker._runtime_props_support = None
    worker.monitor = _FakeMonitor()
    worker.get = server
    worker.post = server
    return worker


def test_failure_is_cached_outside_dynamic_gpus_setup():
    """After a failure, get and set answer from the cache instead of calling the server on every poll."""
    server = _FakeServer(error=_gateway_timeout_error())
    worker = _make_worker(server=server)

    assert worker.get_runtime_properties() is None
    assert worker.set_runtime_properties(key="available_gpus", value="0_0") is False
    assert worker.get_runtime_properties() is None
    assert server.calls == 1


def test_dynamic_gpus_setup_asks_server_on_every_attempt():
    """The --dynamic-gpus setup recovers as soon as the server does, despite the cached failure."""
    server = _FakeServer(error=_gateway_timeout_error())
    worker = _make_worker(server=server)
    gpu_queues = [("queue-id", (1.0, 1.0))]

    with pytest.raises(ValueError):
        worker._setup_dynamic_gpus(gpu_queues=gpu_queues, gpu_indexes=[0])
    server.error = None
    available_gpus, _ = worker._setup_dynamic_gpus(gpu_queues=gpu_queues, gpu_indexes=[0])

    assert available_gpus == {"0": 0.0}


def test_failure_log_identifies_worker_and_request(capsys):
    """The warning carries the worker id and the apiserver request id, or none when the error came from elsewhere."""
    worker = _make_worker(server=_FakeServer(error=_apiserver_error()))
    worker.get_runtime_properties()
    worker = _make_worker(server=_FakeServer(error=_gateway_timeout_error()))
    worker.get_runtime_properties()

    out = capsys.readouterr().out
    assert "worker=test-worker:dgpu0-7, request id=trx-123" in out
    assert "General data error" in out
    assert "worker=test-worker:dgpu0-7, request id=None" in out
    assert "504 Gateway Time-out" in out
