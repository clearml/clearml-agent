"""
Tests for reporting a running task as pending again when its pod is replaced (k8s glue).

Regression coverage for DAG-14696: when a running task's pod was preempted and the job replaced it with a pod that
could not be scheduled, the task kept showing as running:
- the pending pods daemon enqueued it back into the k8s pending queue, but the task was still listed there from
  when it was launched, so the enqueue failed as already queued;
- the jobs cleanup counted the preempted pod as a failure and deleted the job, replacement pod included.
"""
import logging
import unittest
from unittest import mock

from clearml_agent.glue.k8s import K8sIntegration
from clearml_agent.glue.pending_pods_daemon import PendingPodsDaemon

PENDING_QUEUE = "pending-queue-id"
TASK = "task-id"


class _Response:
    def __init__(self, ok=True, data=None, result_msg=None):
        self.ok = ok
        self._data = data or {}
        self._result_msg = result_msg

    def json(self):
        return {"data": self._data, "meta": {"result_msg": self._result_msg}}


class _Session:
    def __init__(self, status="in_progress", already_queued=True):
        self.requests = []
        self._status = status
        self._queued = already_queued

    def send_request(self, service, action, json=None, **_):
        self.requests.append((service, action, json))
        if (service, action) == ("tasks", "get_all"):
            return _Response(data={"tasks": [{"status": self._status}]})
        if (service, action) == ("queues", "remove_task"):
            removed, self._queued = self._queued, False
            return _Response(ok=removed)
        if (service, action) == ("tasks", "enqueue"):
            # like the server: a task already in the queue cannot be added to it again
            return _Response(ok=not self._queued, result_msg="task is already queued")
        return _Response()

    def actions(self):
        return [(service, action) for service, action, _ in self.requests]


def _make_daemon(session):
    agent = mock.Mock(_session=session, k8s_pending_queue_id=PENDING_QUEUE, log=logging.getLogger("test"))
    return PendingPodsDaemon(polling_interval=1, agent=agent)


class UpdatePendingTaskMsgTests(unittest.TestCase):
    def test_running_task_still_in_the_pending_queue_is_enqueued(self):
        session = _Session(already_queued=True)
        _make_daemon(session)._update_pending_task_msg(TASK, "Unschedulable")

        self.assertEqual(
            [("tasks", "get_all"), ("queues", "remove_task"), ("tasks", "enqueue"), ("tasks", "update")],
            session.actions()
        )
        enqueue = session.requests[2][2]
        self.assertEqual(PENDING_QUEUE, enqueue["queue"])
        self.assertTrue(enqueue["force"])
        # recognized as parked in the pending queue by the stale tasks cleanups
        self.assertEqual("k8s pending scheduler", enqueue["status_reason"])

    def test_running_task_not_in_the_pending_queue_is_enqueued(self):
        session = _Session(already_queued=False)
        daemon = _make_daemon(session)
        with mock.patch.object(daemon.log, "warning") as warning:
            daemon._update_pending_task_msg(TASK, "Unschedulable")

        self.assertIn(("tasks", "enqueue"), session.actions())
        warning.assert_not_called()

    def test_task_that_is_not_running_is_not_enqueued(self):
        session = _Session(status="queued")
        _make_daemon(session)._update_pending_task_msg(TASK, "Unschedulable")

        self.assertEqual([("tasks", "get_all"), ("tasks", "update")], session.actions())


def _pod(job="clearml-id-a", deleted=False, conditions=None):
    pod = {
        "metadata": {"name": job + "-xyz", "labels": {"job-name": job}},
        "status": {"phase": "Failed", "conditions": conditions or []},
    }
    if deleted:
        pod["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    return pod


def _delete_jobs_with_failed_pods(pods):
    glue = object.__new__(K8sIntegration)
    glue.using_jobs = True
    glue.log = logging.getLogger("test")
    glue.get_pods_for_jobs = mock.Mock(return_value=pods)
    glue.get_jobs_info = mock.Mock(return_value={})
    glue._delete_jobs_by_names = mock.Mock(side_effect=lambda names, **_: names)
    with mock.patch("clearml_agent.glue.k8s.ENV_COMPLETED_POD_DELETION_DELAY_MINUTES") as delay:
        delay.get.return_value = 0
        return sorted(glue._delete_completed_or_failed_pods(namespace="ns"))


class DeleteJobsWithFailedPodsTests(unittest.TestCase):
    def test_job_of_a_failed_pod_is_deleted(self):
        self.assertEqual(["clearml-id-a"], _delete_jobs_with_failed_pods([_pod()]))

    def test_job_of_a_deleted_pod_is_kept(self):
        self.assertEqual([], _delete_jobs_with_failed_pods([_pod(deleted=True)]))

    def test_job_of_an_evicted_pod_is_kept(self):
        disrupted = [{"type": "DisruptionTarget", "status": "True", "reason": "TerminationByKubelet"}]
        self.assertEqual([], _delete_jobs_with_failed_pods([_pod(conditions=disrupted)]))

    def test_only_jobs_of_failed_pods_are_deleted(self):
        pods = [_pod("clearml-id-a"), _pod("clearml-id-b", deleted=True)]
        self.assertEqual(["clearml-id-a"], _delete_jobs_with_failed_pods(pods))


if __name__ == "__main__":
    unittest.main()
