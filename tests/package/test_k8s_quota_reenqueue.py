"""
Tests for re-enqueueing tasks whose pod is rejected by a Kubernetes ResourceQuota (k8s glue).

Regression coverage for DAG-20699: when a namespace ResourceQuota was used up, kubectl apply failed with
"... is forbidden: exceeded quota: ..." and the task was set to failed. It should wait for quota instead, like the
in-tree controllers do:
- the quota rejection re-enqueues the task, other kubectl errors still fail it;
- the re-enqueued task is removed from the k8s pending queue it was pushed into before applying;
- lower priority queues wait until the task is launched (strict priority), and it is retried once per polling cycle;
- the wait is logged to the task once, not on every retry.
"""
import logging
import re
import unittest
from unittest import mock

from clearml_agent.glue.errors import ReenqueueTaskException
from clearml_agent.glue.k8s import K8sIntegration

PENDING_QUEUE = "pending-queue-id"
TASK = "task-id"

QUOTA_ERROR = (
    'Error from server (Forbidden): error when creating "/tmp/clearml_k8stmpl_k3x9q2.yml": pods '
    '"clearml-id-7e3d1c9a5b2f48e6a0c4d8b1f2a6e9c3" is forbidden: exceeded quota: team-quota, '
    'requested: requests.nvidia.com/gpu=1, used: requests.nvidia.com/gpu=5, limited: requests.nvidia.com/gpu=5\n'
)
JOB_QUOTA_ERROR = (
    'Error from server (Forbidden): error when creating "/tmp/clearml_k8stmpl_x.yml": jobs.batch "clearml-id-a" '
    'is forbidden: exceeded quota: jobs, requested: count/jobs.batch=1, used: count/jobs.batch=2, '
    'limited: count/jobs.batch=2\n'
)
# the pod spec does not satisfy the quota: retrying cannot help
SPEC_QUOTA_ERROR = (
    'Error from server (Forbidden): error when creating "/tmp/clearml_k8stmpl_y.yml": pods "clearml-id-a" '
    'is forbidden: failed quota: compute: must specify limits.cpu for: clearml-id-a\n'
)
LIMIT_RANGE_ERROR = (
    'Error from server (Forbidden): error when creating "/tmp/clearml_k8stmpl_z.yml": pods "clearml-id-a" '
    'is forbidden: maximum cpu usage per Container is 2, but limit is 4\n'
)


class _Response:
    def __init__(self, ok=True, data=None):
        self.ok = ok
        self.text = "" if ok else "error"
        self._data = data or {}

    def json(self):
        return {"data": self._data}


def _make_glue(task_status="queued", ok=True):
    glue = object.__new__(K8sIntegration)
    glue.log = logging.getLogger("test")
    glue.k8s_pending_queue_id = PENDING_QUEUE
    glue.ignore_kubectl_errors_re = None
    glue._reenqueued_tasks_msgs = {}
    glue._session = mock.Mock()
    glue._session.send_request.side_effect = lambda service, action, **_: (
        _Response(data={"tasks": [{"status": task_status}]}) if action == "get_all" else _Response(ok=ok)
    )
    glue.send_logs = mock.Mock()
    glue.resource_applied = mock.Mock()
    glue.set_task_info = mock.Mock()
    glue._set_task_failed_while_applying = mock.Mock()
    return glue


def _apply(glue, error):
    glue._kubectl_apply = mock.Mock(return_value=("", error, "clearml-id-a"))
    glue.apply_template_and_handle_result(
        pod_name="clearml-id-a", clearml_conf_create_script=[], labels=[], queue="queue", task_id=TASK,
        namespace="ns", template={}, docker_image="image", docker_args=None, docker_bash=None,
        session=mock.Mock(), task_session=None, queue_name="queue", task_data={}, ports_mode=False, pod_count=0,
    )


class ApplyResultTests(unittest.TestCase):
    def test_quota_exceeded_reenqueues(self):
        glue = _make_glue()
        with self.assertRaises(ReenqueueTaskException) as ctx:
            _apply(glue, QUOTA_ERROR)
        self.assertIn(
            "team-quota, requested: requests.nvidia.com/gpu=1, used: requests.nvidia.com/gpu=5, "
            "limited: requests.nvidia.com/gpu=5",
            str(ctx.exception)
        )
        glue._set_task_failed_while_applying.assert_not_called()
        glue.resource_applied.assert_not_called()

    def test_job_quota_exceeded_reenqueues(self):
        with self.assertRaises(ReenqueueTaskException):
            _apply(_make_glue(), JOB_QUOTA_ERROR)

    def test_quota_exceeded_takes_precedence_over_ignored_errors(self):
        glue = _make_glue()
        glue.ignore_kubectl_errors_re = re.compile(".*")
        with self.assertRaises(ReenqueueTaskException):
            _apply(glue, QUOTA_ERROR)
        glue.resource_applied.assert_not_called()

    def test_other_errors_fail_the_task(self):
        for error in (SPEC_QUOTA_ERROR, LIMIT_RANGE_ERROR, "error: unable to read template\n"):
            with self.subTest(error=error):
                glue = _make_glue()
                _apply(glue, error)
                glue._set_task_failed_while_applying.assert_called_once()
                glue.resource_applied.assert_not_called()

    def test_success(self):
        glue = _make_glue()
        _apply(glue, "")
        glue._set_task_failed_while_applying.assert_not_called()
        glue.resource_applied.assert_called_once()


class ReEnqueueTaskTests(unittest.TestCase):
    def test_task_is_removed_from_the_pending_queue_before_it_is_enqueued(self):
        glue = _make_glue()
        calls = mock.Mock()
        calls.attach_mock(glue._session.api_client.queues.remove_task, "remove_task")
        calls.attach_mock(glue._session.send_request, "send_request")

        self.assertTrue(glue._re_enqueue_task(queue="queue", task_id=TASK))

        self.assertEqual(
            [
                ("remove_task", {"task": TASK, "queue": PENDING_QUEUE}),
                ("send_request", "add_task"),
                ("send_request", "move_task_to_front"),
            ],
            [
                (name, kwargs if name == "remove_task" else kwargs["action"])
                for name, _, kwargs in calls.mock_calls
                if kwargs.get("action") != "get_all"
            ]
        )

    def test_task_not_in_the_pending_queue_is_still_enqueued(self):
        glue = _make_glue()
        glue._session.api_client.queues.remove_task.side_effect = Exception("task not in queue")

        self.assertTrue(glue._re_enqueue_task(queue="queue", task_id=TASK))
        self.assertEqual(["add_task", "move_task_to_front"], _sent_actions(glue))

    def test_task_that_cannot_be_enqueued_is_failed(self):
        glue = _make_glue(ok=False)

        self.assertFalse(glue._re_enqueue_task(queue="queue", task_id=TASK))
        self.assertEqual(["add_task"], _sent_actions(glue))
        self.assertEqual("failed", glue._session.get.call_args.kwargs["action"])

    def test_task_aborted_meanwhile_is_not_enqueued_back(self):
        # e.g. aborted by a user while its pods were being created: the abort must stick
        for status in ("stopped", "created", "failed"):
            with self.subTest(status=status):
                glue = _make_glue(task_status=status)
                self.assertFalse(glue._re_enqueue_task(queue="queue", task_id=TASK))
                self.assertEqual([], _sent_actions(glue))
                glue._session.api_client.queues.remove_task.assert_called_once_with(task=TASK, queue=PENDING_QUEUE)
                glue._session.get.assert_not_called()


def _sent_actions(glue):
    """ API requests sent (status reads aside) """
    return [
        c.kwargs.get("action") or c.args[1] for c in glue._session.send_request.call_args_list
        if (c.kwargs.get("action") or c.args[1]) != "get_all"
    ]


class _StopLoop(Exception):
    pass


FOREVER = 1000


def _run_tasks_loop(queued_tasks, quota_rejections, max_sleeps=20):
    """
    Run run_tasks_loop until a polling cycle finds nothing to launch (or after max_sleeps sleeps).
    queued_tasks: queue -> list of task ids, popped in order (a re-enqueued task goes back to the front)
    quota_rejections: task id -> number of launch attempts that raise ReenqueueTaskException
    Returns the glue, the task ids launch was attempted for, and the events service mock.
    """
    quota_rejections = dict(quota_rejections)
    never_launched = {t for t, count in quota_rejections.items() if count >= FOREVER}
    glue = _make_glue()
    glue._session.config = {"agent.reload_config": False}
    glue._polling_interval = 0
    glue.worker_id = "worker"
    glue.namespace = "ns"
    glue.max_pods_limit = None
    glue._impersonate_as_task_owner = False
    glue._get_used_pods = mock.Mock(return_value=(0, set()))
    glue._cleanup_old_pods = mock.Mock()
    glue.check_if_suspended = mock.Mock(return_value=False)
    glue.check_if_schedulable = mock.Mock(return_value=True)
    glue.report_monitor = mock.Mock()
    events_service = mock.Mock()
    glue.get_service = mock.Mock(return_value=events_service)

    queued_tasks = {queue: list(tasks) for queue, tasks in queued_tasks.items()}

    def get_next_task(queue, **_):
        tasks = queued_tasks.get(queue)
        return {"entry": {"task": tasks.pop(0)}} if tasks else None

    def re_enqueue_task(queue, task_id):
        queued_tasks[queue].insert(0, task_id)
        return True

    attempts = []

    def run_one_task(queue, task_id, *_, **__):
        attempts.append(task_id)
        if quota_rejections.get(task_id, 0) > 0:
            quota_rejections[task_id] -= 1
            raise ReenqueueTaskException("Kubernetes ResourceQuota exceeded (team-quota)")

    sleeps = []

    def stop_when_idle(_):
        sleeps.append(1)
        if len(sleeps) >= max_sleeps or not any(
            t not in never_launched for tasks in queued_tasks.values() for t in tasks
        ):
            raise _StopLoop()

    glue._get_next_task = mock.Mock(side_effect=get_next_task)
    glue._re_enqueue_task = mock.Mock(side_effect=re_enqueue_task)
    glue.run_one_task = mock.Mock(side_effect=run_one_task)

    with mock.patch("clearml_agent.glue.k8s.sleep", side_effect=stop_when_idle):
        try:
            glue.run_tasks_loop(queues=["high", "low"], worker_params=None)
        except _StopLoop:
            pass
    return glue, attempts, events_service


class RunTasksLoopTests(unittest.TestCase):
    def test_lower_priority_queues_wait_for_a_task_waiting_for_quota(self):
        _, attempts, _ = _run_tasks_loop(
            {"high": ["gpu-task"], "low": ["cpu-1"]}, {"gpu-task": FOREVER}, max_sleeps=5
        )
        # one attempt per polling cycle, the low priority queue is not polled meanwhile
        self.assertEqual(["gpu-task"] * 5, attempts)

    def test_lower_priority_queues_run_once_the_task_is_launched(self):
        _, attempts, _ = _run_tasks_loop({"high": ["gpu-task"], "low": ["cpu-1"]}, {"gpu-task": 2})
        self.assertEqual(["gpu-task", "gpu-task", "gpu-task", "cpu-1"], attempts)

    def test_tasks_behind_it_in_its_queue_wait_for_it(self):
        _, attempts, _ = _run_tasks_loop({"high": ["gpu-task", "gpu-task-2"], "low": []}, {"gpu-task": 2})
        self.assertEqual(["gpu-task", "gpu-task", "gpu-task", "gpu-task-2"], attempts)

    def test_waiting_task_is_logged_once(self):
        glue, attempts, events_service = _run_tasks_loop(
            {"high": ["gpu-task"], "low": ["cpu-1"]}, {"gpu-task": FOREVER}, max_sleeps=5
        )

        self.assertEqual(["gpu-task"] * 5, attempts)
        info_logs = [c for c in glue.send_logs.call_args_list if c.args[0] == "gpu-task"]
        self.assertEqual(1, len(info_logs))
        self.assertIn("ResourceQuota exceeded", info_logs[0].args[1][0])
        pulled_logs = [c.kwargs["task_id"] for c in events_service.send_log_events.call_args_list]
        self.assertEqual(["gpu-task"], pulled_logs)

    def test_new_reason_is_logged(self):
        glue = _make_glue()
        glue._report_reenqueued_task(TASK, "quota (used: 5)")
        glue._report_reenqueued_task(TASK, "quota (used: 5)")
        glue._report_reenqueued_task(TASK, "quota (used: 4)")
        self.assertEqual(2, glue.send_logs.call_count)

    def test_task_is_launched_once_quota_is_available(self):
        glue, attempts, events_service = _run_tasks_loop({"high": ["gpu-task"], "low": []}, {"gpu-task": 2})

        self.assertEqual(["gpu-task"] * 3, attempts)
        self.assertEqual(1, glue.send_logs.call_count)
        self.assertEqual(1, events_service.send_log_events.call_count)
        # launched: a later wait (e.g. after being re-enqueued by a user) is logged again
        self.assertEqual({}, glue._reenqueued_tasks_msgs)


if __name__ == "__main__":
    unittest.main()
