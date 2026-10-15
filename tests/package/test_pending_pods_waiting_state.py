"""
Tests for PendingPodsDaemon._get_waiting_state.

With init containers or native sidecars in the pod, the main container only reports PodInitializing while an
init/sidecar container is stuck pulling its image: the eviction reason must still be picked up, or the Task stays
pending forever.
"""
import unittest

from clearml_agent.glue.pending_pods_daemon import PendingPodsDaemon


def _waiting(reason):
    return {"state": {"waiting": {"reason": reason, "message": "details"}}}


def _pod(init_statuses=None, statuses=None):
    status = {}
    if init_statuses is not None:
        status["initContainerStatuses"] = init_statuses
    if statuses is not None:
        status["containerStatuses"] = statuses
    return {"status": status}


class TestGetWaitingState(unittest.TestCase):
    def test_main_container_waiting(self):
        pod = _pod(statuses=[_waiting("ContainerCreating")])
        self.assertEqual(PendingPodsDaemon._get_waiting_state(pod)["reason"], "ContainerCreating")

    def test_no_statuses(self):
        self.assertIsNone(PendingPodsDaemon._get_waiting_state({"status": {}}))
        self.assertIsNone(PendingPodsDaemon._get_waiting_state({}))

    def test_sidecar_eviction_reason_wins_over_main_pod_initializing(self):
        pod = _pod(
            init_statuses=[{"state": {"running": {}}}, _waiting("ImagePullBackOff")],
            statuses=[_waiting("PodInitializing")],
        )
        self.assertEqual(PendingPodsDaemon._get_waiting_state(pod)["reason"], "ImagePullBackOff")

    def test_non_eviction_init_reason_keeps_main_container_state(self):
        pod = _pod(init_statuses=[_waiting("ContainerCreating")], statuses=[_waiting("PodInitializing")])
        self.assertEqual(PendingPodsDaemon._get_waiting_state(pod)["reason"], "PodInitializing")

    def test_eviction_reason_in_second_main_container(self):
        pod = _pod(statuses=[_waiting("ContainerCreating"), _waiting("InvalidImageName")])
        self.assertEqual(PendingPodsDaemon._get_waiting_state(pod)["reason"], "InvalidImageName")


if __name__ == "__main__":
    unittest.main()
