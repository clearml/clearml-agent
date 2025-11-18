from time import sleep
from typing import Dict, List, Optional

from clearml_agent.backend_api.session import Request
from clearml_agent.glue.utilities import get_bash_output

from .daemon import K8sDaemon
from .utilities import get_path
from .errors import GetPodsError
from .definitions import ENV_POD_MONITOR_DISABLE_ENQUEUE_ON_PREEMPTION


class PendingPodsDaemon(K8sDaemon):
    def __init__(self, polling_interval: float, agent):
        super(PendingPodsDaemon, self).__init__(agent=agent)
        self._polling_interval = polling_interval
        self._last_tasks_msgs = {}  # last msg updated for every task

    def get_pods(self, pod_name=None, debug_msg="Detecting pending pods: {cmd}"):
        filters = ["status.phase=Pending"]
        if pod_name:
            filters.append(f"metadata.name={pod_name}")

        if self._agent.using_jobs:
            return self._agent.get_pods_for_jobs(
                job_condition="status.active=1", pod_filters=filters, debug_msg=debug_msg
            )
        return self._agent.get_pods(filters=filters, debug_msg=debug_msg)

    def _get_pod_name(self, pod: dict):
        return get_path(pod, "metadata", "name")

    def _get_k8s_resource_name(self, pod: dict):
        if self._agent.using_jobs:
            return get_path(pod, "metadata", "labels", "job-name")
        return get_path(pod, "metadata", "name")

    def _get_task_id(self, pod: dict):
        prefix, _, value = self._get_k8s_resource_name(pod).rpartition('-')
        if len(value) > 4:
            return value
        # we assume this is a multi-node rank x (>0) pod
        return prefix.rpartition('-')[-1] or value

    @staticmethod
    def _get_k8s_resource_namespace(pod: dict) -> Optional[str]:
        return pod.get('metadata', {}).get('namespace', None)

    def target(self):
        """
            Handle pending objects (pods or jobs, depending on the agent mode).
            - Delete any pending objects that are not expected to recover
            - Delete any pending objects for whom the associated task was aborted
        """
        while True:
            # noinspection PyBroadException
            try:
                # Get pods (standalone pods if we're in pods mode, or pods associated to jobs if we're in jobs mode)
                pods = self.get_pods()
                if pods is None:
                    raise GetPodsError()

                task_id_to_pod = dict()

                for pod in pods:
                    pod_name = self._get_pod_name(pod)
                    if not pod_name:
                        continue

                    task_id = self._get_task_id(pod)
                    if not task_id:
                        continue

                    namespace = self._get_k8s_resource_namespace(pod)
                    if not namespace:
                        continue

                    updated_pod = self.get_pods(pod_name=pod_name, debug_msg="Refreshing pod information: {cmd}")
                    if not updated_pod:
                        continue
                    pod = updated_pod[0]

                    task_id_to_pod[task_id] = pod

                    msg = None
                    tags = []

                    waiting = get_path(pod, 'status', 'containerStatuses', 0, 'state', 'waiting')
                    if not waiting:
                        condition = get_path(pod, 'status', 'conditions', 0)
                        if condition:
                            reason = condition.get('reason')
                            if reason == 'Unschedulable':
                                message = condition.get('message')
                                msg = reason + (" ({})".format(message) if message else "")
                    else:
                        reason = waiting.get("reason", None)
                        message = waiting.get("message", None)

                        msg = reason + (" ({})".format(message) if message else "")

                        if reason == 'ImagePullBackOff':
                            self.delete_k8s_resource(k8s_resource=pod, msg=reason)
                            try:
                                self._session.api_client.tasks.failed(
                                    task=task_id,
                                    status_reason="K8S glue error: {}".format(msg),
                                    status_message="Changed by K8S glue",
                                    force=True
                                )
                                self._agent.send_logs(
                                    task_id, ["K8S Error: {}".format(msg)],
                                    session=self._session
                                )
                            except Exception as ex:
                                self.log.warning(
                                    'K8S Glue pending monitor: Failed deleting task "{}"\nEX: {}'.format(task_id, ex)
                                )

                            # clean up any msg for this task
                            self._last_tasks_msgs.pop(task_id, None)
                            continue

                    self._update_pending_task_msg(task_id, msg, tags)

                if task_id_to_pod:
                    self._process_tasks_for_pending_pods(task_id_to_pod)

                # clean up any last message for a task that wasn't seen as a pod
                self._last_tasks_msgs = {k: v for k, v in self._last_tasks_msgs.items() if k in task_id_to_pod}
            except GetPodsError:
                pass
            except Exception:
                self.log.exception("Hanging pods daemon loop")

            sleep(self._polling_interval)

    def delete_k8s_resource(self, k8s_resource: dict, msg: str = None):
        delete_cmd = "kubectl delete {kind} {name} -n {namespace} --output name".format(
            kind=self._agent.kind,
            name=self._get_k8s_resource_name(k8s_resource),
            namespace=self._get_k8s_resource_namespace(k8s_resource)
        ).strip()
        self.log.debug(" - deleting {} {}: {}".format(self._agent.kind, (" " + msg) if msg else "", delete_cmd))
        return get_bash_output(delete_cmd).strip()

    def _process_tasks_for_pending_pods(self, task_id_to_details: Dict[str, dict]):
        self._handle_aborted_tasks(task_id_to_details)

    def _handle_aborted_tasks(self, pending_tasks_details: Dict[str, dict]):
        try:
            result = self._session.get(
                service='tasks',
                action='get_all',
                json={
                    "id": list(pending_tasks_details),
                    "status": ["stopped"],
                    "only_fields": ["id"]
                }
            )
            aborted_task_ids = list(filter(None, (task.get("id") for task in result["tasks"])))

            for task_id in aborted_task_ids:
                pod = pending_tasks_details.get(task_id)
                if not pod:
                    self.log.error("Failed locating aborted task {} in pending pods list".format(task_id))
                    continue

                pod_name = self._get_pod_name(pod)
                if not self.get_pods(pod_name=pod_name):
                    self.log.debug("K8S Glue pending monitor: pod {} is no longer pending, skipping".format(pod_name))
                    continue

                resource_name = self._get_k8s_resource_name(pod)
                self.log.info(
                    "K8S Glue pending monitor: task {} was aborted but the k8s resource {} is still pending, "
                    "deleting pod".format(task_id, resource_name)
                )

                result = self._session.get(
                    service='tasks',
                    action='get_all',
                    json={"id": [task_id], "status": ["stopped"], "only_fields": ["id"]},
                )
                if not result["tasks"]:
                    self.log.debug("K8S Glue pending monitor: task {} is no longer aborted, skipping".format(task_id))
                    continue

                output = self.delete_k8s_resource(k8s_resource=pod, msg="Pending resource of an aborted task")
                if not output:
                    self.log.warning("K8S Glue pending monitor: failed deleting resource {}".format(resource_name))
        except Exception as ex:
            self.log.warning(
                'K8S Glue pending monitor: failed checking aborted tasks for pending resources: {}'.format(ex)
            )

    def _update_pending_task_msg(self, task_id: str, msg: str, tags: List[str] = None):
        if not msg or self._last_tasks_msgs.get(task_id, None) == (msg, tags):
            return
        try:
            if ENV_POD_MONITOR_DISABLE_ENQUEUE_ON_PREEMPTION.get():
                # This disables the option to enqueue the task which is supposed to sync the ClearML task status
                # in case the pod was preempted. In some cases this does not happen due to preemption but due to
                # cluster communication lag issues that cause us not to discover the pod is no longer pending and
                # enqueue the task when it's actually already running, thus essentially killing the task
                pass
            else:
                # Make sure the task is queued
                result = self._session.send_request(
                    service='tasks',
                    action='get_all',
                    json={"id": task_id, "only_fields": ["status"]},
                    method=Request.def_method,
                    async_enable=False,
                )
                if result.ok:
                    status = get_path(result.json(), 'data', 'tasks', 0, 'status')
                    # if task is in progress, change its status to enqueued
                    if status == "in_progress":
                        result = self._session.send_request(
                            service='tasks', action='enqueue',
                            json={
                                "task": task_id, "force": True, "queue": self._agent.k8s_pending_queue_id
                            },
                            method=Request.def_method,
                            async_enable=False,
                        )
                        if not result.ok:
                            result_msg = get_path(result.json(), 'meta', 'result_msg')
                            self.log.debug(
                                "K8S Glue pods monitor: failed forcing task status change"
                                " for pending task {}: {}".format(task_id, result_msg)
                            )

            # Update task status message
            payload = {"task": task_id, "status_message": "K8S glue status: {}".format(msg)}
            if tags:
                payload["tags"] = tags
            result = self._session.send_request('tasks', 'update', json=payload, method=Request.def_method)
            if not result.ok:
                result_msg = get_path(result.json(), 'meta', 'result_msg')
                raise Exception(result_msg or result.text)

            self._agent.send_logs(
                task_id, ["Kubernetes Pod status: {}".format(msg)],
                session=self._session
            )

            # update last msg for this task
            self._last_tasks_msgs[task_id] = msg
        except Exception as ex:
            self.log.warning(
                'K8S Glue pods monitor: Failed setting status message for task "{}"\nMSG: {}\nEX: {}'.format(
                    task_id, msg, ex
                )
            )
