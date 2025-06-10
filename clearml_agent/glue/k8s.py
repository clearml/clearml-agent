from __future__ import print_function, division, unicode_literals

import base64
import functools
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict, namedtuple
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from time import sleep, time
from typing import Text, List, Callable, Any, Collection, Optional, Union, Iterable, Dict, Tuple, Set

from clearml_agent.commands.events import Events
from clearml_agent.commands.worker import Worker, get_task_container, set_task_container, get_next_task
from clearml_agent.definitions import (
    ENV_DOCKER_IMAGE,
    ENV_AGENT_GIT_USER,
    ENV_AGENT_GIT_PASS,
    ENV_FORCE_SYSTEM_SITE_PACKAGES,
    ENV_AGENT_DEBUG_GET_NEXT_TASK,
)
from clearml_agent.errors import APIError, UsageError
from clearml_agent.glue.errors import GetPodCountError
from clearml_agent.glue.utilities import get_path, get_bash_output
from clearml_agent.glue.pending_pods_daemon import PendingPodsDaemon
from clearml_agent.helper.base import safe_remove_file
from clearml_agent.helper.dicts import merge_dicts
from clearml_agent.helper.process import get_bash_output, stringify_bash_output
from clearml_agent.helper.resource_monitor import ResourceMonitor
from clearml_agent.interface.base import ObjectID
from clearml_agent.backend_api.session import Request
from clearml_agent.glue.definitions import (
    ENV_START_AGENT_SCRIPT_PATH,
    ENV_DEFAULT_EXECUTION_AGENT_ARGS,
    ENV_POD_AGENT_INSTALL_ARGS,
    ENV_POD_USE_IMAGE_ENTRYPOINT,
    ENV_KUBECTL_IGNORE_ERROR,
    ENV_DEFAULT_SCHEDULER_QUEUE_TAGS,
)
from .._vendor import pyyaml as yaml


class K8sIntegration(Worker):
    SUPPORTED_KIND = ("pod", "job")
    K8S_PENDING_QUEUE = "k8s_scheduler"
    K8S_DEFAULT_NAMESPACE = "clearml"
    AGENT_LABEL = "CLEARML=agent"
    QUEUE_LABEL = "clearml-agent-queue"

    KUBECTL_APPLY_CMD = "kubectl apply --namespace={namespace} -f"

    BASH_INSTALL_SSH_CMD = [
        "(apt-get update -y ; apt-get install -y openssh-server) || "
        "(dnf install -y openssh-server)",
        "mkdir -p /var/run/sshd",
        "echo 'root:training' | chpasswd",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config",
        r"sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd",
        "echo 'AcceptEnv TRAINS_API_ACCESS_KEY TRAINS_API_SECRET_KEY CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY' "
        ">> /etc/ssh/sshd_config",
        'echo "export VISIBLE=now" >> /etc/profile',
        'echo "export PATH=$PATH" >> /etc/profile',
        'echo "ldconfig" >> /etc/profile',
        "/usr/sbin/sshd -p {port}"]

    _CONTAINER_APT_SCRIPT_SECTION = [
        "export DEBIAN_FRONTEND='noninteractive'",
        "echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean",
        "chown -R $(whoami) $HOME/.cache/pip",
        "(apt-get update -y ; apt-get install -y git) || "
        "(dnf install -y git)"
        # should only be added if docker_install_opencv_libs:  # libsm6 libxext6 libxrender-dev libglib2.0-0",
    ]

    CONTAINER_BASH_SCRIPT = [
        *(
            '[ ! -z "$CLEARML_AGENT_SKIP_CONTAINER_APT" ] || {}'.format(line)
            for line in _CONTAINER_APT_SCRIPT_SECTION
        ),
        "declare LOCAL_PYTHON",
        "[ ! -z $LOCAL_PYTHON ] || for i in {{20..5}}; do (which python3.$i 2> /dev/null || command -v python3.$i) && python3.$i -m pip --version && "
        "export LOCAL_PYTHON=$(which python3.$i 2> /dev/null || command -v python3.$i) && break ; done",
        '[ ! -z "$CLEARML_AGENT_SKIP_CONTAINER_APT" ] || [ ! -z "$LOCAL_PYTHON" ] || '
        'apt-get install -y python3-pip || dnf install -y python3-pip',
        "[ ! -z $LOCAL_PYTHON ] || export LOCAL_PYTHON=python3",
        "rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED",  # remove PEP 668
        "{extra_bash_init_cmd}",
        "[ ! -z $CLEARML_AGENT_NO_UPDATE ] || $LOCAL_PYTHON -m pip install clearml-agent{agent_install_args}",
        "{extra_docker_bash_script}",
        "$LOCAL_PYTHON -m clearml_agent execute {default_execution_agent_args} --id {task_id}"
    ]

    DEFAULT_POD_NAME_PREFIX = "clearml-id-"
    DEFAULT_LIMIT_POD_LABEL = "ai.clearml.agent.serial=pod-{pod_number}"

    _edit_hyperparams_version = "2.9"

    def __init__(
            self,
            k8s_pending_queue_name=None,
            container_bash_script=None,
            debug=False,
            ports_mode=False,
            num_of_services=20,
            base_pod_num=1,
            user_props_cb=None,
            runtime_cb=None,
            overrides_yaml=None,
            template_yaml=None,
            clearml_conf_file=None,
            extra_bash_init_script=None,
            namespace=None,
            max_pods_limit=None,
            pod_name_prefix=None,
            limit_pod_label=None,
            force_system_packages=None,
            **kwargs
    ):
        """
        Initialize the k8s integration glue layer daemon

        :param str k8s_pending_queue_name: queue name to use when task is pending in the k8s scheduler
        :param str container_bash_script: container bash script to be executed in k8s (default: CONTAINER_BASH_SCRIPT)
            Notice this string will use format() call, if you have curly brackets they should be doubled { -> {{
            Format arguments passed: {task_id} and {extra_bash_init_cmd}
        :param bool debug: Switch logging on
        :param bool ports_mode: Adds a label to each pod which can be used in services in order to expose ports.
            Requires the `num_of_services` parameter.
        :param int num_of_services: Number of k8s services configured in the cluster. Required if `port_mode` is True.
            (default: 20)
        :param int base_pod_num: Used when `ports_mode` is True, sets the base pod number to a given value (default: 1)
        :param callable user_props_cb: An Optional callable allowing additional user properties to be specified
            when scheduling a task to run in a pod. Callable can receive an optional pod number and should return
            a dictionary of user properties (name and value). Signature is [[Optional[int]], Dict[str,str]]
        :param callable runtime_cb: An Optional callable allowing additional task runtime to be specified (see user_props_cb)
        :param str overrides_yaml: YAML file containing the overrides for the pod (optional)
        :param str template_yaml: YAML file containing the template for the pod (optional).
            If provided the pod is scheduled with kubectl apply and overrides are ignored, otherwise with kubectl run.
        :param str clearml_conf_file: clearml.conf file to be use by the pod itself (optional)
        :param str extra_bash_init_script: Additional bash script to run before starting the Task inside the container
        :param str namespace: K8S namespace to be used when creating the new pods (default: clearml)
        :param int max_pods_limit: Maximum number of pods that K8S glue can run at the same time
        """
        super(K8sIntegration, self).__init__()
        self.kind = os.environ.get("CLEARML_K8S_GLUE_KIND", "pod").strip().lower()
        if self.kind not in self.SUPPORTED_KIND:
            raise UsageError(f"Kind '{self.kind}' not supported (expected {','.join(self.SUPPORTED_KIND)})")
        self.using_jobs = self.kind == "job"
        self.pod_name_prefix = pod_name_prefix or self.DEFAULT_POD_NAME_PREFIX
        self.limit_pod_label = limit_pod_label or self.DEFAULT_LIMIT_POD_LABEL
        self.k8s_pending_queue_name = k8s_pending_queue_name or self.K8S_PENDING_QUEUE
        self.k8s_pending_queue_id = None
        self.container_bash_script = container_bash_script or self.CONTAINER_BASH_SCRIPT
        if force_system_packages is None:
            force_system_packages = ENV_FORCE_SYSTEM_SITE_PACKAGES.get()
        self._force_system_site_packages = force_system_packages if force_system_packages is not None else True
        if self._force_system_site_packages:
            # Use system packages, because by we will be running inside a docker
            self._session.config.put("agent.package_manager.system_site_packages", True)
        # Add debug logging
        if debug:
            self.log.logger.disabled = False
            self.log.logger.setLevel(logging.DEBUG)
            self.log.logger.addHandler(logging.StreamHandler())
        self.ports_mode = ports_mode
        self.num_of_services = num_of_services
        self.base_pod_num = base_pod_num
        self._edit_hyperparams_support = None
        self._user_props_cb = user_props_cb
        self._runtime_cb = runtime_cb
        self.conf_file_content = None
        self.overrides_json_string = None
        self.template_dict = None
        self.extra_bash_init_script = extra_bash_init_script or None
        if self.extra_bash_init_script and not isinstance(self.extra_bash_init_script, str):
            self.extra_bash_init_script = ' ; '.join(self.extra_bash_init_script)  # noqa
        self.namespace = namespace or self.K8S_DEFAULT_NAMESPACE
        self.pod_limits = []
        self.pod_requests = []
        self.max_pods_limit = max_pods_limit if not self.ports_mode else None

        self._load_overrides_yaml(overrides_yaml)

        if template_yaml:
            self.template_dict = self._load_template_file(template_yaml)

        clearml_conf_file = clearml_conf_file or kwargs.get('trains_conf_file')

        if clearml_conf_file:
            with open(os.path.expandvars(os.path.expanduser(str(clearml_conf_file))), 'rt') as f:
                self.conf_file_content = f.read()

        self._agent_label = None

        self._pending_pods_daemon = self._create_daemon_instance(
            cls_=PendingPodsDaemon,
            polling_interval=self._polling_interval
        )
        self._pending_pods_daemon.start()

        self._min_cleanup_interval_per_ns_sec = 1.0
        self._last_pod_cleanup_per_ns = defaultdict(lambda: 0.)

        self._server_supports_same_state_transition = (
                self._session.feature_set != "basic" and self._session.check_min_server_version("3.22.3")
        )

        self.ignore_kubectl_errors_re = (
            re.compile(ENV_KUBECTL_IGNORE_ERROR.get()) if ENV_KUBECTL_IGNORE_ERROR.get() else None
        )

    @property
    def agent_label(self):
        return self._get_agent_label()

    def _create_daemon_instance(self, cls_, **kwargs):
        return cls_(agent=self, **kwargs)

    def _load_overrides_yaml(self, overrides_yaml):
        if not overrides_yaml:
            return
        overrides = self._load_template_file(overrides_yaml)
        if not overrides:
            return
        containers = overrides.get('spec', {}).get('containers', [])
        for c in containers:
            resources = {str(k).lower(): v for k, v in c.get('resources', {}).items()}
            if not resources:
                continue
            if resources.get('limits'):
                self.pod_limits += ['{}={}'.format(k, v) for k, v in resources['limits'].items()]
            if resources.get('requests'):
                self.pod_requests += ['{}={}'.format(k, v) for k, v in resources['requests'].items()]
        # remove double entries
        self.pod_limits = list(set(self.pod_limits))
        self.pod_requests = list(set(self.pod_requests))
        if self.pod_limits or self.pod_requests:
            self.log.warning('Found pod container requests={} limits={}'.format(
                self.pod_limits, self.pod_requests))
        if containers:
            self.log.warning('Removing containers section: {}'.format(overrides['spec'].pop('containers')))
        self.overrides_json_string = json.dumps(overrides)

    @staticmethod
    def _load_template_file(path):
        with open(os.path.expandvars(os.path.expanduser(str(path))), 'rt') as f:
            return yaml.load(f, Loader=getattr(yaml, 'FullLoader', None))

    @staticmethod
    def _get_path(d, *path, default=None):
        try:
            return functools.reduce(
                lambda a, b: a[b], path, d
            )
        except (IndexError, KeyError):
            return default

    def _get_kubectl_options(self, command, extra_labels=None, filters=None, output="json", labels=None, ns=None):
        # type: (str, Iterable[str], Iterable[str], str, Iterable[str], str) -> Dict
        if labels is False:
            labels = []
        elif not labels:
            labels = [self._get_agent_label()]
        labels = list(labels) + (list(extra_labels) if extra_labels else [])
        d = {
            "-n": ns or str(self.namespace),
            "-o": output,
        }
        if labels:
            d["-l"] = ",".join(labels)
        if filters:
            d["--field-selector"] = ",".join(filters)
        return d

    def get_kubectl_command(self, command, output="json", **args):
        opts = self._get_kubectl_options(command, output=output, **args)
        return 'kubectl {command} {opts}'.format(
            command=command, opts=" ".join(x for item in opts.items() for x in item)
        )

    def _set_task_user_properties(self, task_id: str, task_session=None, **properties: str):
        session = task_session or self._session
        if self._edit_hyperparams_support is not True:
            # either not supported or never tested
            if self._edit_hyperparams_support == self._session.api_version:
                # tested against latest api_version, not supported
                return
            if not self._session.check_min_api_version(self._edit_hyperparams_version):
                # not supported due to insufficient api_version
                self._edit_hyperparams_support = self._session.api_version
                return
        try:
            session.get(
                service="tasks",
                action="edit_hyper_params",
                task=task_id,
                hyperparams=[
                    {
                        "section": "properties",
                        "name": k,
                        "value": str(v),
                    }
                    for k, v in properties.items()
                ],
            )
            # definitely supported
            self._runtime_props_support = True
        except APIError as error:
            if error.code == 404:
                self._edit_hyperparams_support = self._session.api_version

    def _get_agent_label(self):
        if not self.worker_id:
            print('WARNING! no worker ID found!!!')
            return self.AGENT_LABEL

        if not self._agent_label:
            h = hashlib.md5()
            h.update(str(self.worker_id).encode('utf-8'))
            self._agent_label = '{}-{}'.format(self.AGENT_LABEL, h.hexdigest()[:8])

        return self._agent_label

    RunningPod = namedtuple("RunningPod", "name queue namespace")

    def _get_running_pods(self):
        try:
            kubectl_cmd = self.get_kubectl_command(
                "get pods",
                output="jsonpath=\"{{range .items[*]}}{{.metadata.name}}{{' '}}{{.metadata.namespace}}{{' '}}"
                       "{{.metadata.labels.{}}}{{'\\n'}}{{end}}\"".format(self.QUEUE_LABEL)
            )
            self.log.debug("Getting used pods: {}".format(kubectl_cmd))
            output = stringify_bash_output(get_bash_output(kubectl_cmd, raise_error=True))

            if not output:
                # No such pod exist so we can use the pod_number we found
                return []

            try:
                return [
                    self.RunningPod(
                        name=parts[0],
                        namespace=parts[1],
                        queue=parts[2]
                    )
                    for parts in (line.split(" ") for line in output.splitlines())
                ]
            except Exception as ex:
                raise Exception("Failed parsing used pods command response for cleanup: {}".format(ex))
        except Exception as ex:
            raise Exception('Failed obtaining used pods information: {}'.format(ex))

    def _get_used_pods(self):
        # type: () -> Tuple[int, Set[str]]
        # noinspection PyBroadException
        try:
            items = self._get_running_pods()
            if not items:
                return 0, set([])
            current_pod_count = len(items)
            namespaces = {item.namespace for item in items}
            self.log.debug(" - found {} pods in namespaces {}".format(current_pod_count, ", ".join(namespaces)))
            return current_pod_count, namespaces
        except Exception as ex:
            self.log.debug("Failed getting used pods: {}", ex)
            return -1, set([])

    def _is_same_tenant(self, task_session):
        if not task_session or task_session is self._session:
            return True
        # noinspection PyStatementEffect
        try:
            tenant = self._session.get_decoded_token(self._session.token, verify=False)["tenant"]
            task_tenant = task_session.get_decoded_token(task_session.token, verify=False)["tenant"]
            return tenant == task_tenant
        except Exception as ex:
            print("ERROR: Failed getting tenant for task session: {}".format(ex))

    def get_jobs_info(self, info_path: str, condition: str = None, namespace=None, debug_msg: str = None)\
            -> Dict[str, str]:
        cond = "==".join((x.strip("=") for x in condition.partition("=")[::2]))
        output = f"jsonpath='{{range .items[?(@.{cond})]}}{{@.{info_path}}}{{\" \"}}{{@.metadata.namespace}}{{\"\\n\"}}{{end}}'"
        kubectl_cmd = self.get_kubectl_command("get job", output=output, ns=namespace)
        if debug_msg:
            self.log.debug(debug_msg.format(cmd=kubectl_cmd))
        output = stringify_bash_output(get_bash_output(kubectl_cmd))
        output = output.strip("'")  # for Windows debugging :(
        try:
            data_items = dict(l.strip().partition(" ")[::2] for l in output.splitlines())
            return data_items
        except Exception as ex:
            self.log.warning('Failed parsing kubectl output:\n{}\nEx: {}'.format(output, ex))

    def get_pods_for_jobs(self, job_condition: str = None, pod_filters: List[str] = None, debug_msg: str = None):
        # Use metadata.uid so job related pods can be found filterin g following list with this param
        controller_uids = self.get_jobs_info(
            "metadata.uid", condition=job_condition, debug_msg=debug_msg
        )
        if not controller_uids:
            # No pods were found for these jobs
            return []
        pods = self.get_pods(filters=pod_filters, debug_msg=debug_msg)
        return [
            pod for pod in pods
            if get_path(pod, "metadata", "labels", "controller-uid") in controller_uids
        ]

    def get_pods(self, filters: List[str] = None, debug_msg: str = None):
        kubectl_cmd = self.get_kubectl_command(
            "get pods",
            filters=filters,
            labels=False if self.using_jobs else None,
        )
        if debug_msg:
            self.log.debug(debug_msg.format(cmd=kubectl_cmd))
        output = stringify_bash_output(get_bash_output(kubectl_cmd))
        try:
            output_config = json.loads(output)
        except Exception as ex:
            self.log.warning('Failed parsing kubectl output:\n{}\nEx: {}'.format(output, ex))
            return
        return output_config.get('items', [])

    def _get_pod_count(self, extra_labels: List[str] = None, msg: str = None):
            kubectl_cmd_new = self.get_kubectl_command(
                f"get {self.kind}s",
                extra_labels= extra_labels
            )
            self.log.debug("{}{}".format((msg + ": ") if msg else "", kubectl_cmd_new))
            process = subprocess.Popen(kubectl_cmd_new.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            output = stringify_bash_output(output)
            error = stringify_bash_output(error)

            try:
                return len(json.loads(output).get("items", []))
            except (ValueError, TypeError) as ex:
                self.log.warning(
                    "K8S Glue pods monitor: Failed parsing kubectl output:\n{}\nEx: {}".format(output, ex)
                )
                raise GetPodCountError()

    def resource_applied(self, resource_name: str, namespace: str, task_id: str, session):
        """ Called when a resource (pod/job) was applied """
        pass

    def ports_mode_supported_for_task(self, task_id: str, task_data):
        return self.ports_mode

    def get_default_docker_image(self, session, queue: str) -> str:
        return str(ENV_DOCKER_IMAGE.get() or session.config.get("agent.default_docker.image", "nvidia/cuda"))

    def run_one_task(self, queue: Text, task_id: Text, worker_args=None, task_session=None, **_):
        print('Pulling task {} launching on kubernetes cluster'.format(task_id))
        session = task_session or self._session
        task_data = session.api_client.tasks.get_all(id=[task_id])[0]

        # push task into the k8s queue, so we have visibility on pending tasks in the k8s scheduler
        if self._is_same_tenant(task_session):
            try:
                print('Pushing task {} into temporary pending queue'.format(task_id))

                if not self._server_supports_same_state_transition:
                    _ = session.api_client.tasks.stop(task_id, force=True, status_reason="moving to k8s pending queue")

                # Just make sure to clean up in case the task is stuck in the queue (known issue)
                self._session.api_client.queues.remove_task(
                    task=task_id,
                    queue=self.k8s_pending_queue_id,
                )

                for attempt in range(2):
                    res = self._session.send_request(
                        "tasks",
                        "enqueue",
                        json={
                            "task": task_id,
                            "queue": self.k8s_pending_queue_id,
                            "status_reason": "k8s pending scheduler",
                            "update_execution_queue": False,
                        }
                    )
                    if res.ok:
                        break

                    # noinspection PyBroadException
                    try:
                        result_subcode = res.json()["meta"]["result_subcode"]
                        result_msg = res.json()["meta"]["result_msg"]
                    except Exception:
                        result_subcode = None
                        result_msg = res.text

                    if attempt == 0 and res.status_code == 400 and result_subcode == 701:
                        # Invalid queue ID, only retry once
                        self._ensure_pending_queue_exists()
                        continue
                    raise Exception(result_msg)

            except Exception as e:
                self.log.error("ERROR: Could not push back task [{}] to k8s pending queue {} [{}], error: {}".format(
                    task_id, self.k8s_pending_queue_name, self.k8s_pending_queue_id, e))
                return

        container = get_task_container(session, task_id)
        if not container.get('image'):
            container['image'] = self.get_default_docker_image(session, queue)
            container['arguments'] = session.config.get("agent.default_docker.arguments", None)
            set_task_container(
                session, task_id, docker_image=container['image'], docker_arguments=container['arguments']
            )

        # get the clearml.conf encoded file, make sure we use system packages!

        git_user = ENV_AGENT_GIT_USER.get() or self._session.config.get("agent.git_user", None)
        git_pass = ENV_AGENT_GIT_PASS.get() or self._session.config.get("agent.git_pass", None)
        extra_config_values = [
            'agent.package_manager.system_site_packages: true' if self._force_system_site_packages else '',
            'agent.git_user: "{}"'.format(git_user) if git_user else '',
            'agent.git_pass: "{}"'.format(git_pass) if git_pass else '',
        ]

        # noinspection PyProtectedMember
        config_content = (
            self.conf_file_content or (session._config_file and Path(session._config_file).read_text()) or ""
        ) + '\n{}\n'.format('\n'.join(x for x in extra_config_values if x))

        hocon_config_encoded = config_content.encode("ascii")

        clearml_conf_create_script = ["echo '{}' | base64 --decode >> ~/clearml.conf".format(
            base64.b64encode(
                hocon_config_encoded
            ).decode('ascii')
        )]

        if task_session:
            clearml_conf_create_script.append(
                "export CLEARML_AUTH_TOKEN=$(echo '{}' | base64 --decode)".format(
                    base64.b64encode(task_session.token.encode("ascii")).decode('ascii')
                )
            )

        ports_mode = False
        if self.ports_mode_supported_for_task(task_id, task_data):
            print("Kubernetes looking for available pod to use")
            ports_mode = True

        # noinspection PyBroadException
        try:
            queue_name = self._session.api_client.queues.get_by_id(queue=queue).name
        except Exception:
            queue_name = 'k8s'

        # Search for a free pod number
        pod_count = 0
        pod_number = self.base_pod_num
        while ports_mode or self.max_pods_limit:
            pod_number = self.base_pod_num + pod_count

            try:
                items_count = self._get_pod_count(
                    extra_labels=[self.limit_pod_label.format(pod_number=pod_number)] if ports_mode else None,
                    msg="Looking for a free pod/port"
                )
            except GetPodCountError:
                self.log.warning(
                    "K8S Glue pods monitor: task '{}' will be enqueued back to queue '{}'".format(
                        task_id, queue
                    )
                )
                session.api_client.tasks.stop(task_id, force=True)
                # noinspection PyBroadException
                try:
                    self._session.api_client.tasks.enqueue(task_id, queue=queue, status_reason='kubectl parsing error')
                except:
                    self.log.warning("Failed enqueuing task to queue '{}'".format(queue))
                return

            if not items_count:
                # No such pod exist so we can use the pod_number we found (result exists but with no items)
                break

            if self.max_pods_limit:
                current_pod_count = items_count
                max_count = self.max_pods_limit
            else:
                current_pod_count = pod_count
                max_count = self.num_of_services - 1

            if current_pod_count >= max_count:
                # All pods are taken, exit
                self.log.warning(
                    "All k8s services are in use, task '{}' "
                    "will be enqueued back to queue '{}'".format(
                        task_id, queue
                    )
                )
                session.api_client.tasks.stop(task_id, force=True)
                # noinspection PyBroadException
                try:
                    self._session.api_client.tasks.enqueue(
                        task_id, queue=queue, status_reason='k8s max pod limit (no free k8s service)'
                    )
                except:
                    self.log.warning("Failed enqueuing task to queue '{}'".format(queue))
                return
            elif self.max_pods_limit:
                # max pods limit hasn't reached yet, so we can create the pod
                break
            pod_count += 1

        labels = self._get_pod_labels(queue, queue_name, task_data)
        if ports_mode:
            labels.append(self.limit_pod_label.format(pod_number=pod_number))

        if ports_mode:
            print("Kubernetes scheduling task id={} on pod={} (pod_count={})".format(task_id, pod_number, pod_count))
        else:
            print("Kubernetes scheduling task id={}".format(task_id))

        try:
            template = self._resolve_template(task_session, task_data, queue, task_id)
        except Exception as ex:
            print("ERROR: Failed resolving template (skipping): {}".format(ex))
            return

        try:
            namespace = template['metadata']['namespace'] or self.namespace
        except (KeyError, TypeError, AttributeError):
            namespace = self.namespace

        if not template:
            print("ERROR: no template for task {}, skipping".format(task_id))
            return

        pod_name = self.pod_name_prefix + str(task_id)

        self.apply_template_and_handle_result(
            pod_name=pod_name,
            clearml_conf_create_script=clearml_conf_create_script,
            labels=labels,
            queue=queue,
            task_id=task_id,
            namespace=namespace,
            template=template,
            docker_image=container['image'],
            docker_args=container.get('arguments'),
            docker_bash=container.get('setup_shell_script'),
            session=session,
            task_session=task_session,
            pod_number=pod_number,
            queue_name=queue_name,
            task_data=task_data,
            ports_mode=ports_mode,
            pod_count=pod_count,
        )

    def apply_template_and_handle_result(
            self,
            pod_name,
            clearml_conf_create_script: List[str],
            labels,
            queue,
            task_id,
            namespace,
            template,
            docker_image,
            docker_args,
            docker_bash,
            session,
            task_session,
            queue_name,
            task_data,
            ports_mode,
            pod_count,
            pod_number=None,
            base_spec: dict = None,
    ):
        """Apply the provided template with all custom settings and handle bookkeeping for the reaults"""

        output, error, pod_name = self._kubectl_apply(
            pod_name=pod_name,
            template=template,
            pod_number=pod_number,
            clearml_conf_create_script=clearml_conf_create_script,
            labels=labels,
            docker_image=docker_image,
            docker_args=docker_args,
            docker_bash=docker_bash,
            task_id=task_id,
            queue=queue,
            namespace=namespace,
            task_token=task_session.token.encode("ascii") if task_session else None,
            base_spec=base_spec,
        )

        print('kubectl output:\n{}\n{}'.format(error, output))
        if error:
            if self.ignore_kubectl_errors_re and self.ignore_kubectl_errors_re.match(error):
                print(f"Ignoring error due to {ENV_KUBECTL_IGNORE_ERROR.key}")
            else:
                self._set_task_failed_while_applying(
                    session, task_id, f"Running kubectl encountered an error: {error}"
                )
                return

        if pod_name:
            self.resource_applied(
                resource_name=pod_name, namespace=namespace, task_id=task_id, session=session
            )

        self.set_task_info(
            task_id=task_id, task_session=task_session, queue_name=queue_name, ports_mode=ports_mode,
            pod_number=pod_number, pod_count=pod_count, task_data=task_data
        )

    def _set_task_failed_while_applying(self, session, task_id: str, error: str):
        send_log = "Running kubectl encountered an error: {}".format(error)
        self.log.error(send_log)
        self.send_logs(task_id, send_log.splitlines())

        # Make sure to remove the task from our k8s pending queue
        self._session.api_client.queues.remove_task(
            task=task_id,
            queue=self.k8s_pending_queue_id,
        )
        # Set task as failed
        session.api_client.tasks.failed(task_id, force=True)

    def set_task_info(
            self, task_id: str, task_session, task_data, queue_name: str, ports_mode: bool, pod_number, pod_count
    ):
        user_props = {"k8s-queue": str(queue_name)}
        runtime = {}
        if ports_mode:
            agent_label = self._get_agent_label()
            user_props.update({
                "k8s-pod-number": pod_number,
                "k8s-pod-label": agent_label,  # backwards-compatibility / legacy
                "k8s-internal-pod-count": pod_count,
                "k8s-agent": agent_label,
            })

        if self._user_props_cb:
            # noinspection PyBroadException
            try:
                custom_props = self._user_props_cb(pod_number) if ports_mode else self._user_props_cb()
                user_props.update(custom_props)
            except Exception:
                pass

        if self._runtime_cb:
            # noinspection PyBroadException
            try:
                custom_runtime = self._runtime_cb(pod_number) if ports_mode else self._runtime_cb()
                runtime.update(custom_runtime)
            except Exception:
                pass

        if user_props:
            self._set_task_user_properties(
                task_id=task_id,
                task_session=task_session,
                **user_props
            )

        if runtime:
            task_runtime = self._get_task_runtime(task_id) or {}
            task_runtime.update(runtime)

            try:
                res = task_session.send_request(
                    service='tasks', action='edit', method=Request.def_method,
                    json={
                        "task": task_id, "force": True, "runtime": task_runtime
                    },
                )
                if not res.ok:
                    raise Exception("failed setting runtime property")
            except Exception as ex:
                print("WARNING: failed setting custom runtime properties for task '{}': {}".format(task_id, ex))

    def _get_task_runtime(self, task_id) -> Optional[dict]:
        try:
            res = self._session.send_request(
                service='tasks', action='get_by_id', method=Request.def_method,
                json={"task": task_id, "only_fields": ["runtime"]},
            )
            if not res.ok:
                raise ValueError(f"request returned {res.status_code}")
            data = res.json().get("data")
            if not data or "task" not in data:
                raise ValueError("empty data in result")
            return data["task"].get("runtime", {})
        except Exception as ex:
            print(f"ERROR: Failed getting runtime properties for task {task_id}: {ex}")

    def _get_pod_labels(self, queue, queue_name, task_data):
        return [
            self._get_agent_label(),
            "{}={}".format(self.QUEUE_LABEL, self._safe_k8s_label_value(queue)),
            "{}-name={}".format(self.QUEUE_LABEL, self._safe_k8s_label_value(queue_name))
        ]

    def _get_docker_args(self, docker_args, flags, target=None, convert=None):
        # type: (List[str], Collection[str], Optional[str], Callable[[str], Any]) -> Union[dict, List[str]]
        """
        Get docker args matching specific flags.

        :argument docker_args: List of docker argument strings (flags and values)
        :argument flags: List of flags/names to intercept (e.g. "--env" etc.)
        :argument target: Controls return format. If provided, returns a dict with a target field containing a list
         of result strings, otherwise returns a list of result strings
        :argument convert: Optional conversion function for each result string
        """
        args = docker_args[:] if docker_args else []
        results = []
        while args:
            cmd = args.pop(0).strip()
            if cmd in flags:
                env = args.pop(0).strip()
                if convert:
                    env = convert(env)
                results.append(env)
            else:
                self.log.warning('skipping docker argument {} (only -e --env supported)'.format(cmd))
        if target:
            return {target: results} if results else {}
        return results

    def get_task_worker_id(self, template, task_id, pod_name, namespace, queue):
        return f"{self.worker_id}:{task_id}"

    def _create_template_container(
        self, pod_name: str, task_id: str, docker_image: str, docker_args: List[str],
        docker_bash: str, clearml_conf_create_script: List[str], task_worker_id: str, task_token: str = None
    ) -> dict:
        container = self._get_docker_args(
            docker_args,
            target="env",
            flags={"-e", "--env"},
            convert=lambda env: {'name': env.partition("=")[0], 'value': env.partition("=")[2]},
        )

        def add_or_update_env_var(name, value):
            env_vars = container.get('env', [])
            for entry in env_vars:
                if entry.get('name') == name:
                    entry['value'] = value
                    break
            else:
                container['env'] = env_vars + [{'name': name, 'value': value}]

        # Set worker ID
        add_or_update_env_var('CLEARML_WORKER_ID', task_worker_id)

        if ENV_POD_USE_IMAGE_ENTRYPOINT.get():
            # Don't add a cmd and args, just the image

            # Add the task ID and token since we need it (it's usually in the init script passed to us
            add_or_update_env_var('CLEARML_TASK_ID', task_id)
            if task_token:
                # TODO: find a way to base64 encode the token
                add_or_update_env_var('CLEARML_AUTH_TOKEN', task_token)

            return self._merge_containers(
                container, dict(name=pod_name, image=docker_image)
            )

        # Create bash script for container and
        container_bash_script = [self.container_bash_script] if isinstance(self.container_bash_script, str) \
            else self.container_bash_script

        extra_docker_bash_script = '\n'.join(self._session.config.get("agent.extra_docker_shell_script", None) or [])
        if docker_bash:
            extra_docker_bash_script += '\n' + str(docker_bash) + '\n'

        script_encoded = '\n'.join(
            ['#!/bin/bash', ] +
            [line.format(extra_bash_init_cmd=self.extra_bash_init_script or '',
                         task_id=task_id,
                         extra_docker_bash_script=extra_docker_bash_script,
                         default_execution_agent_args=ENV_DEFAULT_EXECUTION_AGENT_ARGS.get(),
                         agent_install_args=ENV_POD_AGENT_INSTALL_ARGS.get())
             for line in container_bash_script])

        extra_bash_commands = list(clearml_conf_create_script or [])

        start_agent_script_path = ENV_START_AGENT_SCRIPT_PATH.get() or "~/__start_agent__.sh"

        extra_bash_commands.append(
            "echo '{content}' | base64 --decode >> {script_path} ; /bin/bash {script_path}".format(
                content=base64.b64encode(
                    script_encoded.encode('ascii')
                ).decode('ascii'),
                script_path=start_agent_script_path
            )
        )

        # Notice: we always leave with exit code 0, so pods are never restarted
        return self._merge_containers(
            container,
            dict(name=pod_name, image=docker_image,
                 command=['/bin/bash'],
                 args=['-c', '{} ; exit 0'.format(' ; '.join(extra_bash_commands))])
        )

    def _kubectl_apply(
        self,
        pod_name,
        clearml_conf_create_script: List[str],
        docker_image,
        docker_args,
        docker_bash,
        labels,
        queue,
        task_id,
        namespace,
        template,
        pod_number=None,
        task_token=None,
        base_spec: dict = None,  # base values for the spec (might be overridden)
    ):
        if "apiVersion" not in template:
            template["apiVersion"] = "batch/v1" if self.using_jobs else "v1"
        if "kind" in template:
            if template["kind"].lower() != self.kind:
                return (
                    "",
                    f"Template kind {template['kind']} does not maych kind {self.kind.capitalize()} set for agent",
                    None
                )
        else:
            template["kind"] = self.kind.capitalize()

        metadata = template.setdefault('metadata', {})
        metadata['name'] = pod_name

        def place_labels(metadata_dict):
            labels_dict = dict(pair.split('=', 1) for pair in labels)
            metadata_dict.setdefault('labels', {}).update(labels_dict)

        if labels:
            # Place labels on base resource (job or single pod)
            place_labels(metadata)

        spec = template.setdefault('spec', {})
        if self.using_jobs:
            spec.setdefault('backoffLimit', 0)
            spec_template = spec.setdefault('template', {})
            if labels:
                # Place same labels for any pod spawned by the job
                place_labels(spec_template.setdefault('metadata', {}))

            spec = spec_template.setdefault('spec', {})

        if base_spec:
            merge_dicts(spec, base_spec)

        containers = spec.setdefault('containers', [])
        spec.setdefault('restartPolicy', 'Never')

        task_worker_id = self.get_task_worker_id(template, task_id, pod_name, namespace, queue)

        container = self._create_template_container(
            pod_name=pod_name,
            task_id=task_id,
            docker_image=docker_image,
            docker_args=docker_args,
            docker_bash=docker_bash,
            clearml_conf_create_script=clearml_conf_create_script,
            task_worker_id=task_worker_id,
            task_token=task_token,
        )

        if containers:
            containers[0] = self._merge_containers(containers[0], container)
        else:
            containers.append(container)

        if self._docker_force_pull:
            for c in containers:
                c.setdefault('imagePullPolicy', 'Always')

        fp, yaml_file = tempfile.mkstemp(prefix='clearml_k8stmpl_', suffix='.yml')
        os.close(fp)
        with open(yaml_file, 'wt') as f:
            yaml.dump(template, f)

        self.log.debug("Applying template:\n{}".format(pformat(template, indent=2)))

        kubectl_cmd = self.KUBECTL_APPLY_CMD.format(
            task_id=task_id,
            docker_image=docker_image,
            queue_id=queue,
            namespace=namespace
        )
        # make sure we provide a list
        if isinstance(kubectl_cmd, str):
            kubectl_cmd = kubectl_cmd.split()

        # add the template file at the end
        kubectl_cmd += [yaml_file]
        try:
            process = subprocess.Popen(kubectl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
        except Exception as ex:
            return None, str(ex), None
        finally:
            safe_remove_file(yaml_file)

        return stringify_bash_output(output), stringify_bash_output(error), pod_name

    def _process_bash_lines_response(self, bash_cmd: str, raise_error=True):
        res = get_bash_output(bash_cmd, raise_error=raise_error)
        lines = [
            line for line in
            (r.strip().rpartition("/")[-1] for r in res.splitlines())
            if line.startswith(self.pod_name_prefix)
        ]
        return lines

    def _delete_pods(self, selectors: List[str], namespace: str, msg: str = None) -> List[str]:
        kubectl_cmd = \
            "kubectl delete pod -l={agent_label} " \
            "--namespace={namespace} --field-selector={selector} --output name".format(
                selector=",".join(selectors),
                agent_label=self._get_agent_label(),
                namespace=namespace,
            )
        self.log.debug("Deleting old/failed pods{} for ns {}: {}".format(
            msg or "", namespace, kubectl_cmd
        ))
        lines = self._process_bash_lines_response(kubectl_cmd)
        self.log.debug(" - deleted pods %s", ", ".join(lines))
        return lines

    def _delete_jobs_by_names(self, names_to_ns: Dict[str, str], msg: str = None) -> List[str]:
        if not names_to_ns:
            return []
        ns_to_names = defaultdict(list)
        for name, ns in names_to_ns.items():
            ns_to_names[ns].append(name)

        results = []
        for ns, names in ns_to_names.items():
            kubectl_cmd = "kubectl delete job --namespace={ns} --output=name {names}".format(
                ns=ns, names=" ".join(names)
            )
            self.log.debug("Deleting jobs {}: {}".format(
                msg or "", kubectl_cmd
            ))
            lines = self._process_bash_lines_response(kubectl_cmd)
            if not lines:
                continue
            self.log.debug(" - deleted jobs %s", ", ".join(lines))
            results.extend(lines)
        return results

    def _delete_completed_or_failed_pods(self, namespace, msg: str = None):
        if not self.using_jobs:
            return self._delete_pods(
                selectors=["status.phase!=Pending", "status.phase!=Running"], namespace=namespace, msg=msg
            )

        job_names_to_delete = {}

        # locate failed pods for jobs
        failed_pods = self.get_pods_for_jobs(
            job_condition="status.active=1",
            pod_filters=["status.phase!=Pending", "status.phase!=Running", "status.phase!=Terminating"],
            debug_msg="Deleting failed pods: {cmd}"
        )
        if failed_pods:
            job_names_to_delete = {
                get_path(pod, "metadata", "labels", "job-name"): get_path(pod, "metadata", "namespace")
                for pod in failed_pods
                if get_path(pod, "metadata", "labels", "job-name")
            }
            self.log.debug(f" - found jobs with failed pods: {' '.join(job_names_to_delete)}")

        completed_job_names = self.get_jobs_info(
            "metadata.name", condition="status.succeeded=1", namespace=namespace, debug_msg=msg
        )
        if completed_job_names:
            self.log.debug(f" - found completed jobs: {' '.join(completed_job_names)}")
            job_names_to_delete.update(completed_job_names)

        return self._delete_jobs_by_names(names_to_ns=job_names_to_delete, msg=msg)

    def _cleanup_old_pods(self, namespaces, extra_msg=None):
        # type: (Iterable[str], Optional[str]) -> Dict[str, List[str]]
        self.log.debug("Cleaning up pods")
        deleted_pods = defaultdict(list)
        for namespace in namespaces:
            if time() - self._last_pod_cleanup_per_ns[namespace] < self._min_cleanup_interval_per_ns_sec:
                # Do not try to clean up the same namespace too quickly
                continue

            try:
                res = self._delete_completed_or_failed_pods(namespace, extra_msg)
                deleted_pods[namespace].extend(res)
            except Exception as ex:
                self.log.error("Failed deleting completed/failed pods for ns %s: %s", namespace, str(ex))
            finally:
                self._last_pod_cleanup_per_ns[namespace] = time()

        # Locate tasks belonging to deleted pods that are still marked as pending or running
        tasks_to_abort = []
        try:
            task_ids = list(filter(None, (
                pod_name[len(self.pod_name_prefix):].strip()
                for pod_names in deleted_pods.values()
                for pod_name in pod_names
            )))
            if task_ids:
                result = self._session.get(
                    service='tasks',
                    action='get_all',
                    json={"id": task_ids, "status": ["in_progress", "queued"], "only_fields": ["id", "status", "status_reason"]},
                    method=Request.def_method,
                )
                tasks_to_abort = result["tasks"]
        except Exception as ex:
            self.log.warning('Failed getting running tasks for deleted {}(s): {}'.format(self.kind, ex))

        for task in tasks_to_abort:
            task_id = task.get("id")
            status = task.get("status")
            status_reason = (task.get("status_reason") or "").lower()
            if not task_id or not status:
                self.log.warning('Failed getting task information: id={}, status={}'.format(task_id, status))
                continue
            if status == "queued" and "pushed back by policy manager" in status_reason:
                # Task was pushed back to policy queue by policy manager, don't touch it
                continue
            try:
                if status == "queued":
                    self._session.get(
                        service='tasks',
                        action='dequeue',
                        json={
                            "task": task_id,
                            "force": True,
                            "status_reason": "Pod deleted (not pending or running)",
                            "status_message": "{} deleted by agent {}".format(
                                self.kind.capitalize(), self.worker_id or "unknown"
                            )
                        },
                        method=Request.def_method,
                    )
                self._session.get(
                    service='tasks',
                    action='failed',
                    json={
                        "task": task_id,
                        "force": True,
                        "status_reason": "Pod deleted (not pending or running)",
                        "status_message": "{} deleted by agent {}".format(
                            self.kind.capitalize(), self.worker_id or "unknown"
                        )
                    },
                    method=Request.def_method,
                )
            except Exception as ex:
                self.log.warning('Failed setting task {} to status "failed": {}'.format(task_id, ex))

        return deleted_pods

    def check_if_suspended(self) -> bool:
        pass

    def check_if_schedulable(self, queue: str) -> bool:
        return True

    def _ensure_pending_queue_exists(self):
        resolved_ids = self._resolve_queue_names(
            [self.k8s_pending_queue_name],
            create_if_missing=True,
            create_system_tags=ENV_DEFAULT_SCHEDULER_QUEUE_TAGS.get()
        )
        if not resolved_ids:
            raise ValueError(
                "Failed resolving or creating k8s pending queue {}".format(self.k8s_pending_queue_name)
            )
        self.k8s_pending_queue_id = resolved_ids[0]

    def run_tasks_loop(self, queues: List[Text], worker_params, **kwargs):
        """
        :summary: Pull and run tasks from queues.
        :description: 1. Go through ``queues`` by order.
                      2. Try getting the next task for each and run the first one that returns.
                      3. Go to step 1
        :param queues: IDs of queues to pull tasks from
        :type queues: list of ``Text``
        :param worker_params: Worker command line arguments
        :type worker_params: ``clearml_agent.helper.process.WorkerParams``
        """
        # print("debug> running tasks loop")

        events_service = self.get_service(Events)

        if not self.k8s_pending_queue_id:
            self._ensure_pending_queue_exists()

        _last_machine_update_ts = 0
        while True:
            # Get used pods and namespaces
            current_pods, namespaces = self._get_used_pods()

            # just in case there are no pods, make sure we look at our base namespace
            namespaces.add(self.namespace)

            # check if have pod limit, then check if we hit it.
            if self.max_pods_limit:
                if current_pods >= self.max_pods_limit:
                    print("Maximum {} limit reached {}/{}, sleeping for {:.1f} seconds".format(
                        self.kind, current_pods, self.max_pods_limit, self._polling_interval))
                    # delete old completed / failed pods
                    self._cleanup_old_pods(namespaces, f" due to {self.kind} limit")
                    # go to sleep
                    sleep(self._polling_interval)
                    continue

            # iterate over queues (priority style, queues[0] is highest)
            # print("debug> iterating over queues")
            for queue in queues:
                # delete old completed / failed pods
                self._cleanup_old_pods(namespaces, extra_msg="Cleanup cycle {cmd}")

                if self.check_if_suspended():
                    print("Agent is suspended, sleeping for {:.1f} seconds".format(self._polling_interval))
                    sleep(self._polling_interval)
                    break

                if not self.check_if_schedulable(queue):
                    continue

                # get next task in queue
                try:
                    # print(f"debug> getting tasks for queue {queue}")
                    response = self._get_next_task(queue=queue, get_task_info=self._impersonate_as_task_owner)
                except Exception as e:
                    print("Warning: Could not access task queue [{}], error: {}".format(queue, e))
                    continue
                else:
                    if not response:
                        continue
                    try:
                        task_id = response["entry"]["task"]
                    except (KeyError, TypeError, AttributeError):
                        print("No tasks in queue {}".format(queue))
                        continue

                    print('Received task {} from queue {}'.format(task_id, queue))

                    task_session = None
                    if self._impersonate_as_task_owner:
                        try:
                            task_user = response["task_info"]["user"]
                            task_company = response["task_info"]["company"]
                        except (KeyError, TypeError, AttributeError):
                            print("Error: cannot retrieve owner user for the task '{}', skipping".format(task_id))
                            continue

                        task_session = self.get_task_session(task_user, task_company)
                        if not task_session:
                            print(
                                "Error: Could not login as the user '{}' for the task '{}', skipping".format(
                                    task_user, task_id
                                )
                            )
                            continue

                    events_service.send_log_events(
                        self.worker_id,
                        task_id=task_id,
                        lines="task {} pulled from {} by worker {}".format(
                            task_id, queue, self.worker_id
                        ),
                        level="INFO",
                        session=task_session,
                    )

                    self.report_monitor(ResourceMonitor.StatusReport(queues=queues, queue=queue, task=task_id))
                    self.run_one_task(queue, task_id, worker_params, task_session)
                    self.report_monitor(ResourceMonitor.StatusReport(queues=self.queues))
                    break
            else:
                # sleep and retry polling
                print("No tasks in Queues, sleeping for {:.1f} seconds".format(self._polling_interval))
                sleep(self._polling_interval)

            if self._session.config["agent.reload_config"]:
                self.reload_config()

    def k8s_daemon(self, queue, **kwargs):
        """
        Start the k8s Glue service.
        This service will be pulling tasks from *queue* and scheduling them for execution using kubectl.
        Notice all scheduled tasks are pushed back into K8S_PENDING_QUEUE,
        and popped when execution actually starts. This creates full visibility into the k8s scheduler.
        Manually popping a task from the K8S_PENDING_QUEUE,
        will cause the k8s scheduler to skip the execution once the scheduled tasks needs to be executed

        :param list(str) queue: queue name to pull from
        """
        queues = queue if isinstance(queue, (list, tuple)) else ([queue] if queue else None)
        return self.daemon(
            queues=[ObjectID(name=q) for q in queues] if queues else None,
            log_level=logging.INFO, foreground=True, docker=False, **kwargs,
        )

    def _get_next_task(self, queue, get_task_info):
        return get_next_task(
            self._session, queue=queue, get_task_info=get_task_info
        )

    def _resolve_template(self, task_session, task_data, queue, task_id):
        if self.template_dict:
            return deepcopy(self.template_dict)

    @classmethod
    def get_ssh_server_bash(cls, ssh_port_number):
        return ' ; '.join(line.format(port=ssh_port_number) for line in cls.BASH_INSTALL_SSH_CMD)

    @staticmethod
    def _merge_containers(c1, c2):
        def merge_env(k, d1, d2, not_set):
            if k != "env":
                return not_set
            # Merge environment lists, second list overrides first
            return list({
                item['name']: item for envs in (d1, d2) for item in envs
            }.values())

        return merge_dicts(
            c1, c2, custom_merge_func=merge_env
        )

    @staticmethod
    def _safe_k8s_label_value(value):
        """ Conform string to k8s standards for a label value """
        value = value.lower().strip()
        value = re.sub(r'^[^A-Za-z0-9]+', '', value)  # strip leading non-alphanumeric chars
        value = re.sub(r'[^A-Za-z0-9]+$', '', value)  # strip trailing non-alphanumeric chars
        value = re.sub(r'\W+', '-', value)  # allow only word chars (this removed "." which is supported, but nvm)
        value = re.sub(r'_+', '-', value)  # "_" is not allowed as well
        value = re.sub(r'-+', '-', value)  # don't leave messy "--" after replacing previous chars
        return value[:63]
