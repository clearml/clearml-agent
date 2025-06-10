import shlex

from clearml_agent.helper.environment import EnvEntry

ENV_START_AGENT_SCRIPT_PATH = EnvEntry("CLEARML_K8S_GLUE_START_AGENT_SCRIPT_PATH", default="~/__start_agent__.sh")
"""
Script path to use when creating the bash script to run the agent inside the scheduled pod's docker container. 
Script will be appended to the specified file.
"""

ENV_DEFAULT_EXECUTION_AGENT_ARGS = EnvEntry("K8S_GLUE_DEF_EXEC_AGENT_ARGS", default="--full-monitoring --require-queue")
ENV_POD_AGENT_INSTALL_ARGS = EnvEntry("K8S_GLUE_POD_AGENT_INSTALL_ARGS", default="", lstrip=False)
ENV_POD_MONITOR_LOG_BATCH_SIZE = EnvEntry("K8S_GLUE_POD_MONITOR_LOG_BATCH_SIZE", default=5, converter=int)
ENV_POD_MONITOR_DISABLE_ENQUEUE_ON_PREEMPTION = EnvEntry(
    "K8S_GLUE_POD_MONITOR_DISABLE_ENQUEUE_ON_PREEMPTION", default=False, converter=bool
)

ENV_POD_USE_IMAGE_ENTRYPOINT = EnvEntry("K8S_GLUE_POD_USE_IMAGE_ENTRYPOINT", default=False, converter=bool)
"""
Do not inject a cmd and args to the container's image when building the k8s template (depend on the built-in image
entrypoint)
"""

ENV_KUBECTL_IGNORE_ERROR = EnvEntry("K8S_GLUE_IGNORE_KUBECTL_ERROR", default=None)
"""
Ignore kubectl errors matching this string pattern (allows ignoring warnings sent on stderr while 
kubectl actually works and starts the pod)
"""

ENV_DEFAULT_SCHEDULER_QUEUE_TAGS = EnvEntry(
    "K8S_GLUE_DEFAULT_SCHEDULER_QUEUE_TAGS", default=["k8s-glue"], converter=shlex.split
)
