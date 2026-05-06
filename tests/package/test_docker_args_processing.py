"""
Test docker args processing
"""
import logging

from clearml_agent.helper.docker_args import DockerArgsSanitizer


logging.getLogger("urllib3").setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)

extra_args = "--network=host --privileged=false --ipc host -v /host/b:/docker/b"
override_switches = ["privileged", "security-opt", "network", "ipc"]
task_args = "--network host -it  --privileged  -v=/a/b:/c/d -v /c/e:/f/g --rm"


def test_extra_docker_args_processing():
    switches = DockerArgsSanitizer.get_list_of_switches(extra_args.split())
    switches = list(set(switches) & set(override_switches))
    filtered_task_docker_args = DockerArgsSanitizer.filter_switches(task_args.split(), switches)
    print("\nextra_args", extra_args)
    print("extra_args switches", switches)
    print("task_args", task_args)
    print("filtered:", filtered_task_docker_args)
    assert filtered_task_docker_args == ['-it', '-v=/a/b:/c/d', '-v', '/c/e:/f/g', '--rm']


def test_task_docker_args_processing():
    switches = DockerArgsSanitizer.get_list_of_switches(task_args.split())
    switches = list(set(switches) & set(override_switches))
    filtered_task_docker_args = DockerArgsSanitizer.filter_switches(extra_args.split(), switches)
    print("\nextra_args", task_args)
    print("extra_args switches", switches)
    print("task_args", extra_args)
    print("filtered:", filtered_task_docker_args)
    assert filtered_task_docker_args == ['--ipc', 'host', '-v', '/host/b:/docker/b']
