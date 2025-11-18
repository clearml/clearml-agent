import re
import shlex
from typing import Tuple, List, TYPE_CHECKING, Optional
from urllib.parse import urlunparse, urlparse

from clearml_agent.definitions import (
    ENV_AGENT_GIT_PASS,
    ENV_AGENT_SECRET_KEY,
    ENV_AWS_SECRET_KEY,
    ENV_AZURE_ACCOUNT_KEY,
    ENV_AGENT_AUTH_TOKEN,
    ENV_DOCKER_IMAGE,
    ENV_DOCKER_ARGS_HIDE_ENV,
    ENV_FORCE_HOST_MACHINE_IP,
)
from clearml_agent.helper.sdk_client.utilities.networking import get_private_ip
from clearml_agent.helper.os.networking import TcpPorts
from clearml_agent.helper.custom_template import CustomTemplate, TemplateResolver

if TYPE_CHECKING:
    from clearml_agent.session import Session


def sanitize_urls(s: str) -> Tuple[str, bool]:
    """
    Replaces passwords in URLs with asterisks.
    Returns the sanitized string and a boolean indicating whether sanitation was performed.
    """
    regex = re.compile("^([^:]*:)[^@]+(.*)$")
    tokens = re.split(r"\s", s)
    changed = False
    for k in range(len(tokens)):
        if "@" in tokens[k]:
            res = urlparse(tokens[k])
            if regex.match(res.netloc):
                changed = True
                tokens[k] = urlunparse((
                    res.scheme,
                    regex.sub("\\1********\\2", res.netloc),
                    res.path,
                    res.params,
                    res.query,
                    res.fragment
                ))
    return " ".join(tokens) if changed else s, changed


class DockerArgsSanitizer:
    _machine_ip = None

    @classmethod
    def sanitize_docker_command(cls, session, docker_command):
        # type: (Session, List[str]) -> List[str]
        if not docker_command:
            return docker_command

        enabled = (
            session.config.get('agent.hide_docker_command_env_vars.enabled', False) or ENV_DOCKER_ARGS_HIDE_ENV.get()
        )
        if not enabled:
            return docker_command

        keys = set(session.config.get('agent.hide_docker_command_env_vars.extra_keys', []))
        if ENV_DOCKER_ARGS_HIDE_ENV.get():
            keys.update(shlex.split(ENV_DOCKER_ARGS_HIDE_ENV.get().strip()))
        keys.update(
            ENV_AGENT_GIT_PASS.vars,
            ENV_AGENT_SECRET_KEY.vars,
            ENV_AWS_SECRET_KEY.vars,
            ENV_AZURE_ACCOUNT_KEY.vars,
            ENV_AGENT_AUTH_TOKEN.vars,
        )

        parse_embedded_urls = bool(session.config.get(
            'agent.hide_docker_command_env_vars.parse_embedded_urls', True
        ))

        skip_next = False
        result = docker_command[:]
        for i, item in enumerate(docker_command):
            if skip_next:
                skip_next = False
                continue
            try:
                if item in ("-e", "--env"):
                    key, sep, val = result[i + 1].partition("=")
                    if not sep:
                        continue
                    if key in ENV_DOCKER_IMAGE.vars:
                        # special case - this contains a complete docker command
                        val = " ".join(cls.sanitize_docker_command(session, re.split(r"\s", val)))
                    elif key in keys:
                        val = "********"
                    elif parse_embedded_urls:
                        val = sanitize_urls(val)[0]
                    result[i + 1] = "{}={}".format(key, val)
                    skip_next = True
                elif parse_embedded_urls and not item.startswith("-"):
                    item, changed = sanitize_urls(item)
                    if changed:
                        result[i] = item
            except (KeyError, TypeError):
                pass

        return result

    @staticmethod
    def get_list_of_switches(docker_args: List[str]) -> List[str]:
        args = []
        for token in docker_args:
            if token.strip().startswith("-"):
                args += [token.strip().split("=")[0].lstrip("-")]

        return args

    @staticmethod
    def filter_switches(
            docker_args: List[str],
            exclude_switches: List[str] = None,
            include_switches: List[str] = None
    ) -> List[str]:

        assert not (include_switches and exclude_switches), "Either include_switches or exclude_switches but not both"

        # shortcut if we are sure we have no matches
        if not include_switches and (
                not exclude_switches or not any("-{}".format(s) in " ".join(docker_args) for s in exclude_switches)):
            return docker_args

        args = []
        in_switch_args = True if not include_switches else False

        for token in docker_args:
            if token.strip().startswith("-"):
                if "=" in token:
                    switch = token.strip().split("=")[0]
                    in_switch_args = False
                else:
                    switch = token
                    in_switch_args = True

                if not include_switches and switch.lstrip("-") in exclude_switches:
                    # if in excluded, skip the switch and following arguments
                    in_switch_args = False
                elif not exclude_switches and switch.lstrip("-") not in include_switches:
                    # if in excluded, skip the switch and following arguments
                    in_switch_args = False
                else:
                    args += [token]

            elif in_switch_args:
                args += [token]
            else:
                # this is the switch arguments we need to skip
                pass

        return args

    @staticmethod
    def merge_docker_args(config, task_docker_arguments: List[str], extra_docker_arguments: List[str]) -> List[str]:
        base_cmd = []
        # currently only resolving --network, --ipc, --privileged
        override_switches = config.get(
            "agent.protected_docker_extra_args",
            ["privileged", "security-opt", "network", "ipc"]
        )

        if config.get("agent.docker_args_extra_precedes_task", True):
            switches = []
            if extra_docker_arguments:
                switches = DockerArgsSanitizer.get_list_of_switches(extra_docker_arguments)
                switches = list(set(switches) & set(override_switches))
                base_cmd += [str(a) for a in extra_docker_arguments if a]
            if task_docker_arguments:
                docker_arguments = DockerArgsSanitizer.filter_switches(task_docker_arguments, switches)
                base_cmd += [a for a in docker_arguments if a]
        else:
            switches = []
            if task_docker_arguments:
                switches = DockerArgsSanitizer.get_list_of_switches(task_docker_arguments)
                switches = list(set(switches) & set(override_switches))
                base_cmd += [a for a in task_docker_arguments if a]
            if extra_docker_arguments:
                extra_docker_arguments = DockerArgsSanitizer.filter_switches(extra_docker_arguments, switches)
                base_cmd += [a for a in extra_docker_arguments if a]
        return base_cmd

    @staticmethod
    def resolve_port_mapping(config, docker_arguments: List[str]) -> Optional[tuple]:
        """
        If we have port mappings in the docker cmd, this function will do two things
        1. It will add an environment variable (CLEARML_AGENT_HOST_IP) with the host machines IP address
        2. it will return a runtime property ("_external_host_tcp_port_mapping") on the Task with the port mapping merged
        :param config:
        :param docker_arguments:
        :return: new docker commands with additional one to add docker
        (i.e. changing the ports if needed and adding the new env var), runtime property
        """
        if not docker_arguments:
            return None
        # make a copy we are going to change it
        docker_arguments = docker_arguments[:]
        port_mapping_filtered = [
            p for p in DockerArgsSanitizer.filter_switches(docker_arguments, include_switches=["p", "publish"])
            if p and p.strip()
        ]

        if not port_mapping_filtered:
            return None

        # test if network=host was requested, docker will ignore published ports anyhow, so no use in parsing them
        network_filtered = DockerArgsSanitizer.filter_switches(
            docker_arguments, include_switches=["network", "net"])
        network_filtered = [t for t in network_filtered if t.strip == "host" or "host" in t.split("=")]
        # if any network is configured, we ignore it, there is nothing we can do
        if network_filtered:
            return None

        # verifying available ports, remapping if necessary
        port_checks = TcpPorts()
        for i_p in range(len(port_mapping_filtered)):
            port_map = port_mapping_filtered[i_p]
            if not port_map.strip():
                continue
            # skip the flag
            if port_map.strip().startswith("-"):
                continue

            # todo: support udp?!
            # example: "8080:80/udp"
            if port_map.strip().split("/")[-1] == "udp":
                continue

            # either no type specified or tcp
            ports_host, ports_in = port_map.strip().split("/")[0].split(":")[-2:]
            # verify ports available
            port_range = int(ports_host.split("-")[0]), int(ports_host.split("-")[-1])+1
            if not all(port_checks.check_tcp_port_available(p) for p in range(port_range[0], port_range[1])):
                # we need to find a new range (this is a consecutive range)
                new_port_range = port_checks.find_port_range(port_range[1]-port_range[0])

                if not new_port_range:
                    # we could not find any, leave it as it?!
                    break

                # replace the ports,
                for i in range(len(docker_arguments)):
                    if docker_arguments[i].strip() != port_map.strip():
                        continue
                    slash_parts = port_map.strip().split("/")
                    colon_parts = slash_parts[0].split(":")
                    colon_parts[-2] = "{}-{}".format(new_port_range[0], new_port_range[-1]) \
                        if len(new_port_range) > 1 else str(new_port_range[0])

                    docker_arguments[i] = "/".join(slash_parts[1:] + [":".join(colon_parts)])
                    port_mapping_filtered[i_p] = docker_arguments[i]
                    break

        additional_cmd = []
        if not DockerArgsSanitizer._machine_ip:
            DockerArgsSanitizer._machine_ip = ENV_FORCE_HOST_MACHINE_IP.get() or get_private_ip(config)

        if DockerArgsSanitizer._machine_ip:
            additional_cmd += ["-e", "CLEARML_AGENT_HOST_IP={}".format(DockerArgsSanitizer._machine_ip)]

        # sanitize, remove ip/type
        ports = ",".join([":".join(t.strip().split("/")[0].split(":")[-2:])
                          for t in port_mapping_filtered if t.strip() and not t.strip().startswith("-")])

        # update Tasks runtime
        additional_task_runtime = {"_external_host_tcp_port_mapping": ports}

        return docker_arguments+additional_cmd, additional_task_runtime


class DockerArgsTemplateResolver(TemplateResolver):
    def resolve_from_docker_cmd(self, full_docker_cmd):
        if not full_docker_cmd or not self.task_session.check_min_api_version("2.20"):
            return full_docker_cmd

        # convert docker template arguments (i.e. ${CLEARML_} ) based on the current Task
        for i, token in enumerate(full_docker_cmd[:-1]):
            # skip the ones which are obviously not our prefix
            if CustomTemplate.delimiter not in token or CustomTemplate.prefix not in token:
                continue

            queue_id = self.queue_id or self.task_info.get("execution", {}).get("queue")
            queue_name = CustomTemplate.get_queue_name(self.task_session, queue_id)

            tmpl = CustomTemplate(token)
            # replace it
            try:
                full_docker_cmd[i] = tmpl.default_custom_substitute(
                    queue_id=queue_id,
                    queue_name=queue_name,
                    worker_id=self.worker_id,
                    task_info=self.task_info,
                    provider_info=self.provider_info,
                    config=self.task_session.config,
                    user_vaults=self.user_vaults,
                    user_info=self.user_info,
                )
            except Exception as ex:
                print("Failed parsing ClearML Template argument [{}] skipped: error ()".format(token, ex))

        return full_docker_cmd
