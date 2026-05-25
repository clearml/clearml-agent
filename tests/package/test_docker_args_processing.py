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


class _DummyConfig:
    """Minimal stand-in for session.config, supports .get(key, default)."""
    def __init__(self, overrides=None):
        self._overrides = overrides or {}

    def get(self, key, default=None):
        return self._overrides.get(key, default)


# Mirrors the value shipped in agent.conf / docs/clearml.conf so tests reflect real usage.
_SHIPPED_CONF = {"agent.forbidden_docker_args": ["entrypoint"]}


def test_merge_docker_args_strips_entrypoint_from_extra_space_form():
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["-v", "/x:/y"],
        extra_docker_arguments=["--entrypoint", "/bin/echo", "--ipc=host"],
    )
    assert stripped_switches == ["entrypoint"]
    assert "--entrypoint" not in merged
    assert "/bin/echo" not in merged
    assert "--ipc=host" in merged
    assert "-v" in merged and "/x:/y" in merged


def test_merge_docker_args_strips_entrypoint_from_task_equals_form():
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["--entrypoint=/bin/echo", "-v", "/x:/y"],
        extra_docker_arguments=[],
    )
    assert stripped_switches == ["entrypoint"]
    assert "--entrypoint=/bin/echo" not in merged
    assert "-v" in merged and "/x:/y" in merged


def test_merge_docker_args_strips_entrypoint_from_both_sources():
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["--entrypoint", "/bin/sh", "-v", "/x:/y"],
        extra_docker_arguments=["--entrypoint=/bin/echo", "--ipc=host"],
    )
    # Both sources had --entrypoint; deduped to a single switch name in the result.
    assert stripped_switches == ["entrypoint"]
    assert not any(a.startswith("--entrypoint") for a in merged)
    assert "/bin/sh" not in merged and "/bin/echo" not in merged
    assert "-v" in merged and "/x:/y" in merged
    assert "--ipc=host" in merged


def test_merge_docker_args_no_entrypoint_unchanged():
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["-v", "/x:/y"],
        extra_docker_arguments=["--ipc=host"],
    )
    assert stripped_switches == []
    assert "-v" in merged and "/x:/y" in merged
    assert "--ipc=host" in merged


def test_merge_docker_args_missing_conf_key_strips_nothing():
    # If the conf key is absent entirely, the code-level fallback is [] —
    # nothing is stripped, even --entrypoint. This pins the new contract:
    # the shipped conf is the source of truth for the default.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(),  # no overrides — conf key is missing
        task_docker_arguments=[],
        extra_docker_arguments=["--entrypoint", "/bin/echo"],
    )
    assert stripped_switches == []
    assert "--entrypoint" in merged
    assert "/bin/echo" in merged


def test_merge_docker_args_forbidden_list_disabled_via_config():
    # Setting agent.forbidden_docker_args to [] disables stripping entirely.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig({"agent.forbidden_docker_args": []}),
        task_docker_arguments=[],
        extra_docker_arguments=["--entrypoint", "/bin/echo", "--ipc=host"],
    )
    assert stripped_switches == []
    assert "--entrypoint" in merged
    assert "/bin/echo" in merged
    assert "--ipc=host" in merged


def test_merge_docker_args_forbidden_list_custom_switch():
    # A custom forbidden list strips the configured switch and leaves --entrypoint alone.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig({"agent.forbidden_docker_args": ["user"]}),
        task_docker_arguments=[],
        extra_docker_arguments=["--entrypoint", "/bin/echo", "--user", "root", "--ipc=host"],
    )
    assert stripped_switches == ["user"]
    assert "--entrypoint" in merged
    assert "/bin/echo" in merged
    assert "--user" not in merged
    assert "root" not in merged
    assert "--ipc=host" in merged


def test_merge_docker_args_forbidden_list_multiple_switches_each_with_value():
    # Multiple forbidden switches, each with its own value, mixed space- and equals-form.
    # The state machine must reset between switches so that a kept switch's value isn't dropped.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig({"agent.forbidden_docker_args": ["entrypoint", "user"]}),
        task_docker_arguments=[],
        extra_docker_arguments=[
            "--entrypoint", "/bin/echo",
            "--user=root",
            "-v", "/x:/y",
            "--ipc=host",
        ],
    )
    assert sorted(stripped_switches) == ["entrypoint", "user"]
    assert "--entrypoint" not in merged
    assert "/bin/echo" not in merged
    assert "--user=root" not in merged
    assert "-v" in merged and "/x:/y" in merged  # value of kept switch survives
    assert "--ipc=host" in merged


def test_merge_docker_args_entrypoint_followed_by_extra_tokens():
    # docker's --entrypoint consumes exactly one value; args for the entrypoint
    # program itself belong AFTER the image, not after --entrypoint. If a user
    # mistakenly inlines them here, we strip --entrypoint + its single value and
    # leave the trailing tokens untouched (they'll fail at docker run, which is
    # the user's bug — this test pins the behavior so a future refactor doesn't
    # silently start swallowing the trailing tokens too).
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=[],
        extra_docker_arguments=["--entrypoint", "/bin/bash", "-c", "echo hello"],
    )
    assert stripped_switches == ["entrypoint"]
    assert "--entrypoint" not in merged
    assert "/bin/bash" not in merged
    assert "-c" in merged
    assert "echo hello" in merged
