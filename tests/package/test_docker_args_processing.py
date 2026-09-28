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


def test_merge_docker_args_empty_entrypoint_space_form_kept():
    # --entrypoint "" clears the image's built-in ENTRYPOINT without overriding the agent's own
    # startup command, so it's harmless and must not be stripped even though "entrypoint" is
    # forbidden.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=[],
        extra_docker_arguments=["--entrypoint", "", "--ipc=host"],
    )
    assert stripped_switches == []
    assert merged == ["--entrypoint", "", "--ipc=host"]


def test_merge_docker_args_empty_entrypoint_equals_form_kept():
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["--entrypoint=", "-v", "/x:/y"],
        extra_docker_arguments=[],
    )
    assert stripped_switches == []
    assert merged == ["--entrypoint=", "-v", "/x:/y"]


def test_merge_docker_args_dangling_entrypoint_stripped():
    # --entrypoint as the very last token has no value at all (not even an empty one) — this is
    # conservatively treated as non-empty and still stripped, since we can't confirm it's harmless.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=[],
        extra_docker_arguments=["-v", "/x:/y", "--entrypoint"],
    )
    assert stripped_switches == ["entrypoint"]
    assert "--entrypoint" not in merged
    assert "-v" in merged and "/x:/y" in merged


def test_merge_docker_args_non_empty_entrypoint_still_stripped_alongside_empty_one():
    # one source has a real override, the other an empty one — the real one still wins and
    # causes stripping from both sources (current all-or-nothing per-switch-name contract).
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["--entrypoint", "", "-v", "/x:/y"],
        extra_docker_arguments=["--entrypoint=/bin/echo", "--ipc=host"],
    )
    assert stripped_switches == ["entrypoint"]
    assert not any(a.startswith("--entrypoint") for a in merged)
    assert "-v" in merged and "/x:/y" in merged
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


def test_merge_docker_args_empty_value_exemption_is_entrypoint_only():
    # The empty-value carve-out is entrypoint-specific. A different block-listed switch
    # (here 'user') must still be stripped even when its value is empty, so an admin's
    # block-list keeps enforcing and cannot be bypassed with a bare `--user ""`.
    for extra in (["--user", "", "--ipc=host"], ["--user=", "--ipc=host"]):
        merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
            config=_DummyConfig({"agent.forbidden_docker_args": ["entrypoint", "user"]}),
            task_docker_arguments=[],
            extra_docker_arguments=extra,
        )
        assert stripped_switches == ["user"]
        assert not any(a.startswith("--user") for a in merged)
        assert "" not in merged
        assert "--ipc=host" in merged


def test_merge_docker_args_stray_empty_token_is_dropped():
    # A stray "" token that is not an --entrypoint value would become a bare positional at
    # `docker run` (docker reads it as an empty image reference -> "invalid reference format").
    # It must be dropped, matching the long-standing behavior for such junk tokens.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=["-v", "/x:/y", "", "--rm"],
        extra_docker_arguments=["--privileged", "", "--ipc=host"],
    )
    assert stripped_switches == []
    assert "" not in merged
    assert merged.count("") == 0
    for expected in ("-v", "/x:/y", "--rm", "--privileged", "--ipc=host"):
        assert expected in merged


def test_merge_docker_args_double_empty_entrypoint_keeps_single_value():
    # Only the first "" is --entrypoint's value; a second trailing "" is a stray token that
    # would orphan into a bare positional at docker run. Keep the entrypoint's empty value and
    # drop the orphan so the command stays well-formed.
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=[],
        extra_docker_arguments=["--entrypoint", "", ""],
    )
    assert stripped_switches == []
    assert merged == ["--entrypoint", ""]


def test_merge_docker_args_whitespace_entrypoint_is_a_real_override_and_stripped():
    # A whitespace-only value is a real (broken) entrypoint override, not a request to clear the
    # image default, so both forms are stripped - the empty-value exemption applies to "" only.
    for task in (["--entrypoint=   ", "-v", "/x:/y"], ["--entrypoint", "   ", "-v", "/x:/y"]):
        merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
            config=_DummyConfig(_SHIPPED_CONF),
            task_docker_arguments=task,
            extra_docker_arguments=[],
        )
        assert stripped_switches == ["entrypoint"]
        assert not any(a.strip().startswith("--entrypoint") for a in merged)
        assert "-v" in merged and "/x:/y" in merged


def test_merge_docker_args_empty_value_of_non_forbidden_switch_is_dropped():
    # --label is not block-listed, so it is not exempted; its stray space-form "" is dropped like
    # any other empty token (only a kept --entrypoint's empty value is preserved).
    merged, stripped_switches = DockerArgsSanitizer.merge_docker_args(
        config=_DummyConfig(_SHIPPED_CONF),
        task_docker_arguments=[],
        extra_docker_arguments=["--label", "", "--rm"],
    )
    assert stripped_switches == []
    assert merged == ["--label", "--rm"]
