"""
Tests for K8sIntegration._create_bash_script_for_container.

Regression coverage for DAG ticket: a customer init script containing a non-ASCII
character (e.g. U+2713 "✓") used to crash the glue with a UnicodeEncodeError
before the pod was applied, because the script was encoded with the 'ascii' codec.
The script is now encoded as UTF-8, which can represent any character.
"""
import base64
import re
import unittest

from clearml_agent.glue.k8s import K8sIntegration


class _Config:
    """Minimal stand-in for session.config, supports .get(key, default)."""
    def get(self, key, default=None):
        return default


class _Session:
    config = _Config()


def _make_glue(container_bash_script):
    """Build a K8sIntegration instance without running its heavy __init__,
    wiring only the attributes _create_bash_script_for_container touches."""
    glue = object.__new__(K8sIntegration)
    glue.container_bash_script = container_bash_script
    glue.extra_bash_init_script = None
    glue._session = _Session()
    return glue


def _decode_start_agent_script(extra_bash_commands):
    """Extract and decode the base64 blob the method appends as the last command."""
    last_cmd = extra_bash_commands[-1]
    match = re.search(r"echo '([^']+)' \| base64 --decode", last_cmd)
    assert match, "could not find the base64 payload in: {}".format(last_cmd)
    return base64.b64decode(match.group(1)).decode("utf-8")


class CreateBashScriptForContainerTest(unittest.TestCase):
    def test_non_ascii_docker_bash_does_not_raise_and_round_trips(self):
        # U+2713 CHECK MARK is exactly the character from the reported crash.
        glue = _make_glue(["# task {task_id}", "{extra_docker_bash_script}"])

        extra_bash_commands, extra_envs = glue._create_bash_script_for_container(
            task_id="task-123",
            docker_bash="echo ✓ setup done",
            clearml_conf_create_script=["create_conf_cmd"],
        )

        self.assertEqual(extra_envs, {})
        # clearml.conf commands are preserved and kept before the start-agent command
        self.assertEqual(extra_bash_commands[0], "create_conf_cmd")

        decoded = _decode_start_agent_script(extra_bash_commands)
        self.assertTrue(decoded.startswith("#!/bin/bash"))
        self.assertIn("echo ✓ setup done", decoded)
        self.assertIn("# task task-123", decoded)

    def test_ascii_docker_bash_still_works(self):
        glue = _make_glue(["{extra_docker_bash_script}"])

        extra_bash_commands, _ = glue._create_bash_script_for_container(
            task_id="task-abc",
            docker_bash="echo hello world",
            clearml_conf_create_script=None,
        )

        decoded = _decode_start_agent_script(extra_bash_commands)
        self.assertIn("echo hello world", decoded)

    def test_various_non_ascii_characters_round_trip(self):
        # A spread of non-ASCII: emoji, accented latin, CJK, and the original check mark.
        payload = "echo ✓ café 你好 \U0001f680"
        glue = _make_glue(["{extra_docker_bash_script}"])

        extra_bash_commands, _ = glue._create_bash_script_for_container(
            task_id="t",
            docker_bash=payload,
            clearml_conf_create_script=[],
        )

        decoded = _decode_start_agent_script(extra_bash_commands)
        self.assertIn(payload, decoded)

    def test_ascii_encoding_would_have_failed(self):
        # Guards the intent of the fix: the reported character is genuinely non-ASCII,
        # so the previous 'ascii' encode path was the root cause.
        with self.assertRaises(UnicodeEncodeError):
            "echo ✓".encode("ascii")


if __name__ == "__main__":
    unittest.main()
