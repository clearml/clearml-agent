"""
Regression test for DAG-18982 - "Agent regression when executing with conda
due to updating command environ".

At the task launch site the worker calls ``command.update_envs(os.environ)`` on
whatever ``package_api.get_python_command()`` returns. For a pip / venv task that
is an ``Argv`` (``python ...``); for a conda task it is a ``CommandSequence``
(``source .../conda.sh && conda activate env && python ...``).

``Argv`` has always had ``update_envs``, but ``CommandSequence`` did not, so the
polymorphic call raised ``AttributeError: 'CommandSequence' object has no
attribute 'update_envs'`` and every conda task failed right at "Starting Task
Execution". ``CommandSequence`` is always truthy (no ``__bool__`` / ``__len__``),
so the earlier ``if command: command.update_envs(...)`` guard did not spare it.

The fix adds ``CommandSequence.update_envs`` which forwards the envs to each of
its ``Argv`` sub-commands, restoring the interface parity the launch path relies
on.
"""
import os

import pytest

from clearml_agent.helper.process import Argv, CommandSequence


def _conda_style_command():
    # Mirrors CondaAPI.get_python_command():
    #   CommandSequence(self.source, self.pip.get_python_command(extra))
    # i.e. a "source/activate" prefix followed by the python invocation.
    return CommandSequence(
        Argv("source", "/opt/conda/etc/profile.d/conda.sh", "&&", "conda", "activate", "env"),
        Argv("/opt/conda/envs/env/bin/python", "-u", "-c", "print(1)"),
    )


def test_argv_update_envs_baseline():
    # Control: the pip / venv path returns an Argv, which was never affected by
    # the regression. This pins the behavior the conda path must match.
    cmd = Argv("/usr/bin/python", "-c", "print(1)")
    cmd.update_envs({"CLEARML_TASK_ID": "abc123"})
    assert cmd._env["CLEARML_TASK_ID"] == "abc123"


def test_command_sequence_exposes_update_envs():
    # The core contract the regression broke: CommandSequence must expose the
    # same update_envs interface as Argv because the worker calls it
    # polymorphically on the result of package_api.get_python_command().
    assert hasattr(CommandSequence, "update_envs")


def test_command_sequence_update_envs_propagates_to_each_subcommand():
    cmd = _conda_style_command()
    cmd.update_envs({"CLEARML_TASK_ID": "abc123", "FOO": "bar"})
    assert cmd.commands  # sanity: the conda command really is a sequence
    for sub in cmd.commands:
        assert isinstance(sub, Argv)
        assert sub._env["CLEARML_TASK_ID"] == "abc123"
        assert sub._env["FOO"] == "bar"


def test_command_sequence_update_envs_noop_on_empty():
    # Empty / falsy envs must be a no-op, exactly like Argv.update_envs.
    cmd = _conda_style_command()
    cmd.update_envs(None)
    cmd.update_envs({})
    for sub in cmd.commands:
        assert not sub._env


def test_worker_launch_call_does_not_crash_for_any_package_manager():
    # The exact call the worker makes at the launch site is
    # ``command.update_envs(os.environ)``. It must succeed for BOTH the pip / venv
    # command (Argv) and the conda command (CommandSequence).
    for cmd in (Argv("/usr/bin/python", "-c", "pass"), _conda_style_command()):
        cmd.update_envs(os.environ)  # must not raise AttributeError


def test_pre_fix_behavior_would_crash_for_conda(monkeypatch):
    # Documents the pre-fix state for an explicit before/after contrast: with
    # CommandSequence.update_envs removed (as it was before the fix), the launch
    # path's polymorphic call raises AttributeError for a conda command - which
    # is exactly the crash DAG-18982 reports. The Argv (pip / venv) path is
    # unaffected either way.
    monkeypatch.delattr(CommandSequence, "update_envs", raising=False)

    Argv("/usr/bin/python", "-c", "pass").update_envs(os.environ)  # pip path: fine

    with pytest.raises(AttributeError):
        _conda_style_command().update_envs(os.environ)  # conda path: the regression
