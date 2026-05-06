"""
Test handling of jupyter notebook tasks.
Logging is enabled in `clearml_agent/tests/pytest.ini`. Search for `pytest live logging` for more info.
"""
import logging
import re
import select
import subprocess
import time
from contextlib import contextmanager
from typing import Iterator, ContextManager, Sequence, IO, Text
from uuid import uuid4

from clearml_agent.backend_api.services import tasks
from clearml_agent.backend_api.session.client import APIClient
from pathlib2 import Path
from pytest import fixture

from clearml_agent.helper.process import Argv

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)


DEFAULT_TASK_ARGS = {"type": "testing", "name": "test", "input": {"view": {}}}
HERE = Path(__file__).resolve().parent
SHORT_TIMEOUT = 30


@fixture(scope="session")
def client():
    return APIClient()


@contextmanager
def create_task(client, **params):
    """
    Create task in backend
    """
    log.info("creating new task")
    task = client.tasks.create(**params)
    try:
        yield task
    finally:
        log.info("deleting task, id=%s", task.id)
        task.delete(force=True)


def select_read(file_obj, timeout):
    return select.select([file_obj], [], [], timeout)[0]


def run_task(task):
    return Argv("clearml_agent", "--debug", "worker", "execute", "--id", task.id)


@contextmanager
def iterate_output(timeout, command):
    # type: (int, Argv) -> ContextManager[Iterator[Text]]
    """
    Run `command` in a subprocess and return a contextmanager of an iterator
    over its output's lines. If `timeout` seconds have passed, iterator ends and
    the process is killed.
    :param timeout: maximum amount of time to wait for command to end
    :param command: command to run
    """
    log.info("running: %s", command)
    process = command.call_subprocess(
        subprocess.Popen, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        yield _iterate_output(timeout, process)
    finally:
        status = process.poll()
        if status is not None:
            log.info("command %s terminated, status: %s", command, status)
        else:
            log.info("killing command %s", command)
            process.kill()


def _iterate_output(timeout, process):
    # type: (int, subprocess.Popen) -> Iterator[Text]
    """
    Return an iterator over process's output lines.
    over its output's lines.
    If `timeout` seconds have passed, iterator ends and the process is killed.
    :param timeout: maximum amount of time to wait for command to end
    :param process: process to iterate over its output lines
    """
    start = time.time()
    exit_loop = []  # type: Sequence[IO]

    def loop_helper(file_obj):
        # type: (IO) -> Sequence[IO]
        diff = timeout - (time.time() - start)
        if diff <= 0:
            return exit_loop
        return select_read(file_obj, timeout=diff)

    buffer = ""

    for output, in iter(lambda: loop_helper(process.stdout), exit_loop):
        try:
            added = output.read(1024).decode("utf8")
        except EOFError:
            if buffer:
                yield buffer
            return
        buffer += added
        lines = buffer.split("\n", 1)

        while len(lines) > 1:
            line, buffer = lines
            log.debug("--- %s", line)
            yield line
            lines = buffer.split("\n", 1)


def search_lines(lines, search_for, error):
    # type: (Iterator[Text], Text, Text) -> None
    """
    Fail test if `search_for` string appears nowhere in `lines`.
    Consumes `lines` up to point where `search_for` is found.
    :param lines: lines to search in
    :param search_for: string to search lines for
    :param error: error to show if not found
    """
    for line in lines:
        if search_for in line:
            break
    else:
        assert False, error


def search_lines_pattern(lines, pattern, error):
    # type: (Iterator[Text], Text, Text) -> None
    """
    Like `search_lines` but searches for a pattern.
    :param lines: lines to search in
    :param pattern: pattern to search lines for
    :param error: error to show if not found
    """
    for line in lines:
        if re.search(pattern, line):
            break
    else:
        assert False, error


def test_entry_point_warning(client):
    """
   non-empty script.entry_point should output a warning
    """
    with create_task(
        client,
        script=tasks.Script(diff="print('hello')", entry_point="foo.py", repository=""),
        **DEFAULT_TASK_ARGS
    ) as task, iterate_output(SHORT_TIMEOUT, run_task(task)) as output:
        for line in output:
            if "found non-empty script.entry_point" in line:
                break
        else:
            assert False, "did not find warning in output"


def test_run_no_dirs(client):
    """
    The arbitrary `code` directory should be selected when there is no `script.repository`
    """
    uuid = uuid4().hex
    script = "print('{}')".format(uuid)
    with create_task(
        client,
        script=tasks.Script(diff=script, entry_point="", repository="", working_dir=""),
        **DEFAULT_TASK_ARGS
    ) as task, iterate_output(SHORT_TIMEOUT, run_task(task)) as output:
        search_lines(
            output,
            "found literal script",
            "task was not recognized as a literal script",
        )
        search_lines_pattern(
            output,
            r"selected execution directory:.*code",
            r"did not selected empty `code` dir as execution dir",
        )
        search_lines(output, uuid, "did not find uuid {!r} in output".format(uuid))


def test_run_working_dir(client):
    """
    Literal script tasks should respect `working_dir`
    """
    uuid = uuid4().hex
    script = "print('{}')".format(uuid)
    with create_task(
        client,
        script=tasks.Script(
            diff=script,
            entry_point="",
            repository="git@bitbucket.org:seematics/roee_test_git.git",
            working_dir="space dir",
        ),
        **DEFAULT_TASK_ARGS
    ) as task, iterate_output(120, run_task(task)) as output:
        search_lines(
            output,
            "found literal script",
            "task was not recognized as a literal script",
        )
        search_lines_pattern(
            output,
            r"selected execution directory:.*space dir",
            r"did not selected working_dir as set in execution_info",
        )
        search_lines(output, uuid, "did not find uuid {!r} in output".format(uuid))


def test_regular_task(client):
    """
    Test a plain old task
    """
    with create_task(
        client,
        script=tasks.Script(
            entry_point="noop.py",
            repository="git@bitbucket.org:seematics/roee_test_git.git",
        ),
        **DEFAULT_TASK_ARGS
    ) as task, iterate_output(SHORT_TIMEOUT, run_task(task)) as output:
        message = "Done"
        search_lines(
            output, message, "did not reach dummy output message {}".format(message)
        )


def test_regular_task_nested(client):
    """
    `entry_point` should be relative to `working_dir` if present
    """
    with create_task(
        client,
        script=tasks.Script(
            entry_point="noop_nested.py",
            working_dir="no_reqs",
            repository="git@bitbucket.org:seematics/roee_test_git.git",
        ),
        **DEFAULT_TASK_ARGS
    ) as task, iterate_output(SHORT_TIMEOUT, run_task(task)) as output:
        message = "Done"
        search_lines(
            output, message, "did not reach dummy output message {}".format(message)
        )
