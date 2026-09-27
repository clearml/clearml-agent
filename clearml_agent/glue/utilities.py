import functools

from subprocess import DEVNULL

from clearml_agent.helper.process import get_bash_output as _get_bash_output


def get_path(d, *path, default=None):
    try:
        return functools.reduce(
            lambda a, b: a[b], path, d
        )
    except (IndexError, KeyError):
        return default


def get_task_id_from_resource_name(resource_name):
    """The task id from the name of the pod/job created for it: <prefix>-<task id>[-<node rank>]"""
    prefix, _, value = resource_name.rpartition('-')
    if len(value) > 4:
        return value
    # we assume this is a multi-node rank x (>0) pod
    return prefix.rpartition('-')[-1] or value


def get_bash_output(cmd, stderr=DEVNULL, raise_error=False):
    return _get_bash_output(cmd, stderr=stderr, raise_error=raise_error)
