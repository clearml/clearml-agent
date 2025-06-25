""" CLEARML-AGENT Stdout Helper Functions  """
from __future__ import print_function, unicode_literals

import io
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from abc import ABCMeta
from collections import OrderedDict
from functools import total_ordering
from typing import Text, Dict, Any, Optional, AnyStr, IO, Union

from .._vendor import attr
from .._vendor import furl
from .._vendor import six
from .._vendor.attr import fields_dict
from .._vendor.pathlib2 import Path
from .._vendor.six.moves import reduce  # noqa
from .._vendor import pyyaml as yaml

from clearml_agent.errors import CommandFailedError
from clearml_agent.external import pyhocon
from clearml_agent.helper.dicts import filter_keys

pretty_lines = False

log = logging.getLogger(__name__)

use_powershell = os.getenv("CLEARML_AGENT_USE_POWERSHELL", None)


def which(cmd, path=None):
    from clearml_agent.helper.process import find_executable
    result = find_executable(cmd, path)
    if not result:
        raise ValueError('command "{}" not found'.format(cmd))
    return result


def select_for_platform(linux, windows):
    """
    Select between multiple values according to the OS
    :param linux: value to return if OS is linux
    :param windows: value to return if OS is Windows
    """
    return windows if is_windows_platform() else linux


def check_is_binary_python_or_bash(script_binary) -> (bool, bool):
    """
    return if we should treat it as bash script or python based on the binary section of the task
    i.e. task.script.binary
    :param script_binary: e.g. task.script.binary
    :return: tuple boolean (is_python_binary, is_bash_binary)
    """
    is_python_binary = (script_binary or "").split("/")[-1].startswith('python')
    is_bash_binary = (not is_python_binary and
                      (script_binary or "").split("/")[-1] in ('bash', 'zsh', 'sh'))

    if not is_bash_binary and not is_python_binary:
        if (script_binary or "").strip():
            print("WARNING binary '{}' not supported, defaulting to python".format(script_binary))
        is_python_binary = True

    return is_python_binary, is_bash_binary


def bash_c():
    return 'bash -c' if not is_windows_platform() else ('powershell -Command' if use_powershell else 'cmd /c')


def return_list(arg):
    if arg and not isinstance(arg, (tuple, list)):
        return [arg]

    return arg


def print_table(entries, columns=(), titles=(), csv=None, headers=True):
    table = create_table(entries, columns=columns, titles=titles, csv=csv, headers=headers)
    if csv:
        with open(csv, 'w') as output:
            print(table, file=output)
    else:
        print(table)


def create_table(entries, columns=(), titles=(), csv=None, headers=True):
    table = [
        [
            reduce(
                lambda obj, key: obj.get(key, {}),
                column.split('.'),
                entry
            ) or ''
            for column in columns
        ]
        for entry in entries
    ]
    if headers:
        headers = [titles[i] if i < len(titles) and titles[i] else c for i, c in enumerate(columns)]
    else:
        headers = []
    output = ''
    if csv:
        if headers:
            output += ','.join(headers) + '\n'
        for entry in table:
            output += ','.join(map(str, entry)) + '\n'
    else:
        min_col_width = 3
        col_widths = [max(min_col_width, len(h)+1) for h in (headers or table[0])]
        for e in table:
            col_widths = list(map(max, zip(col_widths, [len(h)+1 for h in e])))

        output += '+-' + '+-'.join(['-' * c for c in col_widths]) + '-+' + '\n'
        if headers:
            output += '| ' + '| '.join(['{: <%d}' % c for c in col_widths]).format(*headers) + ' |' + '\n'

        output += '+-' + '+-'.join(['-' * c for c in col_widths]) + '-+' + '\n'

        for entry in table:
            line = map(str, entry)
            output += '| ' + '| '.join(['{: <%d}' % c for c in col_widths]).format(*line) + ' |' + '\n'

        output += '+-' + '+-'.join(['-' * c for c in col_widths]) + '-+' + '\n'
    return output


def create_tree(entries, id='id', parent='parent', node_title='%(id)'):
    tree = OrderedDict()
    all_nodes = dict()
    for t in entries:
        i = t.get(id, None)
        p = t.get(parent, None)
        if not p and i not in tree:
            # push roots
            myd = all_nodes.get(i, OrderedDict())
            # add node title
            tree[node_title % t] = myd
            all_nodes[i] = myd
        elif p:
            # update parent dictionary
            d = all_nodes.get(p, OrderedDict())
            # get node dictionary
            myd = all_nodes.get(i, OrderedDict())
            # add node title
            d[node_title % t] = myd
            all_nodes[p] = d
            all_nodes[i] = myd
        else:
            pass
    return {'': tree}


def print_parameters(param_struct, indent=1):
    text = yaml.safe_dump(param_struct, allow_unicode=True, indent=indent, default_flow_style=False)
    print(text)


def get_list_files(basefolder, filext=('.jpg')):
    filext = [e.lower() for e in filext]
    fileiter = (os.path.join(root, f)
                for root, _, files in os.walk(basefolder)
                for f in files if os.path.splitext(f)[1].lower() in filext)
    return fileiter


def is_windows_platform():
    return any(platform.win32_ver())


def is_linux_platform():
    return 'linux' in platform.system().lower()


def normalize_path(*paths):
    """
    normalize_path

    Joins ``*paths``, expands ``~`` and normalizes path separators.

    :param paths: path components to create path from
    """
    return os.path.normpath(os.path.expandvars(os.path.expanduser(os.path.join(*map(str, paths)))))


def safe_remove_file(filename, error_message=None):
    # noinspection PyBroadException
    try:
        if filename:
            os.remove(filename)
    except Exception:
        if error_message:
            print(error_message)


def safe_remove_tree(filename):
    if not filename:
        return
    # noinspection PyBroadException
    try:
        shutil.rmtree(filename, ignore_errors=True)
    except Exception:
        pass
    # noinspection PyBroadException
    try:
        os.remove(filename)
    except Exception:
        pass


def get_python_path(script_dir, entry_point, package_api, is_conda_env=False):
    # noinspection PyBroadException
    try:
        python_path_sep = ';' if is_windows_platform() else ':'
        python_path_cmd = package_api.get_python_command(
            ["-c", "import sys; print('{}'.join(sys.path))".format(python_path_sep)])
        org_python_path = python_path_cmd.get_output(cwd=script_dir)
        # Add path of the script directory and executable directory
        python_path = '{}{python_path_sep}'.format(
            Path(script_dir).absolute().as_posix(), python_path_sep=python_path_sep)
        if entry_point:
            python_path += '{}{python_path_sep}'.format(
                (Path(script_dir) / Path(entry_point)).parent.absolute().as_posix(),
                python_path_sep=python_path_sep)

        if is_windows_platform():
            python_path = python_path.replace('/', '\\')

        return python_path if is_conda_env else (python_path + org_python_path)
    except Exception:
        return None


def add_python_path(base_path, extra_path):
    try:
        if not extra_path:
            return base_path
        python_path_sep = ';' if is_windows_platform() else ':'
        base_path = base_path or ''
        if not base_path.endswith(python_path_sep):
            base_path += python_path_sep
        base_path += extra_path.replace(':', python_path_sep)
    except:
        pass
    return base_path


class Singleton(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@total_ordering
class CompareAnything(object):
    """
    CompareAnything

    Creates an object which is always the smallest when compared to other objects.
    """

    @staticmethod
    def __eq__(_):
        return False

    @staticmethod
    def __lt__(_):
        return True


def nonstrict_in_place_sort(lst, reverse, *keys):
    """
    nonstrict_in_place_sort

    Sorts a list of dictionaries in-place by ``keys``.
    An element without a certain ``key`` will be considered the smallest in respect to that key.

    :param lst: list to sort
    :type lst: ``[dict]``
    :param reverse: whether to reverse sorting
    :type reverse: ``bool``
    :param keys: Keys to sort by.
                 Elements will be sorted pseudo-lexicographically by the values corresponding to ``*keys``, i.e:
                 the list will be first sorted by the first element of ``*keys``,
                 elements which are equal by the first sort will be internally sorted by
                 the second element of ``*keys`` and so on.
    :type keys: ``[str]``
    """
    lst.sort(
        key=lambda item: tuple(item.get(key, CompareAnything()) for key in keys),
        reverse=reverse,
    )


def load_yaml(path):
    if isinstance(path, Path):
        path = str(path)
    try:
        with open(path) as data_file:
            return yaml.safe_load(data_file) or {}
    except yaml.YAMLError as e:
        raise ValueError('Failed parsing yaml file [{}]: {}'.format(path, e))


def dump_yaml(obj, path=None, dump_all=False, **kwargs):
    base_kwargs = dict(indent=4, allow_unicode=True, default_flow_style=False)
    base_kwargs.update(kwargs)
    if dump_all:
        base_kwargs['Dumper'] = AllDumper
        dump_func = yaml.dump
    else:
        dump_func = yaml.safe_dump
    if not path:
        return dump_func(obj, **base_kwargs)
    path = str(path)
    with open(path, 'w') as output:
        dump_func(obj, output, **base_kwargs)


def _dump_flat_dict(flat_dict):
    if not isinstance(flat_dict, (dict, )):
        flat_dict = {"": flat_dict}

    out = ""
    for k in sorted(flat_dict.keys()):
        out += "{}:\n".format(k)
        values = flat_dict[k]
        if not isinstance(values, (list, tuple)):
            values = [values]
        out += "".join(sorted(["- {}\n".format(v) for v in values], key=lambda t: str(t).lower()))

    return out


def dump_flat_dict(flat_dict):
    # noinspection PyBroadException
    try:
        return _dump_flat_dict(flat_dict)
    except Exception:
        return dump_yaml(flat_dict)


def one_value(dct):
    return next(iter(six.itervalues(dct)))


@attr.s
class RepoInfo(object):
    type = attr.ib(type=str)
    url = attr.ib(type=str)
    branch = attr.ib(type=str)
    commit = attr.ib(type=str)
    root = attr.ib(type=str)


def get_repo_info(repo_type, path):
    assert repo_type in ['git', 'hg']
    if repo_type == 'git':
        commands = dict(
            url='git remote get-url origin',
            branch='git rev-parse --abbrev-ref HEAD',
            commit='git rev-parse HEAD',
            root='git rev-parse --show-toplevel'
        )
    elif repo_type == 'hg':
        commands = dict(
            url='hg paths --verbose',
            branch='hg --debug id -b',
            commit='hg --debug id -i',
            root='hg root'
        )
    else:
        raise RuntimeError("Unknown repository type '{}'".format(repo_type))

    commands_result = {
        name: subprocess.check_output(command.split(), cwd=path).decode().strip()
        for name, command in commands.items()
    }
    return RepoInfo(type=repo_type, **commands_result)


def reverse_home_folder_expansion(path):
    path = str(path)
    if is_windows_platform():
        return path
    return re.sub('^{}/'.format(re.escape(str(Path.home()))), '~/', path)


def represent_ordered_dict(dumper, data):
    """
    Serializes ``OrderedDict`` to YAML by its proper order.
    Registering this function to ``yaml.SafeDumper`` enables using ``yaml.safe_dump`` with ``OrderedDict``s.
    """
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())


def construct_mapping(loader, node):
    """
    Deserialize YAML mappings as ``OrderedDict``s.
    """
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


# yaml.SafeDumper.add_representer(OrderedDict, represent_ordered_dict)
# yaml.SafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)


class AllDumper(yaml.SafeDumper):
    pass


# AllDumper.add_multi_representer(object, lambda dumper, data: dumper.represent_str(str(data)))


def error(message):
    print('\nclearml_agent: ERROR: {}\n'.format(message))


def warning(message):
    print('clearml_agent: Warning: {}'.format(message))


class TqdmStream(object):

    def __init__(self, file_object):
        self.buffer = file_object

    def write(self, data):
        self.buffer.write(data.strip())

    def flush(self):
        self.buffer.write('\n')


def url_join(first, *rest):
    """
    Join url parts similarly to Path.join
    """
    return str(furl.furl(first).path.add(rest)).lstrip('/')


class LowercaseFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        record.levelname = record.levelname.lower()
        return super(LowercaseFormatter, self).format(record, *args, **kwargs)


def mkstemp(
        open_kwargs=None,  # type: Optional[Dict[Text, Any]]
        text=True,         # type: bool
        name_only=False,   # type: bool
        mode=None,         # type: str
        *args,
        **kwargs):
    # type: (...) -> Union[(IO[AnyStr], Text), Text]
    """
    WARNING: the returned file object is strict about its input type,
    make sure to feed it binary/text input in correspondence to the ``text`` argument
    :param open_kwargs: keyword arguments for ``io.open``
    :param text: open in text mode
    :param name_only: close the file and return its name
    :param mode: open file mode
    :param args: tempfile.mkstemp args
    :param kwargs: tempfile.mkstemp kwargs
    """
    fd, name = tempfile.mkstemp(text=text, *args, **kwargs)
    if not mode:
        mode = 'w+'
    if not text and 'b' not in mode:
        mode += 'b'
    if name_only:
        os.close(fd)
        return name
    return io.open(fd, mode, **open_kwargs or {}), name


def named_temporary_file(*args, **kwargs):
    if six.PY2:
        buffering = kwargs.pop('buffering', None)
        if buffering:
            kwargs['bufsize'] = buffering
    return tempfile.NamedTemporaryFile(*args, **kwargs)


def parse_override(string):
    return pyhocon.ConfigFactory.parse_string(string).as_plain_ordered_dict()


def chain_map(*args):
    return reduce(lambda x, y: x.update(y) or x, args, {})


def check_directory_path(path, check_whitespace_in_path=True):
    message = 'Could not create directory "{}": {}'
    if not is_windows_platform() and check_whitespace_in_path:
        match = re.search(r'\s', path)
        if match:
            raise CommandFailedError(
                'directories may not contain whitespace (char: {!r}, position: {})'.format(match.group(0),
                                                                                           match.endpos))
    try:
        Path(os.path.expandvars(path)).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CommandFailedError(message.format(path, e.strerror))
    except Exception as e:
        raise CommandFailedError(message.format(path, e))


def create_file_if_not_exists(path):
    if not os.path.exists(os.path.expanduser(os.path.expandvars(path))):
        open(path, "w").close()


def rm_tree(root):  # type: (Union[Path, Text]) -> None
    """
    A version of shutil.rmtree that handles access errors, specifically hidden files on Windows
    """
    def on_error(func, path, _):
        try:
            if os.path.exists(path) and not os.access(path, os.W_OK):
                os.chmod(path, stat.S_IWUSR)
                func(path)
        except Exception:
            pass
    return shutil.rmtree(os.path.expanduser(os.path.expandvars(Text(root))), onerror=on_error)


def rm_file(filename):  # type: (Union[Path, Text]) -> None
    """
    A version of os.unlink that will not raise error
    """
    try:
        os.unlink(os.path.expanduser(os.path.expandvars(Text(filename))))
    except:
        return False
    return True


def is_conda(config):
    return config['agent.package_manager.type'].lower() == 'conda'


def convert_cuda_version_to_float_single_digit_str(cuda_version):
    """
    Convert a cuda_version (string/float/int) into a float representation, e.g. 11.4
    Notice returns String Single digit only!
    :return str:
    """
    cuda_version = str(cuda_version or 0)
    # if we have patch version we parse it here
    cuda_version_parts = [int(v) for v in cuda_version.split('.')]
    if len(cuda_version_parts) > 1 or cuda_version_parts[0] < 60:
        cuda_version = 10 * cuda_version_parts[0]
        if len(cuda_version_parts) > 1:
            cuda_version += float(".{:d}".format(cuda_version_parts[1]))*10

        cuda_version_full = "{:.1f}".format(float(cuda_version) / 10.)
    else:
        cuda_version = cuda_version_parts[0]
        cuda_version_full = "{:.1f}".format(float(cuda_version) / 10.)

    return cuda_version_full


def convert_cuda_version_to_int_10_base_str(cuda_version):
    """
    Convert a cuda_version (string/float/int) into an integer version, e.g. 112 for cuda 11.2
    Return string
    :return str:
    """
    cuda_version = convert_cuda_version_to_float_single_digit_str(cuda_version)
    return str(int(float(cuda_version)*10))


def get_python_version(python_executable, log=None):
    from clearml_agent.helper.process import Argv
    try:
        output = Argv(python_executable, "--version").get_output(
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as ex:
        # Windows returns 9009 code and suggests to install Python from Windows Store
        if is_windows_platform() and ex.returncode == 9009:
            if log:
                log.debug("version not found: {}".format(ex))
        else:
            if log:
                log.warning("error getting %s version: %s", python_executable, ex)
        return None
    except FileNotFoundError as ex:
        if log:
            log.debug("version not found: {}".format(ex))
        return None

    match = re.search(r"Python ({}(?:\.\d+)*)".format(r"\d+"), output)
    if match:
        if log:
            log.debug("Found: {}".format(python_executable))
        # only return major.minor version
        return ".".join(str(match.group(1)).split(".")[:2])

    return None


class NonStrictAttrs(object):

    @classmethod
    def from_dict(cls, kwargs):
        fields = fields_dict(cls)
        return cls(**filter_keys(lambda key: key in fields, kwargs))


def python_version_string():
    return '{v.major}.{v.minor}'.format(v=sys.version_info)


join_lines = '\n'.join


class HOCONEncoder(json.JSONEncoder):
    """
    pyhocon bugs:
    1. "\\t" is dumped as "\t" instead of "\\t", which is read as the character "\t".
    2. parsed config trees have dummy `pyhocon.config_tree.NoneValue` in them.
        (see: https://github.com/chimpler/pyhocon/issues/111)
    Workaround: dump HOCON to JSON, of which it is a subset, taking care of `NoneValue`s.
    """

    def default(self, o):
        """
        If o is `pyhocon.config_tree.NoneValue`, encode it the same way as `None`.
        """
        if isinstance(o, pyhocon.config_tree.NoneValue):
            return super(HOCONEncoder, self).encode(None)
        return super(HOCONEncoder, self).default(o)


nullable_string = attr.ib(default="", converter=lambda x: x.strip())
normal_path = attr.ib(default="", converter=lambda p: p and normalize_path(p))


@attr.s
class ExecutionInfo(NonStrictAttrs):
    repository = nullable_string
    entry_point = normal_path
    working_dir = normal_path
    branch = nullable_string
    version_num = nullable_string
    tag = nullable_string
    docker_cmd = nullable_string

    @classmethod
    def from_task(cls, task_info):
        # type: (...) -> ExecutionInfo
        """
        extract ExecutionInfo tuple from task parameters
        """
        if not task_info.script:
            raise CommandFailedError("can not run task without script information")
        execution = cls.from_dict(task_info.script.to_dict())
        if not execution.entry_point:
            log.warning("notice: `script.entry_point` is empty")
        if not execution.working_dir:
            entry_point, _, working_dir = execution.entry_point.partition(":")
            execution.entry_point = entry_point
            execution.working_dir = working_dir or ""

        # noinspection PyBroadException
        try:
            execution.docker_cmd = task_info.execution.docker_cmd
        except Exception:
            pass

        return execution


class safe_furl(furl.furl):

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, port):
        """
        Any port value is valid
        """
        self._port = port
