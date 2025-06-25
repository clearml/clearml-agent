from __future__ import unicode_literals, print_function, absolute_import

import linecache
import os
import sys
import time
import trace
from itertools import chain
from types import ModuleType
from typing import Text, Sequence, Union

from .._vendor.pathlib2 import Path

from .._vendor import six

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache


def inclusive_parents(path):
    """
    Return path parents including path itself.
    """
    return chain((path,), path.parents)


def get_module_path(module):
    """
    :param module: Module object or name
    :return: module path
    """
    if isinstance(module, six.string_types):
        module = sys.modules[module]
    path = Path(module.__file__)
    return path.parent if path.stem == '__init__' else path


Module = Union[ModuleType, Text]


class PackageTraceIgnore(object):

    """
    Object that includes package modules in trace and excludes sub modules and all other code.
    """

    def __init__(self, package, ignore_submodules):
        # type: (Module, Sequence[Module]) -> None
        """
        Modules given by name will be searched for in sys.modules, enabling use of "__name__".
        :param package: Package to include modules of
        :param ignore_submodules: sub modules of package to ignore
        """
        self.ignore_submodules = tuple(map(get_module_path, ignore_submodules))
        self.package = package
        self.package_path = get_module_path(package)

    @lru_cache(None)
    def names(self, file_name, module_name=None):
        # type: (Text, Text) -> bool
        """
        Return whether a file should be ignored based on it's path and module name.
        Ignore files which are not part of self.package.
        trace.Ignore's documentation states that module_name is unreliable for packages,
        therefore, it is not used here.

        :param file_name: source file path
        :param module_name: module name
        :return: whether file should be ignored
        """
        file_path = Path(file_name).resolve()
        include = self.include(file_path)
        return not include

    def include(self, base):
        # type: (Path) -> bool
        for path in inclusive_parents(base):
            if not path.exists():
                continue
            if any(path.samefile(sub) for sub in self.ignore_submodules):
                return False
            if path.samefile(self.package_path):
                return True
        return False


class PackageTrace(trace.Trace, object):

    """
    Trace object for tracing only lines from a specific package.
    Some functions are copied and modified for lack of modularity of ``trace.Trace``.
    """

    def __init__(self, package, out_file, ignore_submodules=(), *args, **kwargs):
        super(PackageTrace, self).__init__(*args, **kwargs)
        self.ignore = PackageTraceIgnore(package, ignore_submodules)
        self.__out_file = out_file

    def __out(self, *args, **kwargs):
        print(*args, file=self.__out_file, **kwargs)

    def globaltrace_lt(self, frame, why, arg):
        """
        ## Copied from trace module ##
        Handler for call events.
        If the code block being entered is to be ignored, returns `None',
        else returns self.localtrace.
        """
        if why == 'call':
            code = frame.f_code
            filename = frame.f_globals.get('__file__', None)
            if filename:
                # XXX modname() doesn't work right for packages, so
                # the ignore support won't work right for packages
                ignore_it = self.ignore.names(filename)
                if not ignore_it:
                    if self.trace:
                        filename = Path(filename)
                        modulename = '.'.join(
                            filename.relative_to(self.ignore.package_path).parts[:-1] + (filename.stem,)
                        )
                        self.__out(' --- modulename: %s, funcname: %s' % (modulename, code.co_name))
                    return self.localtrace
            else:
                return None

    def localtrace_trace(self, frame, why, arg):
        """
        ## Copied from trace module ##
        """
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            if self.start_time:
                self.__out('%.2f' % (time.time() - self.start_time), end='')
            bname = os.path.basename(filename)
            self.__out('%s(%d): %s' % (bname, lineno, linecache.getline(filename, lineno)), end='')
        return self.localtrace

    localtrace_trace_and_count = localtrace_trace
