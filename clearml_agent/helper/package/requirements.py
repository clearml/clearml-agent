from __future__ import absolute_import, unicode_literals

import operator
import os
import re
from abc import ABCMeta, abstractmethod
from copy import deepcopy, copy
from itertools import chain, starmap
from operator import itemgetter
from os import path
from typing import Text, List, Type, Optional, Tuple, Dict

from ..._vendor.pathlib2 import Path
from clearml_agent.external.pyhocon import ConfigTree

from ..._vendor import six
from ..._vendor.six.moves.urllib.parse import unquote  # noqa
import logging
from clearml_agent.definitions import PIP_EXTRA_INDICES
from clearml_agent.helper.base import (
    warning, is_conda, which, join_lines, is_windows_platform,
    convert_cuda_version_to_int_10_base_str, dump_yaml, )
from clearml_agent.helper.process import Argv, PathLike
from clearml_agent.helper.gpu.gpustat import get_driver_cuda_version
from clearml_agent.session import Session, normalize_cuda_version
from clearml_agent.external.requirements_parser import parse
from clearml_agent.external.requirements_parser.requirement import Requirement

from .translator import RequirementsTranslator


class SpecsResolutionError(Exception):
    pass


class FatalSpecsResolutionError(Exception):
    pass


@six.python_2_unicode_compatible
class MarkerRequirement(object):

    # if True pip version above 20.x and with support for "package @ scheme://link"
    # default is True
    pip_new_version = True

    def __init__(self, req):  # type: (Requirement) -> None
        self.req = req

    @property
    def marker(self):
        match = re.search(r';\s*(.*)', self.req.line)
        if match:
            return match.group(1)
        return None

    def tostr(self, markers=True):
        if not self.uri:
            parts = [self.name or self.line]

            if self.extras:
                parts.append('[{0}]'.format(','.join(sorted(self.extras))))

            if self.specifier:
                parts.append(self.format_specs())
        elif self.vcs:
            # leave the line as is, let pip handle it
            if self.line:
                return self.line
            else:
                # let's build the line manually
                parts = [
                    self.uri,
                    '@{}'.format(self.revision) if self.revision else '',
                    '#subdirectory={}'.format(self.subdirectory) if self.subdirectory else ''
                ]
        elif self.pip_new_version and self.uri and self.name and self.line and self.local_file:
            # package @ file:///example.com/somewheel.whl
            # leave the line as is, let pip handle it
            return self.line
        else:
            parts = [self.uri]

        if markers and self.marker:
            parts.append('; {0}'.format(self.marker))

        return ''.join(parts)

    def clone(self):
        return MarkerRequirement(copy(self.req))

    __str__ = tostr

    def __repr__(self):
        return '{self.__class__.__name__}[{self}]'.format(self=self)

    def __eq__(self, other):
        return isinstance(other, MarkerRequirement) and str(self) == str(other)

    def __hash__(self):
        return str(self).__hash__()

    def format_specs(self, num_parts=None, max_num_parts=None):
        max_num_parts = max_num_parts or num_parts
        if max_num_parts is None or not self.specs:
            return ','.join(starmap(operator.add, self.specs))

        op, version = self.specs[0]
        # noinspection PyProtectedMember
        for v in SimpleVersion._sub_versions_pep440:
            version = version.replace(v, '.')
        if num_parts:
            version = (version.strip('.').split('.') + ['0'] * num_parts)[:max_num_parts]
        else:
            version = version.strip('.').split('.')[:max_num_parts]
        return op+'.'.join(version)

    def __getattr__(self, item):
        return getattr(self.req, item)

    @property
    def specs(self):  # type: () -> List[Tuple[Text, Text]]
        return self.req.specs

    @property
    def version(self):  # type: () -> Text
        return self.specs[0][1] if self.specs else ""

    @specs.setter
    def specs(self, value):  # type: (List[Tuple[Text, Text]]) -> None
        self.req.specs = value

    def fix_specs(self):
        def solve_by(func, op_is, specs):
            return func([(op, version) for op, version in specs if op == op_is])

        def solve_equal(specs):
            if len(set(version for _, version in self.specs)) > 1:
                raise SpecsResolutionError('more than one "==" spec: {}'.format(specs))
            return specs
        greater = solve_by(lambda specs: [max(specs, key=itemgetter(1))], '<=', self.specs)
        smaller = solve_by(lambda specs: [min(specs, key=itemgetter(1))], '>=', self.specs)
        equal = solve_by(solve_equal, '==', self.specs)
        if equal:
            self.specs = equal
        else:
            self.specs = greater + smaller

    def compare_version(self, requested_version, op=None, num_parts=3):
        """
        compare the requested version with the one we have in the spec,
        If the requested version is 1.2.3 the self.spec should be 1.2.3*
        If the requested version is 1.2 the self.spec should be 1.2*
        etc.

        usage: it returns the value of the following comparison: requested_version "op" self.version

        :param str requested_version:
        :param str op: '==', '>', '>=', '<=', '<', '~='
        :param int num_parts: number of parts to compare
        :return: True if we answer the requested version
        """
        # if we have no specific version, we cannot compare, so assume it's okay
        if not self.specs:
            return True

        version = self.version
        op = (op or self.specs[0][0]).strip()

        return SimpleVersion.compare_versions(
            version_a=requested_version, op=op, version_b=version, num_parts=num_parts)

    def remove_local_file_ref(self):
        if not self.local_file or self.vcs or self.editable or self.path:
            return False
        parts = re.split(r"@\s*{}".format(self.req.uri), self.req.line)
        # if we did not find anything do nothing
        if len(parts) < 2:
            return False
        self.req.line = ''.join(parts).strip()
        self.req.uri = None
        self.req.local_file = False
        return True

    def is_local_package_ref(self):
        # if local file does not exist, remove the reference to it
        if self.vcs or self.editable or self.path or not self.local_file or not self.name or \
                not self.uri or not self.uri.startswith("file://"):
            return False
        return True

    def is_vcs_ref(self):
        return bool(self.vcs)

    def validate_local_file_ref(self):
        # if local file does not exist, remove the reference to it
        if not self.is_local_package_ref():
            return

        local_path = Path(self.uri[len("file://"):])
        if not local_path.exists():
            local_path = Path(unquote(self.uri)[len("file://"):])
            if not local_path.exists():
                line = self.line
                if self.remove_local_file_ref():
                    # print warning
                    logging.getLogger(__name__).warning(
                        'Local file not found [{}], references removed'.format(line))


class SimpleVersion:
    _sub_versions_pep440 = ['a', 'b', 'rc', '.post', '.dev', '+', ]
    VERSION_PATTERN = r"""
        v?
        (?:
            (?:(?P<epoch>[0-9]+)!)?                           # epoch
            (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
            (?P<pre>                                          # pre-release
                [-_\.]?
                (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
                [-_\.]?
                (?P<pre_n>[0-9]+)?
            )?
            (?P<post>                                         # post release
                (?:-(?P<post_n1>[0-9]+))
                |
                (?:
                    [-_\.]?
                    (?P<post_l>post|rev|r)
                    [-_\.]?
                    (?P<post_n2>[0-9]+)?
                )
            )?
            (?P<dev>                                          # dev release
                [-_\.]?
                (?P<dev_l>dev)
                [-_\.]?
                (?P<dev_n>[0-9]+)?
            )?
        )
        (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
    """
    _local_version_separators = re.compile(r"[\._-]")
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    @classmethod
    def split_op_version(cls, line):
        """
        Split a string in the form of ">=1.2.3" into a (op, version), i.e. (">=", "1.2.3")
        Notice is calling with only a version string (e.g. "1.2.3") default operator is "=="
        which means you get ("==", "1.2.3")
        :param line: string examples: "<=0.1.2"
        :return: tuple of (op, version) example ("<=", "0.1.2")
        """
        match = r"\s*([>=<~!]*)\s*(\S*)\s*"
        groups = re.match(match, line).groups()
        return groups[0] or "==", groups[1]

    @classmethod
    def compare_versions(cls, version_a, op, version_b, ignore_sub_versions=True, num_parts=3):
        """
        Compare two versions based on the op operator
        returns bool(version_a op version_b)
        Notice: Ignores a/b/rc/post/dev markers on the version

        :param str version_a:
        :param str op: '==', '===', '>', '>=', '<=', '<', '~='
        :param str version_b:
        :param bool ignore_sub_versions: if true compare only major.minor.patch
            (ignore a/b/rc/post/dev in the comparison)
        :param int num_parts: number of parts to compare, split by . (dot)
        :return bool: version_a op version_b
        """

        if not version_b:
            return True

        # remove trailing "*" in both
        if "*" in version_a:
            ignore_sub_versions = True
            while version_a.endswith(".*"):
                version_a = version_a[:-2]
            if version_a == "*":
                version_a = ""
            num_parts = min(len(version_a.split('.')), len(version_b.split('.')), )

        if "*" in version_b:
            ignore_sub_versions = True
            while version_b.endswith(".*"):
                version_b = version_b[:-2]
            if version_b == "*":
                version_b = ""
            num_parts = min(len(version_a.split('.')), len(version_b.split('.')), )

        if not num_parts:
            num_parts = max(len(version_a.split('.')), len(version_b.split('.')), )

        if op == '~=':
            num_parts = len(version_b.split('.')) - 1
            num_parts = max(num_parts, 2)
            op = '=='
            ignore_sub_versions = True
        elif op == '===':
            op = '=='

        try:
            version_a_key = cls._get_match_key(cls._regex.search(version_a), num_parts, ignore_sub_versions)
            version_b_key = cls._get_match_key(cls._regex.search(version_b), num_parts, ignore_sub_versions)
        except:
            # revert to string based
            for v in cls._sub_versions_pep440:
                version_a = version_a.replace(v, '.')
                version_b = version_b.replace(v, '.')

            version_a = (version_a.strip('.').split('.') + ['0'] * num_parts)[:num_parts]
            version_b = (version_b.strip('.').split('.') + ['0'] * num_parts)[:num_parts]
            version_a_key = ''
            version_b_key = ''
            for i in range(num_parts):
                pad = '{:0>%d}.' % max([9, 1 + len(version_a[i]), 1 + len(version_b[i])])
                version_a_key += pad.format(version_a[i])
                version_b_key += pad.format(version_b[i])

        if op == '==':
            return version_a_key == version_b_key
        if op == '<=':
            return version_a_key <= version_b_key
        if op == '>=':
            return version_a_key >= version_b_key
        if op == '>':
            return version_a_key > version_b_key
        if op == '<':
            return version_a_key < version_b_key
        if op == '!=':
            return version_a_key != version_b_key
        raise ValueError('Unrecognized comparison operator [{}]'.format(op))

    @classmethod
    def max_version(cls, version_a, version_b):
        return version_a if cls.compare_versions(
            version_a=version_a, op='>=', version_b=version_b, num_parts=None) else version_b

    @classmethod
    def min_version(cls, version_a, version_b):
        return version_a if cls.compare_versions(
            version_a=version_a, op='<=', version_b=version_b, num_parts=None) else version_b

    @staticmethod
    def _parse_letter_version(
            letter,  # type: str
            number,  # type: Union[str, bytes, SupportsInt]
    ):
        # type: (...) -> Optional[Tuple[str, int]]

        if letter:
            # We consider there to be an implicit 0 in a pre-release if there is
            # not a numeral associated with it.
            if number is None:
                number = 0

            # We normalize any letters to their lower case form
            letter = letter.lower()

            # We consider some words to be alternate spellings of other words and
            # in those cases we want to normalize the spellings to our preferred
            # spelling.
            if letter == "alpha":
                letter = "a"
            elif letter == "beta":
                letter = "b"
            elif letter in ["c", "pre", "preview"]:
                letter = "rc"
            elif letter in ["rev", "r"]:
                letter = "post"

            return letter, int(number)
        if not letter and number:
            # We assume if we are given a number, but we are not given a letter
            # then this is using the implicit post release syntax (e.g. 1.0-1)
            letter = "post"

            return letter, int(number)

        return ()

    @staticmethod
    def _get_match_key(match, num_parts, ignore_sub_versions):
        if ignore_sub_versions:
            return (0, tuple(int(i) for i in match.group("release").split(".")[:num_parts]),
                    (), (), (), (),)
        return (
            int(match.group("epoch")) if match.group("epoch") else 0,
            tuple(int(i) for i in match.group("release").split(".")[:num_parts]),
            SimpleVersion._parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            SimpleVersion._parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            SimpleVersion._parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            SimpleVersion._parse_local_version(match.group("local")),
        )

    @staticmethod
    def _parse_local_version(local):
        # type: (str) -> Optional[LocalType]
        """
        Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
        """
        if local is not None:
            return tuple(
                part.lower() if not part.isdigit() else int(part)
                for part in SimpleVersion._local_version_separators.split(local)
            )
        return ()


def compare_version_rules(specs_a, specs_b):
    # specs_a/b are a list of tuples: [('==', '1.2.3'), ] or [('>=', '1.2'), ('<', '1.3')]
    # section definition:
    class Section(object):
        def __init__(self, left="-999999999", left_eq=False, right="999999999", right_eq=False):
            self.left, self.left_eq, self.right, self.right_eq = left, left_eq, right, right_eq
    # first create a list of in/out sections for each spec
    # >, >= are left rule
    # <, <= are right rule
    # ~= x.y.z is converted to: >= x.y and < x.y+1
    # ==/=== are converted to: >= and <=
    # != x.y.z will split a section into: left < x.y.z and right > x.y.z
    def create_section(specs):
        section = Section()
        for op, v in specs:
            a = section
            if op == '>':
                a.left = v
                a.left_eq = False
            elif op == '>=':
                a.left = v
                a.left_eq = True
            elif op == '<':
                a.right = v
                a.right_eq = False
            elif op == '<=':
                a.right = v
                a.right_eq = True
            elif op == '==':
                a.left = v
                a.left_eq = True
                a.right = v
                a.right_eq = True
            elif op == '~=':
                new_v = v.split('.')
                a_left = '.'.join(new_v[:-1])
                a.left = a_left if not a.left else SimpleVersion.max_version(a_left, a.left)
                a.left_eq = True
                a_right = '.'.join(new_v[:-2] + [str(int(new_v[-2])+1)])
                a.right = a_right if not a.right else SimpleVersion.min_version(a_right, a.right)
                a.right_eq = False if a.right == a_right else a.right_eq

        return section

    section_a = create_section(specs_a)
    section_b = create_section(specs_b)
    i = Section()
    # then we have a list of sections for spec A/B
    if section_a.left == section_b.left:
        i.left = section_a.left
        i.left_eq = section_a.left_eq and section_b.left_eq
    else:
        i.left = SimpleVersion.max_version(section_a.left, section_b.left)
        i.left_eq = section_a.left_eq if i.left == section_a.left else section_b.left_eq
    if section_a.right == section_b.right:
        i.right = section_a.right
        i.right_eq = section_a.right_eq and section_b.right_eq
    else:
        i.right = SimpleVersion.min_version(section_a.right, section_b.right)
        i.right_eq = section_a.right_eq if i.right == section_a.right else section_b.right_eq

    # return true if any section from A intersects a section from B
    valid = True
    valid &= SimpleVersion.compare_versions(
        version_a=i.left, op='<=' if i.left_eq else '<', version_b=i.right, num_parts=None)
    valid &= SimpleVersion.compare_versions(
        version_a=i.right, op='>=' if i.left_eq else '>', version_b=i.left, num_parts=None)

    return valid


@six.add_metaclass(ABCMeta)
class RequirementSubstitution(object):

    _pip_extra_index_url = PIP_EXTRA_INDICES

    @classmethod
    def set_add_install_extra_index(cls, extra_index_url):
        if extra_index_url not in cls._pip_extra_index_url:
            cls._pip_extra_index_url.append(extra_index_url)

    def __init__(self, session):
        # type: (Session) -> ()
        self._session = session
        self.config = session.config  # type: ConfigTree
        self.suffix = '.post{config[agent.cuda_version]}.dev{config[agent.cudnn_version]}'.format(config=self.config)
        self.package_manager = self.config['agent.package_manager.type']
        self._is_already_installed_cb = None

    @abstractmethod
    def match(self, req):  # type: (MarkerRequirement) -> bool
        """
        Returns whether a requirement needs to be modified by this substitution.
        """
        pass

    @abstractmethod
    def replace(self, req):  # type: (MarkerRequirement) -> Text
        """
        Replace a requirement
        """
        pass

    def set_is_already_installed_cb(self, cb):
        self._is_already_installed_cb = cb

    def is_already_installed(self, req):
        if not self._is_already_installed_cb:
            return False
        # noinspection PyBroadException
        try:
            return self._is_already_installed_cb(req)
        except BaseException as ex:
            # debug could not resolve something
            print("Warning: Requirements post install callback exception (check if package installed): {}".format(ex))
            return False

    def post_scan_add_req(self):  # type: () -> Optional[MarkerRequirement]
        """
        Allows the RequirementSubstitution to add an extra line/requirements after
        the initial requirements scan is completed.
        Called only once per requirements.txt object
        """
        return None

    def post_install(self, session):
        pass

    @classmethod
    def get_pip_version(cls, package):
        output = Argv(
            'pip',
            'search',
            package,
            *(chain.from_iterable(('-i', x) for x in cls._pip_extra_index_url))
        ).get_output()
        # ad-hoc pattern to duplicate the behavior of the old code
        return re.search(r'{} \((\d+\.\d+\.[^.]+)'.format(package), output).group(1)

    @property
    def cuda_version(self):
        return convert_cuda_version_to_int_10_base_str(self.config['agent.cuda_version'])

    @property
    def cudnn_version(self):
        return self.config['agent.cudnn_version']


class SimpleSubstitution(RequirementSubstitution):

    @property
    @abstractmethod
    def name(self):
        pass

    def match(self, req):  # type: (MarkerRequirement) -> bool
        return (self.name == req.name or (
            req.uri and
            re.match(r'https?://', req.uri) and
            self.name in req.uri
        ))

    def replace(self, req):  # type: (MarkerRequirement) -> Text
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        if req.uri:
            return re.sub(
                r'({})(.*?)(-cp)'.format(self.name),
                r'\1\2{}\3'.format(self.suffix),
                req.uri,
                count=1)

        if req.specs:
            _, version_number = req.specs[0]
            # assert packaging_version.parse(version_number)
        else:
            version_number = self.get_pip_version(self.name)

        req.specs = [('==', version_number + self.suffix)]
        return Text(req)

    def replace_back(self, list_of_requirements):  # type: (Dict) -> Dict
        """
        :param list_of_requirements: {'pip': ['a==1.0', ]}
        :return: {'pip': ['a==1.0', ]}
        """
        return list_of_requirements


@six.add_metaclass(ABCMeta)
class CudaSensitiveSubstitution(SimpleSubstitution):

    def match(self, req):  # type: (MarkerRequirement) -> bool
        return self.cuda_version and self.cudnn_version and \
            super(CudaSensitiveSubstitution, self).match(req)


class CudaNotFound(Exception):
    pass


class RequirementsManager(object):

    def __init__(self, session, base_interpreter=None):
        # type: (Session, PathLike) -> ()
        self._session = session
        self.config = deepcopy(session.config)  # type: ConfigTree
        self.handlers = []  # type: List[RequirementSubstitution]
        agent = self.config['agent']
        self.active = not agent.get('cpu_only', False)
        self.found_cuda = False
        if self.active:
            try:
                agent['cuda_version'], agent['cudnn_version'] = self.get_cuda_version(self.config)
                self.found_cuda = True
            except Exception:
                # if we have a cuda version, it is good enough (we dont have to have cudnn version)
                if agent.get('cuda_version'):
                    self.found_cuda = True
        pip_cache_dir = Path(self.config["agent.pip_download_cache.path"]).expanduser() / (
            'cu'+agent['cuda_version'] if self.found_cuda else 'cpu')
        self.translator = RequirementsTranslator(session, interpreter=base_interpreter,
                                                 cache_dir=pip_cache_dir.as_posix())
        self._base_interpreter = base_interpreter
        self._cwd = None
        self._installed_parsed_packages = set()

    def register(self, cls):  # type: (Type[RequirementSubstitution]) -> None
        self.handlers.append(cls(self._session))

    def set_cwd(self, cwd):
        self._cwd = str(cwd) if cwd else None

    def _replace_one(self, req):  # type: (MarkerRequirement) -> Optional[Text]
        match = re.search(r';\s*(.*)', Text(req))
        if match:
            req.markers = match.group(1).split(',')
        if not self.active:
            return None
        for handler in self.handlers:
            if handler.match(req):
                return handler.replace(req)
        return None

    def replace(
            self,
            requirements,  # type: Text
            existing_packages=None,  # type: List[MarkerRequirement]
            pkg_skip_existing_local=True,  # type: bool
            pkg_skip_existing_vcs=True,  # type: bool
            pkg_skip_existing=True,  # type: bool
    ):  # type: (...) -> Text
        parsed_requirements = self.parse_requirements_section_to_marker_requirements(
            requirements=requirements, cwd=self._cwd, skip_local_file_validation=True)

        if parsed_requirements and existing_packages:
            skipped_packages = None
            if pkg_skip_existing:
                skipped_packages = set(parsed_requirements) & set(existing_packages)
            elif pkg_skip_existing_local or pkg_skip_existing_vcs:
                existing_packages = [
                    p for p in existing_packages if (
                        (pkg_skip_existing_local and p.is_local_package_ref()) or
                        (pkg_skip_existing_vcs and p.is_vcs_ref())
                    )
                ]
                skipped_packages = set(parsed_requirements) & set(existing_packages)

            if skipped_packages:
                # maintain order
                num_skipped_packages = len(parsed_requirements)
                parsed_requirements = [p for p in parsed_requirements if p not in skipped_packages]
                num_skipped_packages -= len(parsed_requirements)
                print("Skipping {} pre-installed packages:\n{}Remaining {} additional packages to install".format(
                    num_skipped_packages,
                    dump_yaml(sorted([str(p) for p in skipped_packages])),
                    len(parsed_requirements)
                ))

                # nothing to install!
                if not parsed_requirements:
                    return ""

        # sanity check
        if not parsed_requirements:
            # return the original requirements just in case
            return requirements

        # remove local file reference that do not exist
        for p in parsed_requirements:
            p.validate_local_file_ref()

        def replace_one(i, req):
            # type: (int, MarkerRequirement) -> Optional[Text]
            try:
                return self._replace_one(req)
            except FatalSpecsResolutionError:
                warning('could not resolve python wheel replacement for {}'.format(req))
                raise
            except Exception:
                warning('could not resolve python wheel replacement for \"{}\", '
                        'using original requirements line: {}'.format(req, i))
                return None

        new_requirements = tuple(replace_one(i, req) for i, req in enumerate(parsed_requirements))
        conda = is_conda(self.config)
        result = map(
            lambda x, y: (x if x is not None else y.tostr(markers=not conda)),
            new_requirements,
            parsed_requirements
        )
        if not conda:
            result = map(self.translator.translate, result)

        result = list(result)
        # add post scan add requirements call back
        double_req_set = None
        for h in self.handlers:
            reqs = h.post_scan_add_req()
            if reqs:
                if double_req_set is None:
                    def safe_parse_name(line):
                        try:
                            return Requirement.parse(line).name
                        except:  # noqa
                            return None
                    double_req_set = set([safe_parse_name(r) for r in result if r])

                for r in (reqs if isinstance(reqs, (tuple, list)) else [reqs]):
                    if r and (not r.name or r.name not in double_req_set):
                        result.append(r.tostr())
                    elif r:
                        print("SKIPPING additional auto installed package: \"{}\"".format(r))

        return join_lines(result)

    def post_install(self, session, package_manager=None):
        if package_manager:
            self.update_installed_packages_state(package_manager.freeze())
        for h in self.handlers:
            try:
                h.post_install(session)
            except Exception as ex:
                print('RequirementsManager handler {} raised exception: {}'.format(h, ex))
                raise

    def replace_back(self, requirements):
        if self.translator:
            requirements = self.translator.replace_back(requirements)

        for h in self.handlers:
            try:
                requirements = h.replace_back(requirements)
            except Exception:
                pass
        return requirements

    def get_interpreter(self):
        return self._base_interpreter

    def update_installed_packages_state(self, requirements):
        """
        Updates internal Installed Packages objects, so that later we can detect
        if we already have a pre-installed package
        :param requirements: is the output of a freeze() call, i.e. dict {'pip': "package==version"}
        """
        requirements = requirements if not isinstance(requirements, dict) else requirements.get("pip")
        self._installed_parsed_packages = self.parse_requirements_section_to_marker_requirements(
                requirements=requirements, cwd=self._cwd)
        for h in self.handlers:
            h.set_is_already_installed_cb(self._callback_is_already_installed)

    def get_installed_packages_state(self):
        """
        Return the internal Installed Packages objects, the one used internally so that later we can detect
        :return requirements: is the list of requirements objects updated via self.update_installed_packages_state
        """
        return self._installed_parsed_packages

    def _callback_is_already_installed(self, req):
        for p in (self._installed_parsed_packages or []):
            if p.name != req.name:
                continue
            # if this is version control package, only return true of both installed and requests specify commit ID
            if req.vcs:
                return p.vcs and req.revision and req.revision == p.revision

            if not req.specs and not p.specs:
                return True

            # return if this is the same version
            return req.specs and p.specs and req.compare_version(p, op="==")

        return False

    @staticmethod
    def get_cuda_version(config):  # type: (ConfigTree) -> (Text, Text)
        # we assume os.environ already updated the config['agent.cuda_version'] & config['agent.cudnn_version']
        cuda_version = config['agent.cuda_version']
        cudnn_version = config['agent.cudnn_version']
        if cuda_version and cudnn_version:
            return normalize_cuda_version(cuda_version), normalize_cuda_version(cudnn_version)

        if not cuda_version:
            cuda_version = get_driver_cuda_version()

        if not cuda_version and is_windows_platform():
            try:
                cuda_vers = [int(k.replace('CUDA_PATH_V', '').replace('_', '')) for k in os.environ.keys()
                             if k.startswith('CUDA_PATH_V')]
                cuda_vers = max(cuda_vers)
                if cuda_vers > 40:
                    cuda_version = cuda_vers
            except:
                pass

        if not cuda_version:
            try:
                try:
                    nvcc = 'nvcc.exe' if is_windows_platform() else 'nvcc'
                    if is_windows_platform() and 'CUDA_PATH' in os.environ:
                        nvcc = os.path.join(os.environ['CUDA_PATH'], nvcc)

                    output = Argv(nvcc, '--version').get_output()
                except OSError:
                    raise CudaNotFound('nvcc not found')
                match = re.search(r'release (.{3})', output).group(1)
                cuda_version = Text(int(float(match) * 10))
            except:
                pass

        if not cuda_version:
            try:
                try:
                    output = Argv('nvidia-smi',).get_output()
                except OSError:
                    raise CudaNotFound('nvcc not found')
                match = re.search(r'CUDA Version: ([0-9]+).([0-9]+)', output)
                match = match.group(1)+'.'+match.group(2)
                cuda_version = Text(int(float(match) * 10))
            except:
                pass

        if not cudnn_version:
            try:
                cuda_lib = which('nvcc')
                if is_windows_platform:
                    cudnn_h = path.sep.join(cuda_lib.split(path.sep)[:-2] + ['include', 'cudnn.h'])
                else:
                    cudnn_h = path.join(path.sep, *(cuda_lib.split(path.sep)[:-2] + ['include', 'cudnn.h']))

                cudnn_major, cudnn_minor = None, None
                try:
                    include_file = open(cudnn_h)
                except OSError:
                    raise CudaNotFound('Could not read cudnn.h')
                with include_file:
                    for line in include_file:
                        if 'CUDNN_MAJOR' in line:
                            cudnn_major = line.split()[-1]
                        if 'CUDNN_MINOR' in line:
                            cudnn_minor = line.split()[-1]
                        if cudnn_major and cudnn_minor:
                            break
                cudnn_version = cudnn_major + (cudnn_minor or '0')
            except:
                pass

        return (normalize_cuda_version(cuda_version or 0),
                normalize_cuda_version(cudnn_version or 0))

    @staticmethod
    def parse_requirements_section_to_marker_requirements(requirements, cwd=None, skip_local_file_validation=False):
        def safe_parse(req_str):
            # noinspection PyBroadException
            try:
                return list(parse(req_str, cwd=cwd))
            except Exception as ex:
                return [Requirement(req_str)]

        def create_req(x):
            r = MarkerRequirement(x)
            if not skip_local_file_validation:
                r.validate_local_file_ref()
            return r

        if not requirements:
            return tuple()

        parsed_requirements = tuple(
            map(
                create_req,
                [r for line in (requirements.splitlines() if isinstance(requirements, str) else requirements)
                 for r in safe_parse(line)]
            )
        )
        return parsed_requirements
