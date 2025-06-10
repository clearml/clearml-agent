from __future__ import unicode_literals

import json
import re
import os
import subprocess
from collections import OrderedDict
from functools import partial
from itertools import chain
from typing import Text, Iterable, Union, Dict, Set, Sequence, Any

from ..._vendor import six
from ..._vendor import pyyaml as yaml
from time import time
from ..._vendor.attr import attrs, attrib, Factory
from ..._vendor.pathlib2 import Path
from clearml_agent.external.requirements_parser import parse
from clearml_agent.external.requirements_parser.requirement import Requirement

from clearml_agent.errors import CommandFailedError
from clearml_agent.helper.base import (
    rm_tree, NonStrictAttrs, select_for_platform, is_windows_platform, ExecutionInfo,
    convert_cuda_version_to_float_single_digit_str, convert_cuda_version_to_int_10_base_str, )
from clearml_agent.helper.process import Argv, Executable, DEVNULL, CommandSequence, PathLike, find_executable
from clearml_agent.helper.package.requirements import SimpleVersion
from clearml_agent.session import Session
from .base import PackageManager
from .pip_api.venv import VirtualenvPip
from .requirements import RequirementsManager, MarkerRequirement
from ...backend_api.session.defs import ENV_CONDA_ENV_PACKAGE, ENV_USE_CONDA_BASE_ENV

package_normalize = partial(re.compile(r"""\[version=['"](.*)['"]\]""").sub, r"\1")


def package_set(packages):
    return set(map(package_normalize, packages))


def _package_diff(path, packages):
    # type: (Union[Path, Text], Iterable[Text]) -> Set[Text]
    return package_set(Path(path).read_text().splitlines()) - package_set(packages)


class CondaPip(VirtualenvPip):
    def __init__(self, source=None, *args, **kwargs):
        super(CondaPip, self).__init__(*args, interpreter=Path(kwargs.get('path'), "python.exe")
                                       if is_windows_platform() and kwargs.get('path') else None, **kwargs)
        self.source = source

    def run_with_env(self, command, output=False, **kwargs):
        if not self.source:
            return super(CondaPip, self).run_with_env(command, output=output, **kwargs)
        command = CommandSequence(self.source, Argv("pip", *command))
        return (command.get_output if output else command.check_call)(
            stdin=DEVNULL, **kwargs
        )


class CondaAPI(PackageManager):

    """
    A programmatic interface for controlling conda
    """

    MINIMUM_VERSION = "4.3.30"

    def __init__(self, session, path, python, requirements_manager, execution_info=None, **kwargs):
        # type: (Session, PathLike, float, RequirementsManager, ExecutionInfo, Any) -> None
        """
        :param python: base python version to use (e.g python3.6)
        :param path: path of env
        """
        super(CondaAPI, self).__init__()
        self.session = session
        self.python = python
        self.source = None
        self.requirements_manager = requirements_manager
        self.path = path
        self.env_read_only = False
        self.extra_channels = self.session.config.get('agent.package_manager.conda_channels', [])
        # install into base conda environment (should only be used if running in docker mode)
        self.use_conda_base_env = ENV_USE_CONDA_BASE_ENV.get(
            default=self.session.config.get('agent.package_manager.use_conda_base_env', None)
        )
        # notice this will not install any additional packages into the selected environment
        self.conda_env_as_base_docker = \
            self.session.config.get('agent.package_manager.conda_env_as_base_docker', None) or \
            bool(ENV_CONDA_ENV_PACKAGE.get())
        if ENV_CONDA_ENV_PACKAGE.get():
            self.conda_pre_build_env_path = ENV_CONDA_ENV_PACKAGE.get()
        else:
            self.conda_pre_build_env_path = execution_info.docker_cmd if execution_info else None
        self.pip = CondaPip(
            session=self.session,
            source=self.source,
            python=self.python,
            requirements_manager=self.requirements_manager,
            path=self.path,
        )
        try:
            self.conda = (
                find_executable("conda") or
                Argv(select_for_platform(windows="where", linux="which"), "conda").get_output(
                    shell=select_for_platform(windows=True, linux=False)).strip()
            )
        except Exception:
            raise ValueError("ERROR: package manager \"conda\" selected, "
                             "but \'conda\' executable could not be located")
        try:
            output = Argv(self.conda, "--version").get_output(stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            raise CommandFailedError(
                "Unable to determine conda version: {ex}, output={ex.output}".format(
                    ex=ex
                )
            )
        self.conda_version = self.get_conda_version(output)
        if SimpleVersion.compare_versions(self.conda_version, '<', self.MINIMUM_VERSION):
            raise CommandFailedError(
                "conda version '{}' is smaller than minimum supported conda version '{}'".format(
                    self.conda_version, self.MINIMUM_VERSION
                )
            )

    @staticmethod
    def get_conda_version(output):
        match = re.search(r"(\d+\.){0,2}\d+", output)
        if not match:
            raise CommandFailedError("Unidentified conda version string:", output)
        return match.group(0)

    @property
    def bin(self):
        return self.pip.bin

    def _parse_package_marker_match_python_ver(self, line=None, marker_req=None):
        if line:
            marker_req = MarkerRequirement(Requirement.parse(line))

        try:
            mock_req = MarkerRequirement(Requirement.parse(marker_req.marker.replace("'", "").replace("\"", "")))
        except Exception as ex:
            print("WARNING: failed parsing, assuming package is okay {}".format(ex))
            return marker_req

        if not mock_req.compare_version(requested_version=self.python):
            print("SKIPPING package `{}` not required python version {}".format(marker_req.tostr(), self.python))
            return None
        return marker_req

    # noinspection SpellCheckingInspection
    def upgrade_pip(self):
        # do not change pip version if pre built environement is used
        if self.env_read_only:
            print('Conda environment in read-only mode, skipping pip upgrade.')
            return ''

        pip_versions = []
        for req_pip_line in self.pip.get_pip_versions():
            req = self._parse_package_marker_match_python_ver(line=req_pip_line)
            if req:
                pip_versions.append(req.tostr(markers=False))

        return self._install(
            *select_for_platform(
                windows=pip_versions,
                linux=pip_versions
            )
        )

    def create(self):
        """
        Create a new environment
        """
        if self.conda_env_as_base_docker and self.conda_pre_build_env_path:
            if Path(self.conda_pre_build_env_path).is_dir():
                self._init_existing_environment(self.conda_pre_build_env_path)
                return self
            elif Path(self.conda_pre_build_env_path).is_file():
                print("Restoring Conda environment from {}".format(self.conda_pre_build_env_path))
                tar_path = find_executable("tar")
                self.path.mkdir(parents=True, exist_ok=True)
                output = Argv(
                    tar_path,
                    "-xzf",
                    self.conda_pre_build_env_path,
                    "-C",
                    self.path,
                ).get_output()

                self.source = self.pip.source = ("conda", "activate", self.path.as_posix())
                conda_env = self._get_conda_sh()
                self.source = self.pip.source = CommandSequence(('source', conda_env.as_posix()), self.source)
                # unpack cleanup
                print("Fixing prefix in Conda environment {}".format(self.path))
                CommandSequence(('source', conda_env.as_posix()),
                                ((self.path / 'bin' / 'conda-unpack').as_posix(), )).get_output()
                return self
            else:
                raise ValueError("Could not restore Conda environment, cannot find {}".format(
                    self.conda_pre_build_env_path))
        elif self.use_conda_base_env:
            try:
                base_path = Path(self.conda).parent.parent.as_posix()
                print("Using base conda environment at {}".format(base_path))
                self._init_existing_environment(base_path, is_readonly=False)
                return self
            except Exception as ex:
                print("WARNING: Failed using base conda environment, reverting to new environment: {}".format(ex))

        command = Argv(
            self.conda,
            "create",
            "--yes",
            "--mkdir",
            "--prefix",
            self.path,
            "python={}".format(self.python),
        )
        print('Executing Conda: {}'.format(command.serialize()))
        output = command.get_output(stderr=DEVNULL)
        match = re.search(
            r"\W*(.*activate) ({})".format(re.escape(str(self.path))), output
        )
        self.source = self.pip.source = (
            tuple(match.group(1).split()) + (match.group(2),)
            if match
            else ("conda", "activate", self.path.as_posix())
        )

        conda_env = self._get_conda_sh()
        if conda_env.is_file() and not is_windows_platform():
            self.source = self.pip.source = CommandSequence(('source', conda_env.as_posix()), self.source)

        return self

    def _init_existing_environment(self, conda_pre_build_env_path, is_readonly=True):
        print("Using pre-existing Conda environment from {}".format(conda_pre_build_env_path))
        self.path = Path(conda_pre_build_env_path)
        self.source = ("conda", "activate", self.path.as_posix())
        conda_env = self._get_conda_sh()
        self.source = CommandSequence(('source', conda_env.as_posix()), self.source)

        conda_packages_json = json.loads(
            self._run_command((self.conda, "list", "--json", "-p", self.path), raw=True))

        try:
            for package in conda_packages_json:
                if package.get("name") == "python" and package.get("version"):
                    self.python = ".".join(package.get("version").split(".")[:2])
                    print("Existing conda environment, found python version {}".format(self.python))
                    break
        except Exception as ex:
            print("WARNING: failed detecting existing conda python version: {}".format(ex))

        self.pip = CondaPip(
            session=self.session,
            source=self.source,
            python=self.python,
            requirements_manager=self.requirements_manager,
            path=self.path,
        )
        self.pip.source = self.source

        self.env_read_only = is_readonly

    def remove(self):
        """
        Delete a conda environment.
        Use 'conda env remove', then 'rm_tree' to be safe.

        Conda seems to load "vcruntime140.dll" from all its environment on startup.
        This means environment have to be deleted using 'conda env remove'.
        If necessary, conda can be fooled into deleting a partially-deleted environment by creating an empty file
        in '<ENV>\\conda-meta\\history' (value found in 'conda.gateways.disk.test.PREFIX_MAGIC_FILE').
        Otherwise, it complains that said directory is not a conda environment.

        See: https://github.com/conda/conda/issues/7682
        """
        try:
            self._run_command(("env", "remove", "-p", self.path))
        except Exception:
            pass
        rm_tree(self.path)
        # if we failed removing the path, change it's name
        if is_windows_platform() and Path(self.path).exists():
            try:
                Path(self.path).rename(Path(self.path).as_posix() + '_' + str(time()))
            except Exception:
                pass

    def _install_from_file(self, path):
        """
        Install packages from requirement file.
        """
        self._install("--file", path)

    def _install(self, *args):
        # type: (*PathLike) -> ()
        # if we are in read only mode, do not install anything
        if self.env_read_only:
            print('Conda environment in read-only mode, skipping package installing: {}'.format(args))
            return
        channels_args = tuple(
            chain.from_iterable(("-c", channel) for channel in self.extra_channels)
        )
        self._run_command(("install", "-p", self.path) + channels_args + args)

    def _get_pip_packages(self, packages):
        # type: (Iterable[Text]) -> Sequence[Text]
        """
        Return subset of ``packages`` which are not available on conda
        """
        pips = []
        while True:
            with self.temp_file("conda_reqs", packages) as path:
                try:
                    self._install_from_file(path)
                except PackageNotFoundError as e:
                    pips.append(e.pkg)
                    packages = _package_diff(path, {e.pkg})
                else:
                    break
        return pips

    def install_packages(self, *packages):
        # type: (*Text) -> ()
        return self._install(*packages)

    def uninstall_packages(self, *packages):
        # if we are in read only mode, do not uninstall anything
        if self.env_read_only:
            print('Conda environment in read-only mode, skipping package uninstalling: {}'.format(packages))
            return ''
        return self._run_command(("uninstall", "-p", self.path))

    def install_from_file(self, path):
        """
        Try to install packages from conda. Install packages which are not available from conda with pip.
        """
        requirements = {}
        # assume requirements.txt
        with open(path, 'rt') as f:
            requirements['pip'] = f.read()
        self.load_requirements(requirements)

    def freeze(self, freeze_full_environment=False):
        requirements = self.pip.freeze()
        req_lines = []
        conda_lines = []

        # noinspection PyBroadException
        try:
            pip_lines = requirements['pip']
            conda_packages_json = json.loads(
                self._run_command((self.conda, "list", "--json", "-p", self.path), raw=True))
            for r in conda_packages_json:
                # check if this is a pypi package, if it is, leave it outside
                if not r.get('channel') or r.get('channel') == 'pypi':
                    name = (r['name'].replace('-', '_'), r['name'])
                    pip_req_line = [l for l in pip_lines
                                    if l.split('==', 1)[0].strip() in name or l.split('@', 1)[0].strip() in name]
                    if pip_req_line and \
                            ('@' not in pip_req_line[0] or
                             not pip_req_line[0].split('@', 1)[1].strip().startswith('file://')):
                        req_lines.append(pip_req_line[0])
                        continue

                    req_lines.append(
                        '{}=={}'.format(name[1], r['version']) if r.get('version') else '{}'.format(name[1]))
                    continue

                # check if we have it in our required packages
                name = r['name']
                # hack support pytorch/torch different naming convention
                if name == 'pytorch':
                    name = 'torch'
                # skip over packages with _
                if name.startswith('_'):
                    continue
                conda_lines.append('{}=={}'.format(name, r['version']) if r.get('version') else '{}'.format(name))
            # make sure we see the conda packages, put them into the pip as well
            if conda_lines:
                req_lines = ['# Conda Packages', ''] + conda_lines + ['', '# pip Packages', ''] + req_lines

            requirements['pip'] = req_lines
            requirements['conda'] = conda_lines
        except Exception:
            pass

        if freeze_full_environment:
            # noinspection PyBroadException
            try:
                conda_env_json = json.loads(
                    self._run_command((self.conda, "env", "export", "--json", "-p", self.path), raw=True))
                conda_env_json.pop('name', None)
                conda_env_json.pop('prefix', None)
                conda_env_json.pop('channels', None)
                requirements['conda_env_json'] = json.dumps(conda_env_json)
            except Exception:
                pass

        return requirements

    def _load_conda_full_env(self, conda_env_dict, requirements):
        # noinspection PyBroadException
        try:
            cuda_version = int(self.session.config.get('agent.cuda_version', 0))
        except Exception:
            cuda_version = 0

        conda_env_dict['channels'] = self.extra_channels
        if 'dependencies' not in conda_env_dict:
            conda_env_dict['dependencies'] = []
        new_dependencies = OrderedDict()
        pip_requirements = None
        for line in conda_env_dict['dependencies']:
            if isinstance(line, dict):
                pip_requirements = line.pop('pip', None)
                continue
            name = line.strip().split('=', 1)[0].lower()
            if name == 'pip':
                continue
            elif name == 'python':
                line = 'python={}'.format('.'.join(line.split('=')[1].split('.')[:2]))
            elif name == 'tensorflow-gpu' and cuda_version == 0:
                line = 'tensorflow={}'.format(line.split('=')[1])
            elif name == 'tensorflow' and cuda_version > 0:
                line = 'tensorflow-gpu={}'.format(line.split('=')[1])
            elif name in ('cupti', 'cudnn'):
                # cudatoolkit should pull them based on the cudatoolkit version
                continue
            elif name.startswith('_'):
                continue
            new_dependencies[line.split('=', 1)[0].strip()] = line

        # fix packages:
        conda_env_dict['dependencies'] = list(new_dependencies.values())

        with self.temp_file("conda_env", yaml.dump(conda_env_dict), suffix=".yml") as name:
            print('Conda: Trying to install requirements:\n{}'.format(conda_env_dict['dependencies']))
            result = self._run_command(
                ("env", "update", "-p", self.path, "--file", name)
            )

        # check if we need to remove specific packages
        bad_req = self._parse_conda_result_bad_packges(result)
        if bad_req:
            print('failed installing the following conda packages: {}'.format(bad_req))
            return False

        if pip_requirements:
            # create a list of vcs packages that we need to replace in the pip section
            vcs_reqs = {}
            if 'pip' in requirements:
                pip_lines = requirements['pip'].splitlines() \
                    if isinstance(requirements['pip'], six.string_types) else requirements['pip']
                for line in pip_lines:
                    try:
                        marker = list(parse(line))
                    except Exception:
                        marker = None
                    if not marker:
                        continue

                    m = MarkerRequirement(marker[0])
                    if m.vcs:
                        vcs_reqs[m.name] = m
            try:
                pip_req_str = [str(vcs_reqs.get(r.split('=', 1)[0], r)) for r in pip_requirements
                               if not r.startswith('pip=') and not r.startswith('virtualenv=')]
                print('Conda: Installing requirements: step 2 - using pip:\n{}'.format(pip_req_str))
                PackageManager._selected_manager = self.pip
                self.pip.load_requirements({'pip': '\n'.join(pip_req_str)})
            except Exception as e:
                print(e)
                raise e
            finally:
                PackageManager._selected_manager = self

        self.requirements_manager.post_install(self.session, package_manager=self)

    def load_requirements(self, requirements):
        # if we are in read only mode, do not uninstall anything
        if self.env_read_only:
            print('Conda environment in read-only mode, skipping requirements installation.')
            return None

        # if we have a full conda environment, use it and pass the pip to pip
        if requirements.get('conda_env_json'):
            # noinspection PyBroadException
            try:
                conda_env_json = json.loads(requirements.get('conda_env_json'))
                print('Conda restoring full yaml environment')
                return self._load_conda_full_env(conda_env_json, requirements)
            except Exception:
                print('Could not load fully stored conda environment, falling back to requirements')

        # create new environment file
        conda_env = dict()
        conda_env['channels'] = self.extra_channels
        reqs = []
        if isinstance(requirements['pip'], six.string_types):
            requirements['pip'] = requirements['pip'].split('\n')
        if isinstance(requirements.get('conda'), six.string_types):
            requirements['conda'] = requirements['conda'].split('\n')
        has_torch = False
        has_matplotlib = False
        has_cudatoolkit = False
        cuda_version_full = 0
        # noinspection PyBroadException
        try:
            # notice this is an integer version: 112 (means 11.2)
            cuda_version = str(self.session.config.get('agent.cuda_version', "")).strip()
            if not cuda_version:
                cuda_version = 0
            else:
                cuda_version_full = convert_cuda_version_to_float_single_digit_str(cuda_version)
                cuda_version = int(convert_cuda_version_to_int_10_base_str(cuda_version))
        except Exception:
            cuda_version = 0

        # notice 'conda' entry with empty string is a valid conda requirements list, it means pip only
        # this should happen if experiment was executed on non-conda machine or old trains client
        conda_supported_req = requirements['pip'] if requirements.get('conda', None) is None else requirements['conda']
        conda_supported_req_names = []
        pip_requirements = []
        for r in conda_supported_req:
            try:
                marker = list(parse(r))
            except:
                marker = None
            if not marker:
                continue

            m = MarkerRequirement(marker[0])
            m.validate_local_file_ref()
            # conda does not support version control links
            if m.vcs:
                pip_requirements.append(m)
                continue
            # Skip over pip
            if m.name in ('pip', 'virtualenv', ):
                continue
            # python version, only major.minor
            if m.name == 'python' and m.specs:
                m.specs = [(m.specs[0][0], '.'.join(m.specs[0][1].split('.')[:2])), ]
                if '.' not in m.specs[0][1]:
                    continue

            if m.name.lower() in ('cudatoolkit', 'cuda-toolkit'):
                # skip cuda if we are running on CPU
                if not cuda_version:
                    continue

                has_cudatoolkit = True
                # cuda version, only major.minor
                requested_cuda_version = '.'.join(m.specs[0][1].split('.')[:2])
                # make sure that the cuda_version we support can install the requested cuda (major version)
                if int(float(requested_cuda_version)) > int(float(cuda_version)/10.0):
                    continue
                m.specs = [(m.specs[0][0], str(requested_cuda_version)), ]

            conda_supported_req_names.append(m.name.lower())
            if m.req.name.lower() == 'matplotlib':
                has_matplotlib = True
            elif m.req.name.lower().startswith('torch'):
                has_torch = True

            if m.req.name.lower() in ('torch', 'pytorch'):
                has_torch = True
                m.req.name = 'pytorch'

            if m.req.name.lower() in ('tensorflow_gpu', 'tensorflow-gpu', 'tensorflow'):
                has_torch = True
                m.req.name = 'tensorflow-gpu' if cuda_version > 0 else 'tensorflow'

            # push the clearml packages into the pip_requirements
            if "clearml" in m.req.name and "clearml" not in self.extra_channels:
                if self.session.debug_mode:
                    print("info: moving `{}` packages to `pip` section".format(m.req))
                pip_requirements.append(m)
                continue

            reqs.append(m)

        if not has_cudatoolkit and cuda_version:
            # nvidia channel is using `cuda-toolkit` and has newer versions of cuda,
            # older cuda can be picked from conda-forge (<12)
            if "nvidia" in self.extra_channels:
                m = MarkerRequirement(Requirement.parse("cuda-toolkit == {}".format(cuda_version_full)))
            else:
                m = MarkerRequirement(Requirement.parse("cudatoolkit == {}".format(cuda_version_full)))
            has_cudatoolkit = True
            reqs.append(m)

        # if we have a conda list, the rest should be installed with pip,
        # this means  any experiment that was executed with pip environment,
        # will be installed using pip
        if requirements.get('conda', None) is not None:
            for r in requirements['pip']:
                try:
                    marker = list(parse(r))
                except:
                    marker = None
                if not marker:
                    continue

                m = MarkerRequirement(marker[0])
                # remove local files reference if it does not exist (leave the package name)
                m.validate_local_file_ref()

                m_name = (m.name or '').lower()
                if m_name in conda_supported_req_names:
                    # this package is in the conda list,
                    # make sure that if we changed version and we match it in conda
                    ## conda_supported_req_names.remove(m_name)
                    for cr in reqs:
                        if m_name.lower().replace('_', '-') == cr.name.lower().replace('_', '-'):
                            # match versions
                            cr.specs = m.specs
                            # # conda always likes "-" not "_" but only on pypi packages
                            # cr.name = cr.name.lower().replace('_', '-')
                            break
                else:
                    # not in conda, it is a pip package
                    pip_requirements.append(m)
                    if m_name == 'matplotlib':
                        has_matplotlib = True

        # Conda requirements Hacks:
        if has_matplotlib:
            reqs.append(MarkerRequirement(Requirement.parse('graphviz')))
            reqs.append(MarkerRequirement(Requirement.parse('python-graphviz')))
            reqs.append(MarkerRequirement(Requirement.parse('kiwisolver')))

        # remove specific cudatoolkit, it should have being preinstalled.
        # allow to override default cudatoolkit, but not the derivative packages, cudatoolkit should pull them
        reqs = [r for r in reqs if r.name not in ('cudnn', 'cupti')]

        if has_torch and cuda_version == 0:
            reqs.append(MarkerRequirement(Requirement.parse('cpuonly')))

        # make sure we have no double entries
        reqs = list(OrderedDict((r.name, r) for r in reqs).values())

        # conform conda packages (version/name)
        for r in reqs:
            # change _ to - in name but not the prefix _ (as this is conda prefix)
            if r.name and not r.name.startswith('_') and not requirements.get('conda', None):
                r.name = r.name.replace('_', '-')

            if has_cudatoolkit and r.specs and len(r.specs[0]) > 1 and r.name in ('cudatoolkit', 'cuda-toolkit'):
                # select specific cuda version if it came from the requirements
                r.specs = [(r.specs[0][0].replace('==', '='), r.specs[0][1].split('.post')[0])]
            elif r.specs and r.specs[0] and len(r.specs[0]) > 1:
                # remove .post from version numbers it fails with ~= version, and change == to ~=
                r.specs = [(s[0].replace('==', '~='), s[1].split('.post')[0]) for s in r.specs]

        while reqs:
            # notice, we give conda more freedom in version selection, to help it choose best combination
            def clean_ver(ar):
                markers = None
                if ar.marker:
                    # check if we really need it based on python version
                    ar = self._parse_package_marker_match_python_ver(marker_req=ar)
                    if not ar:
                        # empty lines should be skipped
                        return ""
                    # if we do make sure we note that we ignored markers
                    print("WARNING: ignoring marker in `{}`".format(ar.tostr()))
                    markers = False
                if ar.specs:
                    ar.specs = [(s[0], s[1] + '.0' if '.' not in s[1] else s[1]) for s in ar.specs]
                return ar.tostr(markers=markers)
            conda_env['dependencies'] = [clean_ver(r) for r in reqs if clean_ver(r)]
            with self.temp_file("conda_env", yaml.dump(conda_env), suffix=".yml") as name:
                print('Conda: Trying to install requirements:\n{}'.format(conda_env['dependencies']))
                if self.session.debug_mode:
                    print('{}:\n{}'.format(name, yaml.dump(conda_env)))
                result = self._run_command(
                    ("env", "update", "-p", self.path, "--file", name)
                )
            # check if we need to remove specific packages
            bad_req = self._parse_conda_result_bad_packges(result)
            if not bad_req:
                break

            solved = False
            for bad_r in bad_req:
                name = bad_r.split('[')[0].split('=')[0].split('~')[0].split('<')[0].split('>')[0]
                # look for name in requirements
                for r in reqs:
                    if r.name.lower() == name.lower():
                        pip_requirements.append(r)
                        reqs.remove(r)
                        solved = True
                        break

            # we couldn't remove even one package,
            # nothing we can do but try pip
            if not solved:
                pip_requirements.extend(reqs)
                break

        if pip_requirements:
            try:
                pip_req_str = [r.tostr() for r in pip_requirements if r.name not in ('pip', 'virtualenv', )]
                print('Conda: Installing requirements: step 2 - using pip:\n{}'.format(pip_req_str))
                PackageManager._selected_manager = self.pip
                if self.session.debug_mode:
                    print('pip requirements.txt:\n{}'.format('\n'.join(pip_req_str)))
                self.pip.load_requirements({'pip': '\n'.join(pip_req_str)})
            except Exception as e:
                print(e)
                raise e
            finally:
                PackageManager._selected_manager = self

        self.requirements_manager.post_install(self.session, package_manager=self)
        return True

    def _parse_conda_result_bad_packges(self, result_dict):
        if not result_dict:
            return None

        if 'bad_deps' in result_dict and result_dict['bad_deps']:
            return result_dict['bad_deps']

        if result_dict.get('error'):
            error_lines = result_dict['error'].split('\n')
            if error_lines[0].strip().lower().startswith("unsatisfiableerror:"):
                empty_lines = [i for i, l in enumerate(error_lines) if not l.strip()]
                if len(empty_lines) >= 2:
                    deps = error_lines[empty_lines[0]+1:empty_lines[1]]
                    try:
                        return yaml.load('\n'.join(deps), Loader=yaml.SafeLoader)
                    except:
                        return None
        return None

    def _run_command(self, command, raw=False, **kwargs):
        # type: (Iterable[Text], bool, Any) -> Union[Dict, Text]
        """
        Run a conda command, returning JSON output.
        The command is prepended with 'conda' and run with JSON output flags.
        :param command: command to run
        :param raw: return text output and don't change command
        :param kwargs: kwargs for Argv.get_output()
        :return: JSON output or text output
        """
        def escape_ansi(line):
            ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
            return ansi_escape.sub('', line)

        # make sure we are not running it with our own PYTHONPATH
        env = dict(**os.environ)
        env.pop('PYTHONPATH', None)

        command = Argv(*command)  # type: Executable
        if not raw:
            command = (self.conda,) + command + ("--quiet", "--json")
        try:
            print('Executing Conda: {}'.format(command.serialize()))
            result = command.get_output(stdin=DEVNULL, env=env, **kwargs)
            if self.session.debug_mode:
                print(result)
        except Exception as e:
            result = e.output if hasattr(e, 'output') else ''
            if self.session.debug_mode:
                print(result)
            if raw:
                raise
        if raw:
            return result

        result = json.loads(escape_ansi(result)) if result else {}
        if result.get('success', False):
            print('Pass')
        elif result.get('error'):
            print('Conda error: {}'.format(result.get('error')))
        return result

    def get_python_command(self, extra=()):
        if not self.source:
            self._init_existing_environment(self.path)
        return CommandSequence(self.source, self.pip.get_python_command(extra=extra))

    def _get_conda_sh(self):
        # type () -> Path
        base_conda_env = Path(self.conda).parent.parent / 'etc' / 'profile.d' / 'conda.sh'
        if base_conda_env.is_file():
            return base_conda_env
        for path in os.environ.get('PATH', '').split(select_for_platform(windows=';', linux=':')):
            conda = find_executable("conda", path=path)
            if not conda:
                continue
            conda_env = Path(conda).parent.parent / 'etc' / 'profile.d' / 'conda.sh'
            if conda_env.is_file():
                return conda_env
        return base_conda_env

    def add_cached_venv(self, *args, **kwargs):
        """
        Copy the local venv folder into the venv cache (keys are based on the requirements+python+docker).
        """
        # do not cache if this is a base conda environment
        if self.conda_env_as_base_docker or self.use_conda_base_env:
            return
        return super().add_cached_venv(*args, **kwargs)

    def get_cached_venv(self, *args, **kwargs):
        """
        Copy a cached copy of the venv (based on the requirements) into destination_folder.
        Return None if failed or cached entry does not exist
        """
        # do not cache if this is a base conda environment
        if self.conda_env_as_base_docker or self.use_conda_base_env:
            return
        return super().get_cached_venv(*args, **kwargs)


# enable hashing with cmp=False because pdb fails on un-hashable exceptions
exception = attrs(str=True, cmp=False)


@exception
class CondaException(Exception, NonStrictAttrs):
    command = attrib()
    message = attrib(default=None)


@exception
class UnknownCondaError(CondaException):
    data = attrib(default=Factory(dict))


@exception
class PackagesNotFoundError(CondaException):
    """
    Conda 4.5 exception - this reports all missing packages.
    """

    packages = attrib(default=())


@exception
class PackageNotFoundError(CondaException):
    """
    Conda 4.3 exception - this reports one missing package at a time,
                          as a singleton YAML list.
    """

    pkg = attrib(default="", converter=lambda val: yaml.load(val, Loader=yaml.SafeLoader)[0].replace(" ", ""))
