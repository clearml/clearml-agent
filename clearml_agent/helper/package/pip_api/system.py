import os
import sys
from itertools import chain
from pathlib import Path
from typing import Text, Optional

from clearml_agent.definitions import PIP_EXTRA_INDICES, PROGRAM_NAME, ENV_PIP_EXTRA_INSTALL_FLAGS
from clearml_agent.helper.package.base import PackageManager
from clearml_agent.helper.process import Argv, DEVNULL
from clearml_agent.session import Session


class SystemPip(PackageManager):

    def __init__(self, interpreter=None, session=None):
        # type: (Optional[Text], Optional[Session]) -> ()
        """
        Program interface to the system pip.
        """
        super(SystemPip, self).__init__()
        self._bin = Path(interpreter or sys.executable)
        self.session = session

    @property
    def bin(self):
        return self._bin

    def create(self):
        pass

    def remove(self):
        pass

    def install_from_file(self, path):
        try:
            self.run_with_env(('install', '-r', path) + self.install_flags(), cwd=self.cwd)
        except Exception:
            print("ERROR: installing pip requirements failed, requirements.txt:\n{}\n".format(
                "\n".join([line for line in Path(path).read_text().splitlines() if line.strip()])
            ))
            raise

    def install_packages(self, *packages):
        self._install(*(packages + self.install_flags()))

    def _install(self, *args):
        self.run_with_env(('install',) + args, cwd=self.cwd)

    def uninstall_packages(self, *packages):
        self.run_with_env(('uninstall', '-y') + packages)

    def download_package(self, package, cache_dir):
        self.run_with_env(
            (
                'download',
                package,
                '--dest', cache_dir,
                '--no-deps',
            ) + self.download_flags()
        )

    def load_requirements(self, requirements):
        requirements = requirements.get('pip') if isinstance(requirements, dict) else requirements
        if not requirements:
            return
        with self.temp_file('cached-reqs', requirements) as path:
            self.install_from_file(path)

    def uninstall(self, package):
        self.run_with_env(('uninstall', '-y', package))

    def freeze(self, freeze_full_environment=False):
        """
        pip freeze to all install packages except the running program
        :return: Dict contains pip as key and pip's packages to install
        :rtype: Dict[str: List[str]]
        """
        packages = self.run_with_env(
            ('freeze',) if not freeze_full_environment else ('freeze', '--all'), output=True).splitlines()
        packages_without_program = [package for package in packages if PROGRAM_NAME not in package]
        return {'pip': packages_without_program}

    def run_with_env(self, command, output=False, **kwargs):
        """
        Run a shell command using environment from a virtualenv script
        :param command: command to run
        :type command: Iterable[Text]
        :param output: return output
        :param kwargs: kwargs for get_output/check_output command
        """
        command = self._make_command(command)
        # make sure we are not running it with our own PYTHONPATH
        env = dict(**os.environ)
        env.pop('PYTHONPATH', None)

        # Debug print
        if self.session.debug_mode:
            print(command)

        return (command.get_output if output else command.check_call)(stdin=DEVNULL, env=env, **kwargs)

    def _make_command(self, command):
        return Argv(self.bin, '-m', 'pip', '--disable-pip-version-check', *command)

    def install_flags(self):
        base_args = tuple(self._base_install_flags or []) + tuple(
            chain.from_iterable(('--extra-index-url', x) for x in PIP_EXTRA_INDICES)
        )

        extra_pip_flags = \
            ENV_PIP_EXTRA_INSTALL_FLAGS.get() or \
            self.session.config.get("agent.package_manager.extra_pip_install_flags", None)

        return (base_args + tuple(extra_pip_flags)) if extra_pip_flags else base_args

    def download_flags(self):
        indices_args = tuple(
            chain.from_iterable(('--extra-index-url', x) for x in PIP_EXTRA_INDICES)
        )

        return indices_args
