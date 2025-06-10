from copy import deepcopy
from functools import wraps

from ..._vendor import attr
import sys
import os
from ..._vendor.pathlib2 import Path

from clearml_agent.definitions import ENV_AGENT_FORCE_POETRY
from clearml_agent.helper.process import Argv, DEVNULL, check_if_command_exists
from clearml_agent.session import Session, POETRY


def prop_guard(prop, log_prop=None):
    assert isinstance(prop, property)
    assert not log_prop or isinstance(log_prop, property)

    def decorator(func):
        message = "%s:%s calling {}, {} = %s".format(
            func.__name__, prop.fget.__name__
        )

        @wraps(func)
        def new_func(self, *args, **kwargs):
            prop_value = prop.fget(self)
            if log_prop:
                log_prop.fget(self).debug(
                    message,
                    type(self).__name__,
                    "" if prop_value else " not",
                    prop_value,
                )
            if prop_value:
                return func(self, *args, **kwargs)

        return new_func

    return decorator


class PoetryConfig:

    def __init__(self, session):
        # type: (Session, str) -> None
        self.session = session
        self._log = session.get_logger(__name__)
        self._python = sys.executable  # default, overwritten from session config in initialize()
        self._initialized = False

    @property
    def log(self):
        return self._log

    @property
    def enabled(self):
        return ENV_AGENT_FORCE_POETRY.get() or self.session.config["agent.package_manager.type"] == POETRY

    _guard_enabled = prop_guard(enabled, log)

    def run(self, *args, **kwargs):
        func = kwargs.pop("func", Argv.get_output)
        kwargs.setdefault("stdin", DEVNULL)
        kwargs['env'] = deepcopy(os.environ)
        if 'VIRTUAL_ENV' in kwargs['env'] or 'CONDA_PREFIX' in kwargs['env']:
            kwargs['env'].pop('VIRTUAL_ENV', None)
            kwargs['env'].pop('CONDA_PREFIX', None)
            kwargs['env'].pop('PYTHONPATH', None)
            if hasattr(sys, "real_prefix") and hasattr(sys, "base_prefix"):
                path = ':'+kwargs['env']['PATH']
                path = path.replace(':'+sys.base_prefix, ':'+sys.real_prefix, 1)
                kwargs['env']['PATH'] = path

        if self.session and self.session.config and args and args[0] == "install":
            extra_args = self.session.config.get("agent.package_manager.poetry_install_extra_args", None)
            if extra_args:
                args = args + tuple(extra_args)

        if check_if_command_exists("poetry"):
            argv = Argv("poetry", *args)
        else:
            argv = Argv(self._python, "-m", "poetry", *args)
        self.log.debug("running: %s", argv)
        return func(argv, **kwargs)

    def _config(self, *args, **kwargs):
        return self.run("config", *args, **kwargs)

    @_guard_enabled
    def initialize(self, cwd=None):
        if not self._initialized:
            # use correct python version -- detected in Worker.install_virtualenv() and written to
            # session
            if self.session.config.get("agent.python_binary", None):
                self._python = self.session.config.get("agent.python_binary")

            if self.session.config.get("agent.package_manager.poetry_version", None) is not None:
                version = str(self.session.config.get("agent.package_manager.poetry_version"))

                # get poetry version
                version = version.replace(' ', '')
                if ('=' in version) or ('~' in version) or ('<' in version) or ('>' in version):
                    version = version
                elif version:
                    version = "==" + version
                # (we are not running it yet)
                argv = Argv(self._python, "-m", "pip", "install", "poetry{}".format(version),
                            "--upgrade", "--disable-pip-version-check")
                # this is just for beauty and checks, we already set the verion in the Argv
                if not version:
                    version = "latest"
            else:
                # mark to install poetry if not already installed (we are not running it yet)
                argv = Argv(self._python, "-m", "pip", "install", "poetry", "--disable-pip-version-check")
                version = ""

            # first upgrade pip if we need to
            try:
                from clearml_agent.helper.package.pip_api.venv import VirtualenvPip
                pip = VirtualenvPip(
                    session=self.session, python=self._python,
                    requirements_manager=None, path=None, interpreter=self._python)
                pip.upgrade_pip()
            except Exception as ex:
                self.log.warning("failed upgrading pip: {}".format(ex))

            # check if we do not have a specific version and poetry is found skip installation
            if not version and check_if_command_exists("poetry"):
                print("Notice: Poetry was found, no specific version required, skipping poetry installation")
            else:
                print('Installing / Upgrading Poetry package to {}'.format(version))
                # now install poetry
                try:
                    print(argv.get_output())
                except Exception as ex:
                    self.log.warning("failed installing poetry: {}".format(ex))

            # now setup poetry
            self._initialized = True
            try:
                self._config("--local", "virtualenvs.in-project",  "true", cwd=cwd)
                # self._config("repositories.{}".format(self.REPO_NAME), PYTHON_INDEX)
                # self._config("http-basic.{}".format(self.REPO_NAME), *PYTHON_INDEX_CREDENTIALS)
            except Exception as ex:
                print("Exception: {}\nError: Failed configuring Poetry virtualenvs.in-project".format(ex))
                raise

    def get_api(self, path):
        # type: (Path) -> PoetryAPI
        return PoetryAPI(self, path)


@attr.s
class PoetryAPI(object):
    config = attr.ib(type=PoetryConfig)
    path = attr.ib(type=Path, converter=Path)

    INDICATOR_FILES = "pyproject.toml", "poetry.lock"

    def install(self):
        # type: () -> bool
        if self.enabled:
            self.config.run("install", "-n", cwd=str(self.path), func=Argv.check_call)
            return True
        return False

    @property
    def enabled(self):
        return self.config.enabled and (
            any((self.path / indicator).exists() for indicator in self.INDICATOR_FILES)
        )

    def freeze(self, freeze_full_environment=False):
        lines = self.config.run("show", cwd=str(self.path)).splitlines()
        lines = [[p for p in line.split(' ') if p] for line in lines]
        return {"pip": [parts[0]+'=='+parts[1]+' # '+' '.join(parts[2:]) for parts in lines]}

    def get_python_command(self, extra):
        if check_if_command_exists("poetry"):
            return Argv("poetry", "run", "python", *extra)
        else:
            return Argv(self.config._python, "-m", "poetry", "run", "python", *extra)

    def upgrade_pip(self, *args, **kwargs):
        pass

    def set_selected_package_manager(self, *args, **kwargs):
        pass

    def out_of_scope_install_package(self, *args, **kwargs):
        pass

    def install_from_file(self, *args, **kwargs):
        pass
