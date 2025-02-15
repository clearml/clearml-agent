from copy import deepcopy
from functools import wraps

from ..._vendor import attr
import sys
import os
from ..._vendor.pathlib2 import Path

from clearml_agent.definitions import ENV_AGENT_FORCE_UV
from clearml_agent.helper.base import select_for_platform
from clearml_agent.helper.process import Argv, DEVNULL, check_if_command_exists
from clearml_agent.session import Session, UV


def prop_guard(prop, log_prop=None):
    assert isinstance(prop, property)
    assert not log_prop or isinstance(log_prop, property)

    def decorator(func):
        message = "%s:%s calling {}, {} = %s".format(func.__name__, prop.fget.__name__)

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


class UvConfig:
    def __init__(self, session):
        # type: (Session, str) -> None
        self.session = session
        self._log = session.get_logger(__name__)
        self._python = (
            sys.executable
        )  # default, overwritten from session config in initialize()
        self._initialized = False

    @property
    def log(self):
        return self._log

    @property
    def enabled(self):
        return (
            ENV_AGENT_FORCE_UV.get()
            or self.session.config["agent.package_manager.type"] == UV
        )

    _guard_enabled = prop_guard(enabled, log)

    def run(self, *args, **kwargs):
        func = kwargs.pop("func", Argv.get_output)
        kwargs.setdefault("stdin", DEVNULL)
        kwargs["env"] = deepcopy(os.environ)
        if "VIRTUAL_ENV" in kwargs["env"] or "CONDA_PREFIX" in kwargs["env"]:
            kwargs["env"].pop("VIRTUAL_ENV", None)
            kwargs["env"].pop("CONDA_PREFIX", None)
            kwargs["env"].pop("PYTHONPATH", None)
            if hasattr(sys, "real_prefix") and hasattr(sys, "base_prefix"):
                path = ":" + kwargs["env"]["PATH"]
                path = path.replace(":" + sys.base_prefix, ":" + sys.real_prefix, 1)
                kwargs["env"]["PATH"] = path

        if self.session and self.session.config and args and args[0] == "sync":
            # Set the cache dir to venvs dir
            cache_dir = self.session.config.get("agent.venvs_dir", None)
            if cache_dir is not None:
                os.environ["UV_CACHE_DIR"] = cache_dir
            
            extra_args = self.session.config.get(
                "agent.package_manager.uv_sync_extra_args", None
            )
            if extra_args:
                args = args + tuple(extra_args)

        if check_if_command_exists("uv"):
            argv = Argv("uv", *args)
        else:
            argv = Argv(self._python, "-m", "uv", *args)
        self.log.debug("running: %s", argv)
        return func(argv, **kwargs)

    @_guard_enabled
    def initialize(self, cwd=None):
        if not self._initialized:
            # use correct python version -- detected in Worker.install_virtualenv() and written to
            # session
            if self.session.config.get("agent.python_binary", None):
                self._python = self.session.config.get("agent.python_binary")

            if (
                self.session.config.get("agent.package_manager.uv_version", None)
                is not None
            ):
                version = str(
                    self.session.config.get("agent.package_manager.uv_version")
                )

                # get uv version
                version = version.replace(" ", "")
                if (
                    ("=" in version)
                    or ("~" in version)
                    or ("<" in version)
                    or (">" in version)
                ):
                    version = version
                elif version:
                    version = "==" + version
                # (we are not running it yet)
                argv = Argv(
                    self._python,
                    "-m",
                    "pip",
                    "install",
                    "uv{}".format(version),
                    "--upgrade",
                    "--disable-pip-version-check",
                )
                # this is just for beauty and checks, we already set the verion in the Argv
                if not version:
                    version = "latest"
            else:
                # mark to install uv if not already installed (we are not running it yet)
                argv = Argv(
                    self._python,
                    "-m",
                    "pip",
                    "install",
                    "uv",
                    "--disable-pip-version-check",
                )
                version = ""

            # first upgrade pip if we need to
            try:
                from clearml_agent.helper.package.pip_api.venv import VirtualenvPip

                pip = VirtualenvPip(
                    session=self.session,
                    python=self._python,
                    requirements_manager=None,
                    path=None,
                    interpreter=self._python,
                )
                pip.upgrade_pip()
            except Exception as ex:
                self.log.warning("failed upgrading pip: {}".format(ex))

            # check if we do not have a specific version and uv is found skip installation
            if not version and check_if_command_exists("uv"):
                print(
                    "Notice: uv was found, no specific version required, skipping uv installation"
                )
            else:
                print("Installing / Upgrading uv package to {}".format(version))
                # now install uv
                try:
                    print(argv.get_output())
                except Exception as ex:
                    self.log.warning("failed installing uv: {}".format(ex))

            # all done.
            self._initialized = True

    def get_api(self, path):
        # type: (Path) -> UvAPI
        return UvAPI(self, path)


@attr.s
class UvAPI(object):
    config = attr.ib(type=UvConfig)
    path = attr.ib(type=Path, converter=Path)

    INDICATOR_FILES = "pyproject.toml", "uv.lock"

    def install(self):
        # type: () -> bool
        if self.enabled:
            self.config.run("sync", "--locked", cwd=str(self.path), func=Argv.check_call)
            return True
        return False

    @property
    def enabled(self):
        return self.config.enabled and (
            any((self.path / indicator).exists() for indicator in self.INDICATOR_FILES)
        )

    def freeze(self, freeze_full_environment=False):
        python = Path(self.path) / ".venv" / select_for_platform(linux="bin/python", windows="scripts/python.exe")
        lines = self.config.run("pip", "freeze", "--python", str(python), cwd=str(self.path)).splitlines()
        # fix local filesystem reference in freeze
        from clearml_agent.external.requirements_parser.requirement import Requirement
        packages = [Requirement.parse(p) for p in lines]
        for p in packages:
            if p.local_file and p.editable:
                p.path = str(Path(p.path).relative_to(self.path))
                p.line = "-e {}".format(p.path)

        return {
            "pip": [p.line for p in packages]
        }

    def get_python_command(self, extra):
        if check_if_command_exists("uv"):
            return Argv("uv", "run", "python", *extra)
        else:
            return Argv(self.config._python, "-m", "uv", "run", "python", *extra)

    def upgrade_pip(self, *args, **kwargs):
        pass

    def set_selected_package_manager(self, *args, **kwargs):
        pass

    def out_of_scope_install_package(self, *args, **kwargs):
        pass

    def install_from_file(self, *args, **kwargs):
        pass
