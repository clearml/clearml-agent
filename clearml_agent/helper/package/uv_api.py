from copy import deepcopy, copy
from functools import wraps

from ..._vendor import attr
import sys
import os
from ..._vendor.pathlib2 import Path

import shutil

from clearml_agent.definitions import ENV_AGENT_FORCE_UV
from clearml_agent.helper.base import python_version_string, rm_tree
from clearml_agent.helper.package.base import get_specific_package_version
from clearml_agent.helper.package.pip_api.venv import VirtualenvPip
from clearml_agent.helper.process import Argv, DEVNULL, check_if_command_exists
from clearml_agent.session import UV


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
    USE_UV_BIN = False

    def __init__(self, session):
        # type: (str) -> None
        self.session = session
        self._log = session.get_logger(__name__)
        self._python = (
            sys.executable
        )  # default, overwritten from session config in initialize()
        self._initialized = False
        self._api = None
        self._cwd = None
        self._is_sync = False
        self._uv_version = None
        self._uv_bin = None
        self._venv_python = None
        self._req_python_version = None

    def set_uv_version(self, version):
        self._uv_version = version

    def set_uv_bin(self, uv_bin_fullpath):
        self._uv_bin = uv_bin_fullpath

    def get_uv_bin(self):
        return self._uv_bin or shutil.which("uv")

    def set_python_version(self, python_version):
        self._req_python_version = python_version

    def get_python_version(self):
        return self._req_python_version

    def get_uv_version(self):
        return self._uv_version

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

    def set_binary(self, binary_path):
        self._python = binary_path or self._python

    def get_binary(self):
        return self._python

    def set_venv_binary(self, binary_path):
        self._venv_python = binary_path or self._python

    def get_venv_binary(self):
        return self._venv_python

    def is_binary_updated(self):
        return self._python != sys.executable

    def run(self, *args, **kwargs):
        func = kwargs.pop("func", Argv.get_output)
        argv = self.get_run_argv(*args, **kwargs)

        self.log.debug("running: %s", argv)

        #synced = self._is_sync
        #self._is_sync = False
        ret = func(argv, **kwargs)
        #self._is_sync = synced
        return ret

    def get_run_argv(self, *args, **kwargs):
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
            extra_args = self.session.config.get(
                "agent.package_manager.uv_sync_extra_args", None
            )
            if extra_args:
                args = args + tuple(extra_args)
            self._is_sync = True

        # Set the cache dir to venvs dir is SYNCed otherwise use the pip download as cache
        if self._is_sync:
            cache_dir = self.session.config.get("agent.venvs_dir", None)
            if cache_dir is not None:
                kwargs["env"]["UV_CACHE_DIR"] = cache_dir
        else:
            cache_dir = self.session.config.get("agent.pip_download_cache.path", None)
            if cache_dir is not None:
                kwargs["env"]["UV_CACHE_DIR"] = cache_dir

        # if we need synced it then we cannot specify the python binary
        if not self._is_sync and self._venv_python:
            # if we have not synced then use the preinstalled venv python,
            # otherwise do not specify it
            args_i = next(i for i, a in enumerate(args+("-", )) if a.startswith("-") or a == "python")
            args = tuple(args[:args_i]) + ("--python", str(self._venv_python),) + tuple(args[args_i:])
        # elif "cwd" in kwargs:
        #     cwd = Path(kwargs["cwd"])/".venv"
        #     if cwd.exists():
        #         args_i = next(i for i, a in enumerate(args+("-", )) if a.startswith("-") or a == "python")
        #         args = tuple(args[:args_i]) + ("--python", str(cwd), ) + tuple(args[args_i:])

        # if check_if_command_exists("uv"):
        #     argv = Argv("uv", *args)
        # else:
        #     argv = Argv(self._python, "-m", "uv", *args)

        if self.USE_UV_BIN:
            argv = Argv(self.get_uv_bin(), *args, **kwargs)
        else:
            argv = Argv(self._python, "-m", "uv", *args, **kwargs)

        return argv

    @_guard_enabled
    def initialize(self, cwd=None,):
        if not self._initialized:
            if cwd:
                self._cwd = cwd

            self._initialized = True

    def get_api(self, session, python, requirements_manager, path, *args, **kwargs):
        if not self._api:

            self._api = UvAPI(
                lockfile_path=self._cwd, lock_config=self,
                session=session,
                python=python or self._python,
                requirements_manager=requirements_manager, path=path, *args, **kwargs)
        return self._api


class UvAPI(VirtualenvPip):
    config = attr.ib(type=UvConfig)

    INDICATOR_FILES = "pyproject.toml", "uv.lock"
    VENV_SUFFIX = "_uv"

    def __init__(self, lockfile_path, lock_config, session, python, requirements_manager,
                 path, interpreter=None, execution_info=None, **kwargs):
        self.lockfile_path = Path(lockfile_path) if lockfile_path else None
        self.lock_config = lock_config
        self._installed = False
        self._enabled = None
        self._created = False
        self._uv_install_path = None
        super(UvAPI, self).__init__(
            session, python, requirements_manager,
            path, interpreter=interpreter, execution_info=execution_info, **kwargs)

    def set_lockfile_path(self, lockfile_path):
        if lockfile_path:
            self.lockfile_path = Path(lockfile_path)

    def install(self, lockfile_path=None):
        # type: (str) -> bool
        self.set_lockfile_path(lockfile_path)

        if self.enabled:
            self.lock_config.run("sync", "--locked", cwd=str(self.lockfile_path), func=Argv.check_call)
            self._installed = True
            # self.lock_config.set_binary(Path(self.lockfile_path) / ".venv" / "bin" / "python")
            return True

        return False

    @property
    def is_installed(self):
        return self._installed

    @property
    def enabled(self):
        if self._enabled is None:
            self._enabled =  self.lockfile_path and self.lock_config.enabled and (
                any((self.lockfile_path / indicator).exists() for indicator in self.INDICATOR_FILES)
            )
        return self._enabled

    @property
    def lock_file_exists(self):
        return (self.lockfile_path and self.lock_config.enabled and
                (self.lockfile_path / self.INDICATOR_FILES[1]).exists())

    def freeze(self, freeze_full_environment=False):
        if not self.is_installed or not self.lockfile_path or not self.lock_config.enabled:
            # there is a bug so we have to call pip to get the freeze because UV will return the wrong list
            # packages = self.run_with_env(('freeze',), output=True).splitlines()
            packages = self.lock_config.get_run_argv(
                "pip", "freeze", cwd=self.lockfile_path).get_output().splitlines()
            # list clearml_agent as well
            # packages_without_program = [package for package in packages if PROGRAM_NAME not in package]
            return {'pip': packages}

        lines = self.lock_config.run(
            "pip", "freeze",
            cwd=str(self.lockfile_path or self._cwd or self.path)
        ).splitlines()
        # fix local filesystem reference in freeze
        from clearml_agent.external.requirements_parser.requirement import Requirement
        packages = [Requirement.parse(p) for p in lines]
        for p in packages:
            if p.local_file and p.editable:
                p.path = str(Path(p.path).relative_to(self.lockfile_path))
                p.line = "-e {}".format(p.path)

        return {
            "pip": [p.line for p in packages]
        }

    def get_python_command(self, extra=()):
        if self.lock_config and self.lockfile_path and self.is_installed:
            return self.lock_config.get_run_argv(
                "run", "--python", str(self.lockfile_path / ".venv" / "bin" / "python"), "python", *extra, cwd=self.lockfile_path)

        # if not self.lock_config.get_venv_binary() and check_if_command_exists("uv"):
        #     return Argv("uv", "run", "--no-project", "--python", self.lock_config.get_venv_binary(), "python", *extra)
        # else:
        #     if UvConfig.USE_UV_BIN:
        #         return Argv(shutil.which("uv"), "run", "--no-project", "--python", self.lock_config.get_venv_binary(), "python", *extra)
        #     else:
        #         return Argv(self.bin, "-m", "uv", "run", "--no-project", "--python", self.lock_config.get_venv_binary(), "python", *extra)
        #
        return Argv(self.lock_config.get_venv_binary(), *extra)

    def _make_command(self, command):
        return self.lock_config.get_run_argv("pip", *command)

    def _add_legacy_resolver_flag(self, pip_pkg_version):
        # no need for legacy flags
        pass

    def get_venv_manager(self):
        # Create a new instance of the parent class dynamically
        parent_class = self.__class__.__bases__[0]
        parent_instance = parent_class.__new__(parent_class)  # noqa
        parent_instance.__dict__ = copy(self.__dict__)
        return parent_instance

    def create(self):
        """
        Create virtualenv.
        Only valid if instantiated with path.
        Use self.python as self.bin does not exist.
        """
        if self._created:
            return

        # if found a lock file, we will create the entire environment when we can "install"
        if self.enabled:
            # create virtualenv for the UV package
            super(UvAPI, self).create()
            self.lock_config.set_binary(self.bin)
            return self

        # no lock file create a venv
        pip_venv = self.install_uv_package()
        self.lock_config.set_venv_binary(self._bin)
        self._bin = pip_venv.bin

        # Otherwise, we create a new venv here
        # if we want UV to create the venv we first need to install it, so we create a "temp" UV venv

        # get python version
        python_version = self.lock_config.get_python_version()
        if self.python and not python_version:
            python_version = self.python.split("/")[-1].lower().replace("python", "").replace(".exe", "")
            try:
                float(python_version)
            except:  # noqa
                python_version = None

        # noinspection PyBroadException
        try:
            # if no python version requested or it's the same as ours create a new venv from the currenbt one
            if not python_version or python_version_string() == python_version:
                if UvConfig.USE_UV_BIN:
                    command = Argv(self.lock_config.get_uv_bin(), "venv",
                                   "--python", sys.executable, *self.create_flags(), str(self.path))
                else:
                    command = pip_venv.get_python_command(
                        extra=("-m", "uv", "venv",
                               "--python", sys.executable, *self.create_flags(), str(self.path))
                    )
            else:
                # create and download the new python version
                if UvConfig.USE_UV_BIN:
                    command = Argv(self.lock_config.get_uv_bin(), "venv",
                                   "--python", python_version, *self.create_flags(), str(self.path))
                else:
                    command = pip_venv.get_python_command(
                        extra=("-m", "uv", "venv",
                               "--python", python_version, *self.create_flags(), str(self.path))
                    )

            print(python_version, python_version_string(), command)
            command.get_output()
        except Exception as ex:
            print("ERROR: UV venv creation failed: {}".format(ex))
            raise ex

        self._created = True

        return self

    def install_uv_package(self, uv_version=None):
        if not uv_version:
            if self.lock_config:
                uv_version = self.lock_config.get_uv_version()
            uv_version = uv_version or self.session.config.get("agent.package_manager.uv_version", None)

        # check the installed version
        existing_uv_version = None
        pip_venv = VirtualenvPip(
            session=self.session, python=self.python, requirements_manager=None,
            path=self.path, interpreter=self.lock_config.get_binary())
        packages = (pip_venv.freeze(freeze_full_environment=True) or dict()).get("pip")
        if packages:
            existing_uv_version = get_specific_package_version({"pip": packages}, package_name="uv")

        argv = None
        version = None
        need_install = True

        if uv_version is not None:
            version = str(uv_version)

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

            if existing_uv_version:
                from clearml_agent.helper.package.requirements import SimpleVersion
                need_install = not SimpleVersion.compare_versions(
                    existing_uv_version,
                    *SimpleVersion.split_op_version(version))

            if need_install:
                # (we are not running it yet)
                argv = (
                    "install",
                    "uv{}".format(version),
                    "--upgrade",
                )

            # this is just for beauty and checks, we already set the version in the Argv
            if not version:
                version = "latest"
        elif not existing_uv_version:
            # mark to install uv if not already installed (we are not running it yet)
            argv = (
                "install",
                "uv",
            )
            version = ""

        # check if we do not have a specific version and uv is found skip installation
        if not version and (existing_uv_version or check_if_command_exists("uv")):
            print(
                "Notice: `uv`{} was found, no specific version required, "
                "skipping uv installation".format(existing_uv_version or "")
            )
            UvConfig.USE_UV_BIN = True
        elif argv:
            if version:
                print("Installing / Upgrading `uv` package to {}".format(version))
            else:
                print("Installing `uv`")

            self._uv_install_path = str(self.path)[:-1] if str(self.path)[-1] == os.pathsep else str(self.path)
            self._uv_install_path += self.VENV_SUFFIX
            pip_venv = VirtualenvPip(
                session=self.session, python=self.python,
                requirements_manager=None, path=self._uv_install_path)
            pip_venv.create()

            # now install uv
            try:
                pip_venv.run_with_env(argv)
            except Exception as ex:
                self.lock_config.log.warning("failed installing uv: {}".format(ex))

            self.lock_config.set_binary(pip_venv.bin)
            if (Path(self._uv_install_path) / "bin" / "uv").exists():
                self.lock_config.set_uv_bin(Path(self._uv_install_path) / "bin" / "uv")
                UvConfig.USE_UV_BIN = True
        else:
            print(
                "Notice: `uv` {}was found, version required is {}, skipping uv installation".format(
                    existing_uv_version + " ", version)
            )

        return pip_venv

    def upgrade_pip(self, *args, **kwargs):
        pass

    def remove(self):
        """
        Delete virtualenv.
        Only valid if instantiated with path.
        """
        super(UvAPI, self).remove()
        uv_path = str(self.path)
        if uv_path and uv_path[-1] == os.pathsep:
            uv_path = uv_path[:-1]
        rm_tree(uv_path + self.VENV_SUFFIX)

