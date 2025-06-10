from __future__ import print_function

import functools
import json
import os
import sys
from os.path import expanduser
from typing import Any

from .._vendor import six
from .._vendor.pathlib2 import Path
from .._vendor.pyparsing import (
    ParseFatalException,
    ParseException,
    RecursiveGrammarException,
    ParseSyntaxException,
)

from clearml_agent.external import pyhocon
from clearml_agent.external.pyhocon import ConfigTree, ConfigFactory

from .defs import (
    Environment,
    DEFAULT_CONFIG_FOLDER,
    LOCAL_CONFIG_PATHS,
    ENV_CONFIG_PATHS,
    LOCAL_CONFIG_FILES,
    LOCAL_CONFIG_FILE_OVERRIDE_VAR,
    ENV_CONFIG_PATH_OVERRIDE_VAR,
)
from .defs import is_config_file
from .entry import Entry, NotSet
from .errors import ConfigurationError
from .log import initialize as initialize_log, logger
from .utils import get_options

try:
    from typing import Text
except ImportError:
    # windows conda-less hack
    Text = Any


log = logger(__file__)


class ConfigEntry(Entry):
    logger = None

    def __init__(self, config, *keys, **kwargs):
        # type: (Config, Text, Any) -> None
        super(ConfigEntry, self).__init__(*keys, **kwargs)
        self.config = config

    def _get(self, key):
        # type: (Text) -> Any
        return self.config.get(key, NotSet)

    def error(self, message):
        # type: (Text) -> None
        log.error(message.capitalize())


class Config(object):
    """
    Represents a server configuration.
    If watch=True, will watch configuration folders for changes and reload itself.
    NOTE: will not watch folders that were created after initialization.
    """

    # used in place of None in Config.get as default value because None is a valid value
    _MISSING = object()
    extra_config_values_env_key_sep = "__"
    extra_config_values_env_key_prefix = [
        "CLEARML_AGENT" + extra_config_values_env_key_sep,
    ]

    def __init__(
        self,
        config_folder=None,
        env=None,
        verbose=True,
        relative_to=None,
        app=None,
        is_server=False,
        **_
    ):
        self._app = app
        self._verbose = verbose
        self._folder_name = config_folder or DEFAULT_CONFIG_FOLDER
        self._roots = []
        self._config = ConfigTree()
        self._env = env or os.environ.get("TRAINS_ENV", Environment.default)
        self.config_paths = set()
        self.is_server = is_server
        self._overrides_configs = None

        if self._verbose:
            print("Config env:%s" % str(self._env))

        if not self._env:
            raise ValueError(
                "Missing environment in either init of environment variable"
            )
        if self._env not in get_options(Environment):
            raise ValueError("Invalid environment %s" % env)

        if relative_to is not None:
            self.load_relative_to(relative_to)

    @property
    def root(self):
        return self.roots[0] if self.roots else None

    @property
    def roots(self):
        return self._roots

    @roots.setter
    def roots(self, value):
        self._roots = value

    @property
    def env(self):
        return self._env

    def logger(self, path=None):
        return logger(path)

    def load_relative_to(self, *module_paths):
        def normalize(p):
            return Path(os.path.abspath(str(p))).with_name(self._folder_name)

        self.roots = list(map(normalize, module_paths))
        self.reload()

    def _reload(self):
        env = self._env
        config = self._config.copy()

        if self.is_server:
            env_config_paths = ENV_CONFIG_PATHS
        else:
            env_config_paths = []

        env_config_path_override = ENV_CONFIG_PATH_OVERRIDE_VAR.get()
        if env_config_path_override:
            env_config_paths = [expanduser(env_config_path_override)]

        # merge configuration from root and other environment config paths
        if self.roots or env_config_paths:
            config = functools.reduce(
                lambda cfg, path: ConfigTree.merge_configs(
                    cfg,
                    self._read_recursive_for_env(path, env, verbose=self._verbose),
                    copy_trees=True,
                ),
                self.roots + env_config_paths,
                config,
            )

        # merge configuration from local configuration paths
        if LOCAL_CONFIG_PATHS:
            config = functools.reduce(
                lambda cfg, path: ConfigTree.merge_configs(
                    cfg,
                    self._read_recursive(path, verbose=self._verbose),
                    copy_trees=True,
                ),
                LOCAL_CONFIG_PATHS,
                config,
            )

        local_config_files = LOCAL_CONFIG_FILES
        local_config_override = LOCAL_CONFIG_FILE_OVERRIDE_VAR.get()
        if local_config_override:
            print(f"Using custom configuration file from env var: {local_config_override}")
            local_config_files = [expanduser(local_config_override)]

        # merge configuration from local configuration files
        if local_config_files:
            config = functools.reduce(
                lambda cfg, file_path: ConfigTree.merge_configs(
                    cfg,
                    self._read_single_file(file_path, verbose=self._verbose),
                    copy_trees=True,
                ),
                local_config_files,
                config,
            )

        config = ConfigTree.merge_configs(
            config, self._read_extra_env_config_values(), copy_trees=True
        )

        config = self.resolve_override_configs(config)

        config["env"] = env
        return config

    def resolve_override_configs(self, initial=None):
        if not self._overrides_configs:
            return initial
        return functools.reduce(
            lambda cfg, override: ConfigTree.merge_configs(cfg, override, copy_trees=True),
            self._overrides_configs,
            initial or ConfigTree(),
        )

    def _read_extra_env_config_values(self) -> ConfigTree:
        """ Loads extra configuration from environment-injected values """
        result = ConfigTree()

        for prefix in self.extra_config_values_env_key_prefix:
            keys = sorted(k for k in os.environ if k.startswith(prefix))
            for key in keys:
                path = (
                    key[len(prefix) :]
                    .replace(self.extra_config_values_env_key_sep, ".")
                    .lower()
                )
                result = ConfigTree.merge_configs(
                    result, ConfigFactory.parse_string("{}: {}".format(path, os.environ[key]))
                )

        return result

    def replace(self, config):
        self._config = config

    def reload(self):
        self.replace(self._reload())

    def initialize_logging(self, debug=False):
        logging_config = self._config.get("logging", None)
        if not logging_config:
            return False

        # handle incomplete file handlers
        deleted = []
        handlers = logging_config.get("handlers", {})
        for name, handler in list(handlers.items()):
            cls = handler.get("class", None)
            is_file = cls and "FileHandler" in cls
            if cls is None or (is_file and "filename" not in handler):
                deleted.append(name)
                del handlers[name]
            elif is_file:
                file = Path(handler.get("filename"))
                if not file.is_file():
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()

        # remove dependency in deleted handlers
        root_logger = logging_config.get("root", None)
        loggers = list(logging_config.get("loggers", {}).values()) + (
            [root_logger] if root_logger else []
        )
        for logger in loggers:
            handlers = logger.get("handlers", None)
            if debug:
                logger['level'] = 'DEBUG'
            if not handlers:
                continue
            logger["handlers"] = [h for h in handlers if h not in deleted]

        extra = None
        if self._app:
            extra = {"app": self._app}
        initialize_log(logging_config, extra=extra)
        return True

    def __getitem__(self, key):
        try:
            return self._config[key]
        except:
            return None

    def __getattr__(self, key):
        c = self.__getattribute__('_config')
        if key.split('.')[0] in c:
            try:
                return c[key]
            except Exception:
                return None
        return getattr(c, key)

    def get(self, key, default=_MISSING):
        value = self._config.get(key, default)
        if value is self._MISSING and not default:
            raise KeyError(
                "Unable to find value for key '{}' and default value was not provided.".format(
                    key
                )
            )
        return value

    def put(self, key, value):
        self._config.put(key, value)

    def pop(self, key, default=None):
        return self._config.pop(key, default=default)

    def to_dict(self):
        return self._config.as_plain_ordered_dict()

    def as_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def _read_recursive_for_env(self, root_path_str, env, verbose=True):
        root_path = Path(root_path_str)
        if root_path.exists():
            default_config = self._read_recursive(
                root_path / Environment.default, verbose=verbose
            )
            if (root_path / env) != (root_path / Environment.default):
                env_config = self._read_recursive(
                    root_path / env, verbose=verbose
                )  # None is ok, will return empty config
                config = ConfigTree.merge_configs(default_config, env_config, True)
            else:
                config = default_config
        else:
            config = ConfigTree()

        return config

    def _read_recursive(self, conf_root, verbose=True):
        conf = ConfigTree()
        if not conf_root:
            return conf
        conf_root = Path(conf_root)

        if not conf_root.exists():
            if verbose:
                print("No config in %s" % str(conf_root))
            return conf

        if verbose:
            print("Loading config from %s" % str(conf_root))
        for root, dirs, files in os.walk(str(conf_root)):

            rel_dir = str(Path(root).relative_to(conf_root))
            if rel_dir == ".":
                rel_dir = ""
            prefix = rel_dir.replace("/", ".")

            for filename in files:
                if not is_config_file(filename):
                    continue

                if prefix != "":
                    key = prefix + "." + Path(filename).stem
                else:
                    key = Path(filename).stem

                file_path = str(Path(root) / filename)

                conf.put(key, self._read_single_file(file_path, verbose=verbose))

        return conf

    @staticmethod
    def _read_single_file(file_path, verbose=True):
        if not file_path or not Path(file_path).is_file():
            return ConfigTree()

        if verbose:
            print("Loading config from file %s" % file_path)

        try:
            return pyhocon.ConfigFactory.parse_file(file_path)
        except ParseSyntaxException as ex:
            msg = "Failed parsing {0} ({1.__class__.__name__}): (at char {1.loc}, line:{1.lineno}, col:{1.column})".format(
                file_path, ex
            )
            six.reraise(
                ConfigurationError,
                ConfigurationError(msg, file_path=file_path),
                sys.exc_info()[2],
            )
        except (ParseException, ParseFatalException, RecursiveGrammarException) as ex:
            msg = "Failed parsing {0} ({1.__class__.__name__}): {1}".format(
                file_path, ex
            )
            six.reraise(ConfigurationError, ConfigurationError(msg), sys.exc_info()[2])
        except Exception as ex:
            print("Failed loading %s: %s" % (file_path, ex))
            raise

    def set_overrides(self, *dicts):
        """ Set several override dictionaries or ConfigTree objects which should be merged onto the configuration """
        self._overrides_configs = [
            d if isinstance(d, ConfigTree) else pyhocon.ConfigFactory.from_dict(d) for d in dicts
        ]
        self.reload()
