from ...backend_config import Config
from ...backend_config.defs import ENV_CONFIG_VERBOSE
from ..._vendor.pathlib2 import Path


def load(*additional_module_paths):
    # type: (str) -> Config
    """
    Load configuration with the API defaults, using the additional module path provided
    :param additional_module_paths: Additional config paths for modules who'se default
    configurations should be loaded as well
    :return: Config object
    """
    config = Config(verbose=ENV_CONFIG_VERBOSE.get(default=False))
    this_module_path = Path(__file__).parent
    config.load_relative_to(this_module_path, *additional_module_paths)
    return config
