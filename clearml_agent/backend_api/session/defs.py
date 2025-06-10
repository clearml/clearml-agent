from clearml_agent.helper.environment import EnvEntry
from clearml_agent.helper.environment.converters import safe_text_to_bool


ENV_HOST = EnvEntry("CLEARML_API_HOST", "TRAINS_API_HOST")
ENV_WEB_HOST = EnvEntry("CLEARML_WEB_HOST", "TRAINS_WEB_HOST")
ENV_FILES_HOST = EnvEntry("CLEARML_FILES_HOST", "TRAINS_FILES_HOST")
ENV_ACCESS_KEY = EnvEntry("CLEARML_API_ACCESS_KEY", "TRAINS_API_ACCESS_KEY")
ENV_SECRET_KEY = EnvEntry("CLEARML_API_SECRET_KEY", "TRAINS_API_SECRET_KEY")
ENV_AUTH_TOKEN = EnvEntry("CLEARML_AUTH_TOKEN")
ENV_VERBOSE = EnvEntry("CLEARML_API_VERBOSE", "TRAINS_API_VERBOSE", type=bool, default=False)
ENV_HOST_VERIFY_CERT = EnvEntry("CLEARML_API_HOST_VERIFY_CERT", "TRAINS_API_HOST_VERIFY_CERT", type=bool, default=True)
ENV_CONDA_ENV_PACKAGE = EnvEntry("CLEARML_CONDA_ENV_PACKAGE", "TRAINS_CONDA_ENV_PACKAGE")
ENV_USE_CONDA_BASE_ENV = EnvEntry("CLEARML_USE_CONDA_BASE_ENV", type=bool)
ENV_NO_DEFAULT_SERVER = EnvEntry("CLEARML_NO_DEFAULT_SERVER", "TRAINS_NO_DEFAULT_SERVER", type=bool, default=True)
ENV_DISABLE_VAULT_SUPPORT = EnvEntry('CLEARML_AGENT_DISABLE_VAULT_SUPPORT', type=bool)
ENV_ENABLE_ENV_CONFIG_SECTION = EnvEntry('CLEARML_AGENT_ENABLE_ENV_CONFIG_SECTION', type=bool)
ENV_ENABLE_FILES_CONFIG_SECTION = EnvEntry('CLEARML_AGENT_ENABLE_FILES_CONFIG_SECTION', type=bool)
ENV_VENV_CONFIGURED = EnvEntry('VIRTUAL_ENV', type=str)
ENV_PROPAGATE_EXITCODE = EnvEntry("CLEARML_AGENT_PROPAGATE_EXITCODE", type=bool, default=False)
ENV_INITIAL_CONNECT_RETRY_OVERRIDE = EnvEntry(
    'CLEARML_AGENT_INITIAL_CONNECT_RETRY_OVERRIDE', default=True, converter=safe_text_to_bool
)
ENV_FORCE_MAX_API_VERSION = EnvEntry("CLEARML_AGENT_FORCE_MAX_API_VERSION", type=str)
# values are 0/None (task per node), 1/2 (multi-node reporting, colored console), -1 (only report rank 0 node)
ENV_MULTI_NODE_SINGLE_TASK = EnvEntry("CLEARML_MULTI_NODE_SINGLE_TASK", type=int, default=None)


"""
Experimental option to set the request method for all API requests and auth login.
This could be useful when GET requests with payloads are blocked by a server as
POST requests can be used instead.

However this has not been vigorously tested and may have unintended consequences.
"""
ENV_API_DEFAULT_REQ_METHOD = EnvEntry("CLEARML_API_DEFAULT_REQ_METHOD", default="GET")
