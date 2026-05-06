import os.path
from pathlib import Path


BOOTSTRAP_REPO_URL = "https://github.com/clearml/clearml-agent-bootstrap"


def check_bootstrap_package_install(config):
    if not config.get("agent.bootstrap.use_bootstrap", None):
        return False

    # noinspection PyBroadException
    try:
        from clearml_agent_bootstrap import install  # noqa
        return True
    except ImportError:
        return False
    except Exception:
        print("WARNING: importing clearml-agent-bootstrap failed")
    return False


def print_bootstrap_missing_notice():
    print("INFO: clearml-agent-bootstrap was not detected.")
    print("      We recommend installing it to significantly reduce agent job start-up time.")
    print("      To install, run: clearml-agent install-bootstrap")
    print("      See: {}".format(BOOTSTRAP_REPO_URL))


def check_latest_bootstrap_version(timeout: float = 5):
    """Compare installed bootstrap version with the latest GitHub release and print the result."""
    from clearml_agent.commands.install_bootstrap import (
        get_installed_bootstrap_version,
        get_latest_bootstrap_version,
    )
    current = get_installed_bootstrap_version()
    if not current:
        print("INFO: clearml-agent-bootstrap is not installed; skipping version check")
        return
    latest = get_latest_bootstrap_version(timeout=timeout)
    if not latest:
        print("INFO: could not determine latest clearml-agent-bootstrap version (network error or timeout)")
        return
    if current == latest:
        print("INFO: clearml-agent-bootstrap v{} is up-to-date".format(current))
    else:
        print("WARNING: clearml-agent-bootstrap v{} is installed, latest is v{}".format(current, latest))
        print("         To update, run: clearml-agent install-bootstrap")


def update_bootstrap(config, force=False):
    from clearml_agent_bootstrap import install  # noqa
    from clearml_agent_bootstrap.version import __version__ as bootstrap_version  # noqa
    target_folder = config.get("agent.bootstrap.host_bootstrap_dir", None) or "~/.clearml.bootstrap"
    target_folder = Path(target_folder).expanduser()
    target_folder.mkdir(parents=True, exist_ok=True)
    uv_python_meta_file=config.get("agent.bootstrap.uv_python_meta_file", None)
    if (uv_python_meta_file and
            (not uv_python_meta_file.startswith("http://") and not uv_python_meta_file.startswith("https://"))):
        uv_python_meta_file = Path(uv_python_meta_file).expanduser().as_posix()
    # override force
    if config.get("agent.bootstrap.always_update", False):
        force = True
    # notice this is not atomic, should not be called while someone is using this folder
    try:
        install(target_folder.as_posix(), python_binary_metadata_file=uv_python_meta_file, force=force)
        print("INFO: using agent bootstrap v{}: {}".format(bootstrap_version, target_folder))
    except Exception as ex:
        print("WARNING: Unpacking bootstrap package {} to {} failed: {}".format(bootstrap_version, target_folder, ex))
        return False

    return target_folder


def get_bootstrap_container_env_vars(config):
    # returns [{env_name: val}, ]
    # None values are removed
    bootstrap_dir = config.get("agent.docker_internal_mounts.bootstrap_folder", None)
    pip_version_string = config.get("agent.package_manager.pip_version", None)
    if pip_version_string:
        pip_version_string = "-U " + " ".join("\"pip {}\"".format(p) for p in pip_version_string)

    envs = dict(
        CLEARML_PIP_VER=pip_version_string or None,
        CLEARML_BOOTSTRAP_DIR=bootstrap_dir,
        CLEARML_BOOTSTRAP_CACHE_DIR=config.get("agent.docker_internal_mounts.bootstrap_cache", None)
            if config.get("agent.bootstrap.host_bootstrap_cache_dir", None) else None,
        CLEARML_FORCE_STATIC_GIT_BIN=config.get("agent.bootstrap.always_use_static_git", None),
        CLEARML_SKIP_GIT_INST=config.get("agent.bootstrap.skip_git_installation", None),
        CLEARML_SKIP_CA_CERT_INST=config.get("agent.bootstrap.always_install_ca_cert", None),
        SSL_CERT_FILE=config.get("agent.bootstrap.map_external_ssl_cert", None),
        UV_PYTHON_DOWNLOADS_JSON_URL=(bootstrap_dir.rstrip("/") + "/download-metadata.json")
            if config.get("agent.bootstrap.uv_python_meta_file", None) and bootstrap_dir else None,
        CLEARML_PYTHON_VER=config.get("agent.bootstrap.uv_def_python_ver", None),
        CLEARML_UPDATE_UV=config.get("agent.bootstrap.auto_update_uv_bin", None),
        UV_INSTALLER_GHE_BASE_URL=config.get("agent.bootstrap.auto_update_uv_url_gh", None),
        UV_INSTALLER_GITHUB_BASE_URL=config.get("agent.bootstrap.auto_update_uv_url_ghe", None),
        UV_NATIVE_TLS=config.get("agent.bootstrap.uv_use_builtin_tls", None),
        UV_INSECURE_HOST=config.get("agent.bootstrap.uv_insecure", None),
        GIT_SSL_NO_VERIFY=config.get("agent.bootstrap.git_insecure", None),
        UV_DEFAULT_INDEX=config.get("agent.bootstrap.uv_default_index_url", None),
        UV_INDEX=config.get("agent.bootstrap.uv_index_url", None),
        UV_INDEX_STRATEGY="unsafe-best-match" if config.get("agent.package_manager.uv_unsafe_best_match", False) else None,
    )
    return {k: ("1" if v else "0") if isinstance(v, bool) else str(v) for k, v in envs.items() if v is not None}


def get_bootstrap_container_mounts(config, envs):
    # returns [{host_directory: val}, ]

    # noinspection PyBroadException
    try:
        host_clearml_bootstrap_dir=Path(
            config.get("agent.bootstrap.host_bootstrap_dir", None) or "~/.clearml.bootstrap"
        ).expanduser().absolute().as_posix()
    except Exception:
        host_clearml_bootstrap_dir=None

    # noinspection PyBroadException
    try:
        host_clearml_bootstrap_cache_dir=Path(
            config.get("agent.bootstrap.host_bootstrap_cache_dir", None) or "~/.clearml.bootstrap.cache"
        ).expanduser().absolute().as_posix()
    except Exception:
        host_clearml_bootstrap_cache_dir=None

    mounts = {
        host_clearml_bootstrap_dir: (envs.get("CLEARML_BOOTSTRAP_DIR") + ":ro")
        if envs.get("CLEARML_BOOTSTRAP_DIR") else None,
        host_clearml_bootstrap_cache_dir: envs.get("CLEARML_BOOTSTRAP_CACHE_DIR", None),
    }
    return {k: v for k, v in mounts.items() if v is not None}


def get_bootstrap_docker_configuration(config):
    # return the full environment and mount point command line for the docker container launch command
    # list of argument, example ['-v', '/host/:/container/', '-e', 'ENV=VAL']
    envs = get_bootstrap_container_env_vars(config)
    mounts = get_bootstrap_container_mounts(config, envs)
    docker_args = []
    for k, v in mounts.items():
        docker_args += ['-v', '{}:{}'.format(k, v)]
    for k, v in envs.items():
        docker_args += ['-e', '{}={}'.format(k, v)]
    return docker_args
