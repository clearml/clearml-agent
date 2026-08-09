"""SNUG integration helpers.

``resolve_shim_path()`` locates the shipped shared library (a Linux ``.so``
loaded via ``LD_PRELOAD`` or a macOS ``.dylib`` loaded via
``DYLD_INSERT_LIBRARIES``); when none is shipped for the platform it returns
None, ``snug_enabled()`` is False, and the executioner's behavior is unchanged.
``injection_env_var()`` is the matching per-OS preload env-var name.

``build_shim_descriptor_fd()`` hands the in-process reporter (linked into the
shim) the backend credentials + task identity + sink config it needs, via an
inherited fd whose number the agent exports as ``CLEARML_SNUG_CRED_FD``. On
Linux that fd is an anonymous ``memfd`` (nothing touches disk); on macOS (no
``memfd_create``) it is an immediately-unlinked 0600 temp file under the
per-user ``$TMPDIR``. The shim reads it at its ctor and reports to the ClearML
backend itself, in-process.

``macos_dyld_injection_supported()`` probes whether macOS SIP / hardened
runtime would strip ``DYLD_INSERT_LIBRARIES`` from (or SIGKILL) a given
interpreter — it does for the system ``/usr/bin/python3`` — so the worker can
degrade gracefully (run the task WITHOUT SNUG) instead of silently no-op'ing or
crashing the task.

Naming convention: every constant/function on this module is the single
source of truth for its corresponding wire-level name (env var, runtime
property, file path). If you find yourself spelling one of these strings
literally elsewhere in the agent, import it from here instead.
"""
import base64
import json
import os
import platform
import subprocess
import tempfile
from typing import Optional
from urllib.parse import urlparse


# -- Call-history control: task User Properties + shim env -------------------
# The call-history capture mode (off|collect|dump|continuous) is switched LIVE
# from the ClearML UI via the task's User Properties (hyperparams section
# "properties") — the one task field editable while a task is running. The
# in-process reporter's poll thread reads these keys
# (clearml_snug/reporter/src/poll.rs). This module is the single source of truth
# for their names; the value strings must match the Rust CallHistoryMode serde
# lowercase strings (clearml_snug/messages/src/lib.rs).
SNUG_USERPROP_SECTION = "properties"
SNUG_USERPROP_CALL_HISTORY = "_snug_call_history"
# Runtime whitelist additions: a task may ADD hosts to meter via this User
# Property — a JSON rule array (e.g. [{"host":"api.foo.com","inject_headers":true}])
# or a comma/space/newline host-list shorthand. The reporter polls it
# (clearml_snug/reporter/src/poll.rs PROP_WHITELIST) and the shim merges the
# additions ON TOP of the immutable admin whitelist — admin hosts always win and
# default_action is never changed. An operator adds it in the UI (live) or before
# launch; when present at dispatch its value is applied from the first request via
# CLEARML_SNUG_WHITELIST_ADDITIONS.
SNUG_USERPROP_WHITELIST = "_snug_whitelist"

SNUG_CALL_HISTORY_OFF = "off"
SNUG_CALL_HISTORY_COLLECT = "collect"
SNUG_CALL_HISTORY_DUMP = "dump"
SNUG_CALL_HISTORY_CONTINUOUS = "continuous"
SNUG_CALL_HISTORY_MODES = (
    SNUG_CALL_HISTORY_OFF,
    SNUG_CALL_HISTORY_COLLECT,
    SNUG_CALL_HISTORY_DUMP,
    SNUG_CALL_HISTORY_CONTINUOUS,
)

# Launch-time predefine: the agent reads the task's _snug_whitelist property at
# dispatch and passes its raw value here so the shim applies the additions from
# the first request (before the reporter's first poll). Same format as the
# property; the shim merges it onto the immutable base in initialize().
ENV_NAME_SNUG_WHITELIST_ADDITIONS = "CLEARML_SNUG_WHITELIST_ADDITIONS"


# -- Wheel layout ------------------------------------------------------------
# Architecture mapping for the shipped shared library. Anything not in this map
# -> shim is unavailable on that machine.
_SUPPORTED_ARCHES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}

# platform.system() -> the shared-library extension for the shim on that OS.
# Linux uses LD_PRELOAD + .so; macOS uses DYLD_INSERT_LIBRARIES + .dylib.
# Anything not in this map (e.g. Windows) -> shim unavailable, SNUG is a no-op.
_SUPPORTED_OS_EXT = {
    "Linux": "so",
    "Darwin": "dylib",
}

# Path inside the installed clearml_agent package where the shim lives, one per
# (arch, ext). Both the Linux .so and the macOS .dylib for a given arch ship
# side-by-side in the SAME arch dir (the extension discriminates), so a single
# universal wheel serves both. Populated by CI when the wheel is built; absent
# in minimal distributions.
_SHIM_RELATIVE_PATH = "snug/lib/{arch}/libclearml_snug.{ext}"

# Path inside the installed clearml_agent package where the SNUG proxy binary
# lives, one per arch. Unlike the shim — a shared library whose .so/.dylib
# extension discriminates the OS — the proxy is a standalone native executable
# that always runs INSIDE the task's Linux execution container (it meters
# clients the LD_PRELOAD shim can't hook). So it ships Linux-only, per-arch,
# with no OS discriminator in the filename; resolve_proxy_path() gates on
# Linux accordingly. Populated by CI when the wheel is built; absent in minimal
# distributions.
_PROXY_RELATIVE_PATH = "snug/lib/{arch}/clearml_snug_proxy"


def _resolve_platform_tags(force_system=None, force_arch=None):
    # type: (Optional[str], Optional[str]) -> Optional[tuple]
    """Return ``(system, arch, ext)`` for the target platform, or ``None`` when
    the shim is not shippable there (unsupported OS or CPU arch).

    Defaults to the RUNNING platform. ``force_system`` (e.g. ``"Linux"``) and
    ``force_arch`` (canonical, e.g. ``"aarch64"``) override it — used by
    --docker mode to resolve the LINUX ``.so`` for the task container even when
    the agent host is macOS (the task container is always Linux)."""
    system = force_system or platform.system()
    ext = _SUPPORTED_OS_EXT.get(system)
    if ext is None:
        return None
    if force_arch is not None:
        arch = force_arch
    else:
        arch = _SUPPORTED_ARCHES.get(platform.machine().lower())
    if arch is None:
        return None
    return system, arch, ext


def resolve_shim_path(force_system=None, force_arch=None):
    # type: (Optional[str], Optional[str]) -> Optional[str]
    """Locate the preload shim shipped inside the installed wheel.

    On Linux this is the ``.so`` loaded via ``LD_PRELOAD``; on macOS the
    ``.dylib`` loaded via ``DYLD_INSERT_LIBRARIES`` (see ``injection_env_var()``).

    When ``CLEARML_SNUG_SHIM_PATH`` is set in the environment and points at an
    existing file, that path takes precedence over the in-wheel lookup. This is
    what the outer agent (running in --docker mode) uses to point the
    in-container agent at a mounted .so without needing the wheel installed
    inside the container to bundle one.

    ``force_system`` / ``force_arch`` override the running platform — used by
    --docker mode to resolve the LINUX ``.so`` for the task container even when
    the agent host is macOS (the container is always Linux). Defaults to the
    running platform, so every other caller is unaffected.

    Returns the absolute path when a shim is shipped for this OS/arch and
    reachable; ``None`` otherwise (e.g. Windows, an unsupported arch, or a wheel
    built without the binary).
    """
    # Docker-mode override: the plumbing mounts the host's .so at a known
    # container path and sets this env var to point at it. The override is
    # platform-independent so the in-container agent (which is always Linux when
    # --docker is used) can find it even if the arch/ext detection below would
    # otherwise fall through.
    override = os.environ.get("CLEARML_SNUG_SHIM_PATH", "").strip()
    if override and os.path.isfile(override):
        return override

    tags = _resolve_platform_tags(force_system, force_arch)
    if tags is None:
        return None
    _system, arch, ext = tags
    try:
        # resource_filename resolves the bundled shim to a real on-disk path
        # (required for the dynamic linker) even if clearml_agent were installed
        # zipped.
        from pkg_resources import resource_filename
        path = resource_filename(
            "clearml_agent",
            _SHIM_RELATIVE_PATH.format(arch=arch, ext=ext),
        )
    except Exception:
        return None
    if not os.path.isfile(path):
        return None
    return path


def resolve_proxy_path(force_system=None, force_arch=None):
    # type: (Optional[str], Optional[str]) -> Optional[str]
    """Locate the SNUG proxy binary shipped inside the installed wheel.

    The proxy is a standalone forward-proxy ELF that meters clients the preload
    shim can't hook (statically-linked BoringSSL). It always executes inside the
    task's Linux container, so only a Linux build is shipped — this mirrors
    ``resolve_shim_path()``'s arch/platform resolution (and its
    ``force_system`` / ``force_arch`` overrides, used by --docker mode from a
    macOS host) but returns ``None`` for any non-Linux target rather than
    handing back a Linux ELF a macOS caller couldn't execute.

    When ``CLEARML_SNUG_PROXY_PATH`` is set and points at an existing file, that
    path wins over the in-wheel lookup — the outer agent (--docker mode) uses it
    to point the in-container agent at a mounted binary without the wheel
    needing to bundle one inside the container.

    Returns the absolute path when a proxy is shipped for this arch and
    reachable; ``None`` otherwise (non-Linux, an unsupported arch, or a wheel
    built without the binary).
    """
    override = os.environ.get("CLEARML_SNUG_PROXY_PATH", "").strip()
    if override and os.path.isfile(override):
        return override

    tags = _resolve_platform_tags(force_system, force_arch)
    if tags is None:
        return None
    system, arch, _ext = tags
    # Linux-only: no macOS/other proxy build is shipped. A macOS host that needs
    # the proxy for the (always-Linux) task container passes force_system="Linux".
    if system != "Linux":
        return None
    try:
        # resource_filename resolves the bundled binary to a real on-disk path
        # even if clearml_agent were installed zipped.
        from pkg_resources import resource_filename
        path = resource_filename(
            "clearml_agent",
            _PROXY_RELATIVE_PATH.format(arch=arch),
        )
    except Exception:
        return None
    if not os.path.isfile(path):
        return None
    return path


def proxy_binary_available(force_system=None, force_arch=None):
    # type: (Optional[str], Optional[str]) -> bool
    """True iff the SNUG proxy binary is shipped and reachable for the target
    platform (defaults to the running platform; ``force_system="Linux"`` probes
    the task-container binary from a macOS host). Thin wrapper over
    ``resolve_proxy_path()``."""
    return resolve_proxy_path(force_system, force_arch) is not None


def injection_env_var():
    # type: () -> str
    """The dynamic-linker preload env-var name for this OS: ``LD_PRELOAD`` on
    Linux, ``DYLD_INSERT_LIBRARIES`` on macOS. Defaults to ``LD_PRELOAD`` on any
    other platform (where ``resolve_shim_path()`` is ``None`` anyway, so the
    value is never consulted)."""
    # Local import to avoid a circular dep at module-load time.
    from clearml_agent.helper.base import is_macos_platform
    return "DYLD_INSERT_LIBRARIES" if is_macos_platform() else "LD_PRELOAD"


def snug_enabled(session):
    """True iff the SNUG should be loaded for tasks on this worker.

    All three must hold:
      1. ``agent.snug.enabled`` is true in the agent config (operator switch).
      2. We're on a platform we ship a shim for (Linux or macOS); the preload
         mechanism (LD_PRELOAD / DYLD_INSERT_LIBRARIES) is unavailable elsewhere.
      3. ``resolve_shim_path()`` finds a usable shim on this platform/arch.

    Returns False when no shim is shipped for this platform. This is by design:
    flipping ``agent.snug.enabled=true`` on a wheel without a bundled shim is a
    no-op, not an error.

    NOTE: this is the worker-level gate. On macOS, whether the shim can actually
    be injected into a SPECIFIC task interpreter is a separate, per-interpreter
    question (SIP / hardened runtime) answered by
    ``macos_dyld_injection_supported()`` at launch time.
    """
    try:
        config_enabled = bool(session.config.get("agent.snug.enabled", False))
    except Exception:
        return False
    if not config_enabled:
        return False
    # Local import to avoid a circular dep at module-load time.
    try:
        from clearml_agent.helper.base import is_linux_platform, is_macos_platform
        if not (is_linux_platform() or is_macos_platform()):
            return False
    except Exception:
        return False
    return resolve_shim_path() is not None


# Cache: resolved-interpreter-path -> bool. The agent is long-lived and an
# interpreter's SIP/hardened status doesn't change at runtime, so probe once.
_DYLD_SUPPORT_CACHE = {}


def macos_dyld_injection_supported(interpreter_path):
    # type: (str) -> bool
    """True iff our shim dylib can ACTUALLY be injected into ``interpreter_path``.

    On macOS two independent protections block injection, and a correct check
    must catch BOTH — probing with a *signed* sentinel gives a dangerous false
    positive for case 2:
      1. **SIP / DYLD stripping** — the system ``/usr/bin/python3`` and other
         SIP/hardened binaries lacking the
         ``com.apple.security.cs.allow-dyld-environment-variables`` entitlement
         have ``DYLD_*`` stripped, so the dylib never loads (silent no-op).
      2. **Hardened-runtime library validation** — even when ``DYLD_*`` survives,
         loading our (non-team-signed) dylib into such a binary makes the OS
         **SIGKILL** the process at load (the task would die with signal 9).

    So we probe by actually loading the real shim: spawn ``interpreter_path``
    with ``DYLD_INSERT_LIBRARIES=<shim>`` + a trivial script, and require the
    child to (a) exit cleanly AND (b) print the shim's ``[snug] init`` ctor line.
    Stripped DYLD -> no init line; a validation kill -> nonzero exit. Either ->
    not injectable -> caller degrades gracefully (runs the task WITHOUT SNUG)
    instead of crashing it.

    Homebrew / pyenv / conda interpreters are injectable; the system python is
    not. Always True on non-macOS (LD_PRELOAD is used). Cached per resolved
    interpreter path; any error/timeout -> False.
    """
    from clearml_agent.helper.base import is_macos_platform
    if not is_macos_platform():
        return True
    shim = resolve_shim_path()
    if not shim:
        return False
    key = os.path.realpath(interpreter_path) if interpreter_path else ""
    if key in _DYLD_SUPPORT_CACHE:
        return _DYLD_SUPPORT_CACHE[key]
    try:
        probe_env = dict(os.environ)
        # Load the REAL (unsigned) shim — this is what exposes a
        # library-validation kill, which a signed sentinel would not.
        probe_env["DYLD_INSERT_LIBRARIES"] = shim
        # No CLEARML_SNUG_CRED_FD -> the ctor logs "[snug] init ... reporter=stderr"
        # and starts no reporter thread; it just proves the dylib loaded + ran.
        res = subprocess.run(
            [interpreter_path, "-c", "pass"],
            env=probe_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        # A completed probe is a DEFINITIVE verdict — injectable iff the child
        # exited cleanly AND printed the shim's ctor line. Cache it (an
        # interpreter's SIP/hardened status doesn't change at runtime).
        supported = res.returncode == 0 and b"[snug] init" in (res.stderr or b"")
        _DYLD_SUPPORT_CACHE[key] = supported
        return supported
    except Exception:
        # A transient failure (subprocess.TimeoutExpired on a momentarily-loaded
        # host, a fork OSError, etc.) is NOT a real injectability verdict — do
        # NOT cache it, or one transient blip would permanently disable SNUG for
        # this interpreter for the rest of a long-lived agent's life. Degrade for
        # THIS launch only; the next task re-probes.
        return False


# -- In-process reporter handoff ---------------------------------------------
# The shim (loaded via LD_PRELOAD) links the reporter library and reports to the
# ClearML backend itself. The agent hands it everything it needs via an anonymous
# memfd: nothing touches disk, the secret never enters the process environment
# block, and the fd is inheritable so it survives execv. The shim reads the fd
# named by CLEARML_SNUG_CRED_FD at its ctor, parses the descriptor, and closes it.


def _hostname(url_or_host):
    # type: (object) -> Optional[str]
    """Extract a bare lowercase hostname from a URL or ``host[:port]`` string.

    ``urlparse`` needs a scheme (or a leading ``//``) to populate ``.hostname``;
    a bare ``api.clear.ml`` otherwise lands in ``.path``. ``.hostname`` already
    lowercases and strips the port + IPv6 brackets. Returns ``None`` for falsy
    or unparseable input.
    """
    if not url_or_host:
        return None
    s = str(url_or_host).strip()
    if not s:
        return None
    if "://" not in s:
        s = "//" + s
    try:
        host = urlparse(s).hostname
    except Exception:
        return None
    return host or None


def _self_hosts(session):
    # type: (object) -> list
    """Hostnames of the ClearML backend this task reports to — the api, files,
    and web (app) servers — so the shim can exclude the task's own ClearML SDK
    traffic from LLM metering/usage.

    The task's ``clearml`` SDK talks to these over urllib3/OpenSSL, which the
    shim hooks; without this list those calls would be metered (and billed when
    the host is whitelisted) as if they were model traffic. The reporter's own
    backend traffic is already invisible (it uses rustls, not the hooked
    OpenSSL); this closes the gap for the task's SDK calls.

    Best-effort: a server we can't resolve is simply omitted (``get_*_host`` can
    raise when it can't compose a URL). Returns deduped, lowercased,
    port-stripped hostnames; the api server (``session.host``) is always first.
    """
    urls = []
    try:
        urls.append(session.host)  # api_server
    except Exception:
        pass
    cfg = getattr(session, "config", None)
    # files / web (app) servers resolve from the same config (classmethods on
    # the Session, callable via the instance). Each is independent + best-effort.
    for attr in ("get_files_server_host", "get_app_server_host"):
        resolver = getattr(session, attr, None)
        if cfg is None or not callable(resolver):
            continue
        try:
            urls.append(resolver(cfg))
        except Exception:
            pass

    hosts = []
    seen = set()
    for u in urls:
        h = _hostname(u)
        if h and h not in seen:
            seen.add(h)
            hosts.append(h)
    return hosts


def _build_descriptor_dict(session, task_id, worker_id="", user="", project=""):
    # type: (object, str, str, str, str) -> dict
    """Assemble the handoff descriptor: backend creds + task identity + sink
    config, so the in-process reporter reaches the ClearML backend without a live
    Python ``Session``.

    Credentials are token-primary: in most deployments only a token is available
    (no access/secret). We pass whatever exists — the current token (for
    immediate use + in-process Bearer-renewal) plus access/secret when present
    (for robust re-login). Field-list precedence for task metrics: the
    ``CLEARML_AGENT_SNUG_TASK_METRICS_FIELDS`` env (comma list) overrides the
    configured list; the reporter validates names and falls back to all when the
    list is empty.
    """
    cfg = session.config
    verify = cfg.get("api.verify_certificate", True)
    ca_cert_path = None
    # ClearML allows api.verify_certificate to be a CA-bundle path string.
    if isinstance(verify, str):
        ca_cert_path = verify
        verify = True

    fields_env = os.environ.get("CLEARML_AGENT_SNUG_TASK_METRICS_FIELDS", "").strip()
    if fields_env:
        task_metrics_fields = [p.strip() for p in fields_env.split(",") if p.strip()]
    else:
        cfg_fields = cfg.get("agent.snug.task_metrics_fields", None)
        task_metrics_fields = list(cfg_fields) if cfg_fields is not None else []

    # Best-effort current bearer token so the reporter can use it immediately and
    # Bearer-renew it (the token-only path). None when the session doesn't expose
    # one, in which case the reporter logs in with access/secret.
    try:
        auth_token = getattr(session, "token", None) or None
    except Exception:
        auth_token = None

    # Usage attribution user for report_llm_usage: prefer the explicitly-passed
    # value (the worker passes the task owner, current_task.user), and fall back
    # to the session's own authenticated user id so a launcher that omits `user`
    # still attributes usage to a real user instead of leaving it "Unattributed".
    # `Session.user_id` is decoded from the auth token's identity; empty/None when
    # unavailable, in which case the backend derives the user from the task.
    try:
        session_user = getattr(session, "user_id", "") or ""
    except Exception:
        session_user = ""
    resolved_user = user or session_user or ""

    return {
        "api_server": session.host,
        "access_key": session.access_key or "",
        "secret_key": session.secret_key or "",
        "auth_token": auth_token,
        "verify_certificate": bool(verify),
        "ca_cert_path": ca_cert_path,
        "task_id": task_id,
        "worker_id": worker_id or "",
        "user": resolved_user,
        "project": project or "",
        "poll_interval_sec": float(cfg.get("agent.snug.poll_interval_sec", 10) or 10),
        "report_usage_events": bool(cfg.get("agent.snug.report_usage_events", True)),
        "report_task_metrics": bool(cfg.get("agent.snug.report_task_metrics", True)),
        "task_metrics_fields": task_metrics_fields,
        "aggregator_url": (cfg.get("agent.snug.aggregator_url", None) or None),
        "self_hosts": _self_hosts(session),
    }


def _memfd_from_bytes(data):
    # type: (bytes) -> int
    """Linux: write ``data`` into an anonymous ``memfd`` and return the
    (inheritable) fd. Nothing touches disk."""
    # flags=0 => no MFD_CLOEXEC => the fd survives execv. set_inheritable is
    # belt-and-suspenders (and is what the non-execv subprocess pass_fds path
    # relies on). memfd_create is Linux 3.17+ / Python 3.8+.
    fd = os.memfd_create("clearml-snug-cred", 0)
    try:
        os.write(fd, data)
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return fd


def _tmpfd_from_bytes(data):
    # type: (bytes) -> int
    """macOS (no ``memfd_create``): write ``data`` into an immediately-unlinked
    0600 temp file and return the (inheritable) fd.

    The path is removed right after creation so only the inherited fd references
    the inode (POSIX unlinked-but-open file); the on-disk bytes are unreachable
    by name and the inode is reclaimed once every fd (the agent's + the task's
    inherited copy) is closed. There is a sub-millisecond window where the 0600
    file exists by name under the per-user mode-0700 ``$TMPDIR``
    (``/var/folders/...`` on macOS) before the unlink. Same
    write->lseek0->set_inheritable discipline as the Linux memfd path; note
    ``mkstemp`` returns an O_CLOEXEC fd, so the ``set_inheritable(True)`` call is
    what makes it survive both execv and the pass_fds subprocess path.
    """
    fd, path = tempfile.mkstemp(prefix="clearml-snug-cred-")
    try:
        os.unlink(path)
        os.write(fd, data)
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return fd


def build_shim_descriptor_fd(session, task_id, worker_id="", user="", project=""):
    # type: (object, str, str, str, str) -> int
    """Build the handoff descriptor and write it into an inheritable fd; return
    the fd number.

    The agent exports the fd number as ``CLEARML_SNUG_CRED_FD``; the shim reads
    it at its ctor, parses the descriptor, and closes it. The fd is an anonymous
    ``memfd`` on Linux (nothing touches disk) and an immediately-unlinked 0600
    temp file on macOS (no ``memfd_create``). We branch on the actual capability
    (``hasattr(os, "memfd_create")``) rather than the OS string so the temp-file
    path is also reachable on a kernel without memfd. The secret never enters
    the process environment block on either path.
    """
    descriptor = _build_descriptor_dict(session, task_id, worker_id, user=user, project=project)
    data = json.dumps(descriptor).encode("utf-8")
    if hasattr(os, "memfd_create"):
        return _memfd_from_bytes(data)
    return _tmpfd_from_bytes(data)


def build_shim_descriptor_b64(session, task_id, worker_id="", user="", project=""):
    # type: (object, str, str, str, str) -> str
    """Build the handoff descriptor and return it base64-encoded, for delivery
    via the ``CLEARML_SNUG_CRED`` env var instead of an inheritable fd.

    An env var survives child spawning that scrubs the inherited cred fd
    (Chromium's process launcher and ``bwrap`` both drop non-allowlisted fds but
    keep the environment), so this is the reporting-credential channel for
    sandboxed hosts — Claude Desktop's Electron network-service and Cowork bwrap
    children, where ``CLEARML_SNUG_CRED_FD`` doesn't survive. Trade-off vs the
    fd: the secret then lives in the process environment block, so callers set it
    only for the sandboxed-app launch path, not for ordinary tasks.
    """
    descriptor = _build_descriptor_dict(session, task_id, worker_id, user=user, project=project)
    data = json.dumps(descriptor).encode("utf-8")
    return base64.b64encode(data).decode("ascii")


# -- Runtime control read (task User Properties) -----------------------------
# The agent reads the task's User Properties (hyperparams section "properties")
# — the same channel the in-process reporter polls — to predefine the shim's
# call-history mode and whitelist additions at dispatch (see
# worker._get_job_os_envs). DISTINCT from the worker runtime properties
# (workers.{get,set}_runtime_properties); the read uses tasks.get_hyper_params.


def get_task_user_property(session, task_id, name):
    # type: (object, str, str) -> Optional[str]
    """Read one task User Property value (hyperparams section "properties"), or
    None if absent/unreadable. Best-effort, never raises. Mirrors the reporter's
    read; uses the agent's existing ``get_hyper_params`` action (see
    ``_get_task_os_env``)."""
    # noinspection PyBroadException
    try:
        resp = session.get(service="tasks", action="get_hyper_params", tasks=[task_id])
        for p in resp['params'][0]['hyperparams']:
            if p.get('section') == SNUG_USERPROP_SECTION and p.get('name') == name:
                return str(p.get('value', ''))
    except Exception:
        return None
    return None
