"""Desktop-app metering glue (forward proxy + launcher/SDK wrappers).

This module is the ISOLATION boundary the SNUG design wants: the generic
metering mechanism (the LD_PRELOAD/DYLD shim, the bundled forward proxy,
the in-process reporter, the credential descriptor) lives in
``clearml_agent.helper.snug`` and is shared by every task. Everything here is the
app-launch plumbing for desktop AI apps whose LLM runtimes STATICALLY link
BoringSSL and so cannot be hooked by the LD_PRELOAD/DYLD shim. It is wired into
``worker.py`` behind a strict opt-in gate (``resolve_app_profile``) so that when
no app profile is selected the worker's behavior is byte-identical to a build
without this module.

The app-specific facts (which launcher(s) to shadow, which SDK binary to wrap and
how to find it, whether an external OAuth browser needs the CA, the default
tokenizer, the per-app whitelist rows) live in an :class:`AppProfile` declared in
``BUILTIN_PROFILES`` and keyed by ``agent.snug.app_mode``. Claude Desktop is the
first such profile; other desktop coding UIs are added as further profiles rather
than by editing the mechanism below.

WHY a proxy at all (and not just the shim): apps like Claude Desktop / Cowork run
the ``code`` path through a **bun**-spawned SDK binary that statically-links
BoringSSL, and the Chromium renderer that runs the ``chat`` path does too. The
LD_PRELOAD shim can only hook processes that dynamically link libssl/libcrypto,
so it can meter neither. The bundled forward proxy is TLS-stack-agnostic and
meters both by sitting in front of them: the SDK binary is pointed at the proxy
via ``HTTPS_PROXY`` and made to trust the proxy's CA via ``NODE_EXTRA_CA_CERTS``;
the Chromium renderer via launch switches (below).

WHY a per-SDK wrapper script (and not a task-wide env var): the ``HTTPS_PROXY``
+ CA-trust env MUST reach ONLY the app's SDK binary — putting it in the task-wide
environment (``_get_job_os_envs``) would route the agent's own ClearML SDK
traffic and every other child through the proxy too. So we rename the SDK ELF to
``<name>.real`` and drop a ``#!/bin/sh`` wrapper in its place that exports the
proxy env only for that exec. A watcher subprocess re-installs the wrapper for
SDKs the app re-downloads on demand.

WHY also wrap the Electron launcher: the Chromium renderer that runs an app's
``chat`` path ALSO statically-links BoringSSL, so the shim can't hook it either.
It can't be pointed at the proxy by env alone (Chromium routes with
``HTTP(S)_PROXY`` but then rejects the proxy's cert), and every JS-injection vector
(CDP / NODE_OPTIONS / asar) is fuse-blocked on hardened builds. What works is
Chromium LAUNCH SWITCHES: we shadow the launcher the same way we shadow the SDK —
rename it to ``.real`` and drop a wrapper that execs it with ``--proxy-server`` /
``--proxy-bypass-list`` / ``--ignore-certificate-errors-spki-list=<CA SPKI>``
appended. No watcher is needed there (the launcher is baked into the image, not
re-downloaded); the wrap is one-shot at setup and restored at teardown.

The functions here are deliberately PURE / stdlib-only (os + shutil +
subprocess) so they unit-test without a live agent, and so the UI devloop can
import + exercise them directly.
"""
import glob
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from collections import namedtuple
from typing import List, Optional

from clearml_agent.helper.snug import build_shim_descriptor_b64
from clearml_agent.snug.whitelist import build_whitelist_env


# ---------------------------------------------------------------------------
# App profile — the data that makes the mechanism below app-agnostic.
# ---------------------------------------------------------------------------

# A single SDK/CLI binary the app spawns for its "code" path and that we shadow
# with a proxy wrapper.
#   binary_name           : the on-disk basename to shadow (e.g. "claude").
#   discovery             : how to find its dir(s):
#                             "home_glob"     — walk home roots for the app's SDK
#                                               layout (the re-downloaded-SDK case).
#                             "explicit_path" — the binary is baked at a fixed
#                                               path in the image; check the dirs
#                                               named in `explicit_dirs`.
#                             "path_lookup"   — reserved; not implemented (raise).
#   container_substr      : home_glob only — the SDK dir's grandparent must
#                           contain this substring (e.g. "claude-code").
#   expects_version_parent: home_glob only — require an intermediate version dir.
#   explicit_dirs         : explicit_path only — absolute dirs to shadow the
#                           binary in. Wrap the dir the app INVOKES (a PATH entry,
#                           symlink and all), NOT a symlink's target dir: the
#                           wrapper finds its `.real`/CA siblings via
#                           $(dirname "$0"), which is the invoked path's dir.
#   watched               : re-install the wrapper on a poll — needed both when
#                           the app re-downloads the SDK and because setup wraps
#                           SDKs only via the watcher (there is no one-shot wrap).
#   wrapper_kind          : which wrapper env recipe to inject. Only "node_bun"
#                           (NODE_EXTRA_CA_CERTS + HTTPS_PROXY) is implemented.
SdkBinary = namedtuple(
    "SdkBinary",
    ["binary_name", "discovery", "container_substr",
     "expects_version_parent", "watched", "wrapper_kind", "explicit_dirs"],
)
SdkBinary.__new__.__defaults__ = ("", False, False, "node_bun", ())

# An Electron/Chromium launcher we shadow to route the renderer through the proxy.
#   path : absolute path of the launcher to shadow.
#   kind : injection recipe. Only "electron_chromium" (the three Chromium
#          switches + proxy/CA env) is implemented.
Launcher = namedtuple("Launcher", ["path", "kind"])
Launcher.__new__.__defaults__ = ("electron_chromium",)

# A desktop AI app whose statically-linked BoringSSL traffic is metered via the
# proxy. See BUILTIN_PROFILES for the concrete instances.
#   app_id                : the gate value (agent.snug.app_mode) and the log /
#                           launcher-marker discriminator.
#   launchers             : Electron launchers to shadow (may be empty).
#   sdk_binaries          : SDK/CLI binaries to wrap (may be empty).
#   default_tokenizer     : the proxy's fallback tokenizer for un-ruled hosts.
#   external_oauth_browser: whether the app opens a SYSTEM browser for OAuth, so
#                           the CA must be installed into the user's NSS stores.
#   decrypt_all           : proxy decrypt-all policy (decrypt every host, not just
#                           known providers).
#   h2_assumed_host       : per-app fallback host for h2 streams whose HPACK
#                           :authority is not decoded (exported to shim-hooked
#                           children as CLEARML_SNUG_H2_ASSUMED_HOST). None leaves
#                           the shim's own default in place.
#   whitelist_contribution: extra whitelist rules the app's hosts need, merged
#                           into the effective whitelist as additions (admin rules
#                           still win on a host collision).
AppProfile = namedtuple(
    "AppProfile",
    ["app_id", "launchers", "sdk_binaries", "default_tokenizer",
     "external_oauth_browser", "decrypt_all", "h2_assumed_host",
     "whitelist_contribution"],
)
AppProfile.__new__.__defaults__ = (True, None, ())


BUILTIN_PROFILES = {
    # Claude Desktop / Cowork: an Electron launcher (chat → claude.ai over h2) and
    # a bun-spawned ``claude`` SDK re-downloaded under the desktop user's home
    # (code → api.anthropic.com over h1). Values reproduce the historical
    # hardcoded behavior exactly (see the golden-value test).
    "claude_desktop": AppProfile(
        app_id="claude_desktop",
        launchers=(Launcher("/usr/bin/claude-desktop-unofficial", "electron_chromium"),),
        sdk_binaries=(
            SdkBinary(
                binary_name="claude",
                discovery="home_glob",
                container_substr="claude-code",
                expects_version_parent=True,
                watched=True,
                wrapper_kind="node_bun",
            ),
        ),
        default_tokenizer="claude",
        external_oauth_browser=True,
        decrypt_all=True,
        # Claude Desktop's shim-hooked leg (the dynamically-linked Cowork node
        # service) speaks h2 to api.anthropic.com; this seeds the per-stream host
        # the shim can't recover from HPACK :authority.
        h2_assumed_host="api.anthropic.com",
        whitelist_contribution=(
            {
                "host": "claude.ai",
                "path_prefix": "/",
                "inject_headers": False,
                "tokenizer": "claude",
                # The consumer chat wire carries no `usage`, so a completion is
                # byte-estimated and told from history-loads by its request line.
                # (The proxy reads these fields once it drives the estimate gate
                # from the whitelist; older binaries ignore unknown fields and use
                # their built-in claude.ai predicates.)
                "estimate_unmeasured": True,
                "completion_path": "*/completion",
                "provider": "anthropic",
            },
        ),
    ),
    # OpenCode: a single bun-compiled standalone CLI. Bun statically links
    # BoringSSL, so the LD_PRELOAD shim can't hook it (same reason Claude
    # Desktop's SDK needs the proxy). No Electron/Chromium leg — the LLM path IS
    # the CLI. The image bakes the binary at /root/.opencode/bin/opencode and
    # every session launches it as `opencode`, resolved through the
    # /usr/local/bin/opencode symlink on PATH; we shadow it in that PATH dir
    # (where $(dirname "$0") lands), so its provider HTTPS routes through the
    # proxy. decrypt_all is off: only the three known providers
    # (api.anthropic.com / api.openai.com / generativelanguage.googleapis.com)
    # are decrypted + metered; OpenCode's other egress (git, package installs,
    # MCP) is blind-tunnelled untouched. external_oauth_browser is off — metering
    # a user-supplied API key needs no system-browser CA trust.
    "opencode": AppProfile(
        app_id="opencode",
        launchers=(),
        sdk_binaries=(
            SdkBinary(
                binary_name="opencode",
                discovery="explicit_path",
                explicit_dirs=("/usr/local/bin",),
                watched=True,
                wrapper_kind="node_bun",
            ),
        ),
        default_tokenizer="approx",
        external_oauth_browser=False,
        decrypt_all=False,
        h2_assumed_host=None,
        whitelist_contribution=(),
    ),
}


# The real SDK / launcher entrypoint after we shadow it with a wrapper.
_REAL_SUFFIX = ".real"
# The CA file dropped beside a wrapper (see install_sdk_wrapper for why beside).
_CA_FILENAME = "snug_ca.pem"
# First 4 bytes of an ELF binary — how we tell "still the real SDK" from
# "already our #!/bin/sh wrapper".
_ELF_MAGIC = b"\x7fELF"

# Marker the app (its claude-code downloader) writes beside the SDK binary once
# it has fully downloaded + verified it. When present the install is complete and
# it is safe to wrap; we also leave it untouched so the app still trusts the
# (now-wrapped) binary at launch.
_SDK_VERIFIED_MARKER = ".verified"

# Filename the proxy writes the CA's SPKI-SHA256 (base64) to, beside the CA cert.
_CA_SPKI_FILENAME = "snug_proxy_ca.spki"

# Nickname the proxy CA is stored under in the desktop user's NSS trust stores.
# Stable so a re-add is idempotent (delete-then-add on this exact name) and
# teardown can remove precisely the entry we installed.
_NSS_CA_NICKNAME = "clearml-snug-proxy"
# The two NSS cert DBs a Chromium/Chrome build may consult, relative to the
# user's home. The EXTERNAL browser an app opens for Google OAuth
# (``?open_in_browser=1``) is NOT launched by our launcher wrapper, so it can't
# be told to trust the proxy via ``--ignore-certificate-errors-spki-list`` the way
# the Electron app is — it reads TLS trust from NSS instead. Chromium looks in
# both of these depending on build/distro, so the CA must land in BOTH.
_NSS_DB_SUBDIRS = (".pki/nssdb", ".local/share/pki/nssdb")
# The file NSS's ``sql:`` backend creates once a DB is initialized; its presence
# is our "already initialized, skip ``certutil -N``" test.
_NSS_DB_SENTINEL = "cert9.db"


def _real_name(binary_name):
    # type: (str) -> str
    """The preserved-original basename for a shadowed binary/launcher."""
    return binary_name + _REAL_SUFFIX


def _launcher_marker(app_id):
    # type: (str) -> str
    """Distinctive comment embedded in an app's launcher wrapper. The original
    launcher may itself be a ``#!/bin/sh`` script, so the shebang alone can't tell
    "already wrapped" from "real launcher" — we match THIS marker instead (the
    ELF-magic role that _is_elf plays for the SDK binary). Per-app so two apps'
    wrappers stay distinguishable at teardown."""
    return "SNUG {} launcher wrapper".format(app_id)


def _cd_debug_enabled():
    # type: () -> bool
    """Whether verbose ``[snug-app]`` progress lines print. Driven by
    ``CLEARML_SNUG_DEBUG_LOG`` (agent config ``agent.snug.debug_log``) — the same
    flag the shim and proxy honor; the agent exports it into the environment."""
    return str(os.environ.get("CLEARML_SNUG_DEBUG_LOG", "")).strip().lower() in ("1", "true", "yes", "on")


def _cd_log(msg, debug=False):
    # type: (str, bool) -> None
    """Emit a greppable ``[snug-app]`` diagnostic to the agent's stdout.

    Failures and skips (``debug=False``, the default) always print: a prod agent
    that silently fails to wrap the SDK is undiagnosable, so these land on the
    task console like the worker's other SNUG prints. Routine progress lines pass
    ``debug=True`` and print only when ``CLEARML_SNUG_DEBUG_LOG`` is set, so the
    console stays quiet under normal operation. ``flush=True`` so they interleave
    with the task's output. Never raises — logging must not break setup.
    """
    if debug and not _cd_debug_enabled():
        return
    try:
        print("[snug-app] {}".format(msg), flush=True)
    except Exception:
        pass


def _is_elf(path):
    # type: (str) -> bool
    """True iff ``path`` is a regular file whose first 4 bytes are the ELF magic.
    Used to distinguish the not-yet-wrapped SDK binary (an ELF) from our shell
    wrapper (starts with ``#!``). Best-effort: any error -> False."""
    try:
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as fh:
            return fh.read(4) == _ELF_MAGIC
    except Exception:
        return False


def render_sdk_wrapper(proxy_url, ca_filename, binary_name, wrapper_kind="node_bun"):
    # type: (str, str, str, str) -> str
    """Return the ``#!/bin/sh`` wrapper that routes an app's SDK binary through
    the SNUG proxy.

    It exports the proxy env (scoped to this exec only, never the task-wide
    environment) and then re-execs the real SDK binary sitting beside it. Only the
    ``node_bun`` recipe is implemented:

      - ``NODE_EXTRA_CA_CERTS`` — makes a Node/bun HTTPS client trust the proxy's
        proxy CA. We use this and deliberately NOT ``SSL_CERT_FILE``: bun ignores
        ``SSL_CERT_FILE`` for its HTTPS client, and even where it were honored it
        REPLACES the entire root store (so every real upstream cert would then
        fail to verify), whereas ``NODE_EXTRA_CA_CERTS`` APPENDS our CA to the
        existing roots — exactly what we want.
      - ``HTTPS_PROXY`` / ``HTTP_PROXY`` — point outbound requests at the local
        proxy.
      - ``NO_PROXY`` / ``no_proxy`` — exempt loopback. An app may itself run a
        local proxy on 127.0.0.1 (OpenCode's session_proxy does); routing that
        loopback call back through this proxy would double-proxy or dead-loop it.

    ``$(dirname "$0")`` resolves paths relative to the wrapper itself so the CA
    and the real binary are found wherever the (possibly re-downloaded, possibly
    ro-bind-mounted under bwrap) SDK dir lives.

    A ``wrapper_kind`` other than ``node_bun`` (e.g. a non-Node CLI's own CA-env
    recipe) is not implemented; onboarding such an app adds a recipe here.
    """
    if wrapper_kind != "node_bun":
        raise ValueError("unsupported sdk wrapper_kind {!r}".format(wrapper_kind))
    return (
        "#!/bin/sh\n"
        '# SNUG SDK metering wrapper (generated). Routes this SDK binary through\n'
        '# the local metering proxy, then execs the real binary.\n'
        'DIR="$(dirname "$0")"\n'
        '# NODE_EXTRA_CA_CERTS APPENDS the proxy CA to the existing root store.\n'
        'export NODE_EXTRA_CA_CERTS="$DIR/{ca}"\n'
        'export HTTPS_PROXY="{proxy}"\n'
        'export HTTP_PROXY="{proxy}"\n'
        '# Loopback must bypass the proxy: the app may run its own local proxy on\n'
        '# 127.0.0.1, so proxying that call would double-proxy or dead-loop it.\n'
        'export NO_PROXY="localhost,127.0.0.1,::1"\n'
        'export no_proxy="localhost,127.0.0.1,::1"\n'
        'exec "$DIR/{real}" "$@"\n'
    ).format(ca=ca_filename, proxy=proxy_url, real=_real_name(binary_name))


def candidate_home_roots(home):
    # type: (str) -> List[str]
    """Deduped existing dirs to scan for app SDKs: the agent's own home plus
    every /home/* user home. The watcher runs in the agent process but an app may
    download its SDK under the DESKTOP user's home (the app is launched via
    `su <user>`), which need not equal the agent's home -- so scan both."""
    roots = []  # type: List[str]
    seen = set()  # realpath-keyed so a home that is also a /home/* symlink target is not scanned twice
    for cand in ([home] if home else []) + sorted(glob.glob("/home/*")):
        if not cand or not os.path.isdir(cand):
            continue
        key = os.path.realpath(cand)
        if key in seen:
            continue
        seen.add(key)
        roots.append(cand)
    return roots


def find_sdk_dirs(home, sdk):
    # type: (str, SdkBinary) -> List[str]
    """Return the SDK dirs (under ``home`` for ``home_glob``; absolute for
    ``explicit_path``) that hold a not-yet-wrapped ``sdk`` binary and therefore
    need wrapping.

    Dispatch on ``sdk.discovery``:
      - ``home_glob`` — the re-downloaded-SDK layout
        ``<home>/.../<*container_substr*>/<version>/<binary_name>``. We walk for
        that shape and keep only dirs whose binary is STILL an ELF (the real
        binary) — a dir we already wrapped has a ``#!/bin/sh`` binary there
        instead, so it is skipped (idempotent). Best-effort; missing dirs are
        simply absent from the walk.
      - ``explicit_path`` — the binary is baked at a fixed location, so we check
        each dir in ``sdk.explicit_dirs`` directly (``home`` is ignored: the
        paths are absolute). Same ELF-vs-wrapper idempotency as home_glob; a
        symlinked binary counts (``isfile``/``_is_elf`` follow it), which is the
        on-PATH case we wrap. The watcher calls this once per home root and
        dedups, so returning the same absolute dir each call is harmless.
      - ``path_lookup`` — reserved for a PATH-resolved single binary; not yet
        implemented (a real app of that shape adds the mode here).
    """
    if sdk.discovery == "explicit_path":
        matches = []  # type: List[str]
        for d in sdk.explicit_dirs:
            bin_path = os.path.join(d, sdk.binary_name)
            if os.path.isfile(bin_path) and _is_elf(bin_path):
                matches.append(d)
        return matches
    if sdk.discovery != "home_glob":
        _cd_log("SDK discovery mode {!r} not implemented (binary {!r})".format(
            sdk.discovery, sdk.binary_name))
        return []
    matches = []  # type: List[str]
    if not home or not os.path.isdir(home):
        return matches
    for root, _dirs, files in os.walk(home):
        if sdk.binary_name not in files:
            continue
        # Match the '<*container_substr*>/<version>/' shape: the grandparent dir
        # name contains container_substr and the parent is a version segment.
        parent = os.path.basename(root)
        grandparent = os.path.basename(os.path.dirname(root))
        if sdk.container_substr and sdk.container_substr not in grandparent:
            continue
        if sdk.expects_version_parent and not parent:
            continue
        bin_path = os.path.join(root, sdk.binary_name)
        if _is_elf(bin_path):
            matches.append(root)
    return matches


def _sdk_binary_sig(bin_path):
    # type: (str) -> Optional[tuple]
    """``(size, mtime)`` of ``bin_path``, or None if it can't be stat'd. Used to
    tell a settled binary from one still being written."""
    try:
        st = os.stat(bin_path)
        return (st.st_size, st.st_mtime)
    except Exception:
        return None


def _sdk_binary_ready(sdk_dir, bin_path, prev_sig, cur_sig):
    # type: (str, str, Optional[tuple], Optional[tuple]) -> bool
    """Whether the app has FINISHED installing the SDK binary, so wrapping it now
    won't clobber an in-flight download/decompress/verify.

    The app writes the binary non-executable, decompresses it (multi-second for a
    ~260MB ELF), runs its OWN size check, chmods +x, then may write a ``.verified``
    marker before spawning it. If the watcher renames the binary and drops the
    ~400-byte shell wrapper in place DURING that window, the app reads the wrapper
    for its size check (and, as root, leaves a file the desktop user can't chmod),
    and the whole install fails. So gate on:
      - the binary being executable — the app chmods +x only after its own size
        check, so an executable binary is already past verification; AND
      - either the app's ``.verified`` marker existing (its explicit completion
        signal), or the binary's ``(size, mtime)`` being unchanged since the last
        poll tick (decompress finished and nothing is still writing it).
    """
    try:
        if not (os.stat(bin_path).st_mode & 0o111):
            return False
    except Exception:
        return False
    if os.path.exists(os.path.join(sdk_dir, _SDK_VERIFIED_MARKER)):
        return True
    return cur_sig is not None and prev_sig == cur_sig


def _chown_to_owner(ref_path, paths):
    # type: (str, tuple) -> None
    """When running as root, chown each of ``paths`` to the uid/gid that owns
    ``ref_path``.

    The agent runs as root, but the app owns its SDK dir + binary as the desktop
    user and keeps managing them (chmod, re-download). Files we create/rename as
    root would be root-owned, so the desktop user hits EPERM/EACCES on its own
    binary. Hand them back to the SDK dir's owner. No-op when not root or when
    ``ref_path`` is itself root-owned; best-effort per entry (mirrors the NSS
    path's _chown_tree/_chown_created_chain reasoning)."""
    try:
        if os.geteuid() != 0:
            return
    except Exception:
        return
    try:
        st = os.stat(ref_path)
    except Exception:
        return
    if st.st_uid == 0:
        return
    for p in paths:
        try:
            os.chown(p, st.st_uid, st.st_gid)
        except Exception:
            pass


def install_sdk_wrapper(sdk_dir, ca_src_path, proxy_url, sdk):
    # type: (str, str, str, SdkBinary) -> bool
    """Shadow the app's SDK binary in ``sdk_dir`` with a proxy wrapper.

    Steps (idempotent):
      1. Copy the proxy CA to ``sdk_dir/snug_ca.pem``. It MUST sit beside the
         wrapper: an app may run the SDK under ``bwrap``, which gives the sandbox
         a fresh tmpfs ``/tmp`` but ro-binds the SDK dir at its real path — so a
         CA written to ``/tmp`` would be invisible inside the sandbox, while one
         beside the wrapper is reachable via ``$(dirname "$0")``. This runs FIRST
         so that if the proxy hasn't written its CA yet, we bail WITHOUT having
         disturbed the SDK (the watcher just retries next tick).
      2. If ``sdk_dir/<binary>`` is still an ELF, rename it to ``<binary>.real``.
         If it is already our wrapper (not an ELF), do nothing and return False.
      3. Write the ``#!/bin/sh`` wrapper to ``sdk_dir/<binary>`` and chmod +x.

    Returns True iff it performed the wrap this call (i.e. found a real ELF and
    replaced it); False if already wrapped or on any failure.
    """
    bin_path = os.path.join(sdk_dir, sdk.binary_name)
    real_path = os.path.join(sdk_dir, _real_name(sdk.binary_name))

    # Already wrapped (binary is our shell script, not an ELF) -> no-op.
    if not _is_elf(bin_path):
        return False

    try:
        # 1. Drop the CA beside the wrapper BEFORE we disturb the SDK binary (see
        # docstring: bwrap ro-binds this dir, tmpfs /tmp is not shared). If the
        # proxy hasn't generated the CA yet this raises and we return False having
        # touched nothing; the watcher retries.
        ca_dst = os.path.join(sdk_dir, _CA_FILENAME)
        shutil.copyfile(ca_src_path, ca_dst)

        # 2. Preserve the real binary. If a stale <binary>.real already exists from
        # a prior partial wrap, the fresh ELF is the authoritative one -> replace.
        if os.path.exists(real_path):
            os.remove(real_path)
        os.rename(bin_path, real_path)

        # The wrapper execs <binary>.real, so it MUST be executable. An app may
        # write the freshly-downloaded SDK binary mode 0600 and chmod +x only just
        # before it runs it; the watcher can rename it into place during that
        # window, so force +x here rather than relying on the inherited mode.
        real_mode = os.stat(real_path).st_mode
        os.chmod(
            real_path,
            real_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )

        # 3. Write + chmod the wrapper into the original name.
        wrapper = render_sdk_wrapper(proxy_url, _CA_FILENAME, sdk.binary_name, sdk.wrapper_kind)
        with open(bin_path, "w") as fh:
            fh.write(wrapper)
        cur = os.stat(bin_path).st_mode
        os.chmod(
            bin_path,
            cur | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )

        # 4. Hand the files we just created/renamed as root back to the SDK dir's
        # owner (the desktop user), so the app can still chmod/re-download its own
        # binary instead of hitting EPERM/EACCES on a root-owned wrapper/CA.
        _chown_to_owner(sdk_dir, (bin_path, real_path, ca_dst))
        _cd_log("wrapped SDK dir {}".format(sdk_dir), debug=True)
        return True
    except Exception as ex:
        _cd_log("wrap failed for {}: {}".format(sdk_dir, ex))
        return False


def _run_watcher(home, ca_src_path, proxy_url, sdk_binaries, poll_sec=0.5):
    # type: (str, str, str, List[SdkBinary], float) -> None
    """Block forever keeping every SDK dir under ``home`` wrapped.

    This is the watcher body, factored out so it can run as a standalone process
    (see ``start_sdk_watcher`` and the ``__main__`` entry below). Each tick scans
    ``candidate_home_roots`` -> ``find_sdk_dirs`` (per watched SDK, union-deduped)
    and calls ``install_sdk_wrapper`` on each found dir.

    The discovery result is logged on the FIRST tick and only when the set of
    found dirs changes — a per-tick log at 0.5s would flood the console, but the
    first-tick + change lines are enough to see whether the walk ever finds the
    SDK across the scanned roots (the HOME-mismatch symptom is a root that finds
    nothing).

    A found dir is only wrapped once its binary is settled (see
    ``_sdk_binary_ready``): wrapping mid-install races the app's own
    download/decompress/verify and breaks it. ``sig_prev`` carries each binary's
    ``(size, mtime)`` across ticks so the "unchanged since last tick" check works.

    Loops until the process is signalled/terminated by its parent (or dies with
    the container): the sleep is wrapped so a KeyboardInterrupt/SIGTERM breaks
    the loop cleanly instead of dumping a traceback.
    """
    prev_found = None
    sig_prev = {}  # type: dict  # sdk_dir -> (size, mtime) seen on the previous tick
    while True:
        try:
            roots = candidate_home_roots(home)
            found = []  # type: List[tuple]
            seen = set()
            for root in roots:
                for sdk in sdk_binaries:
                    for sdk_dir in find_sdk_dirs(root, sdk):
                        key = (sdk_dir, sdk.binary_name)
                        if key not in seen:
                            seen.add(key)
                            found.append((sdk_dir, sdk))
            cur = set(k for k in seen)
            if cur != prev_found:
                _cd_log("watcher poll roots={} found={}".format(
                    roots, [f[0] for f in found]), debug=True)
                prev_found = cur
            # Rebuilt each tick from the current finds, so a wrapped/removed dir
            # drops out of the stability tracking automatically.
            sig_now = {}  # type: dict
            for sdk_dir, sdk in found:
                bin_path = os.path.join(sdk_dir, sdk.binary_name)
                cur_sig = _sdk_binary_sig(bin_path)
                sig_now[sdk_dir] = cur_sig
                if _sdk_binary_ready(sdk_dir, bin_path, sig_prev.get(sdk_dir), cur_sig):
                    install_sdk_wrapper(sdk_dir, ca_src_path, proxy_url, sdk)
                else:
                    _cd_log("SDK {} not settled yet; deferring wrap".format(sdk_dir), debug=True)
            sig_prev = sig_now
        except Exception as ex:
            # Never let a transient FS error kill the watcher; retry next tick.
            _cd_log("watcher loop error: {}".format(ex))
        try:
            time.sleep(poll_sec)
        except (KeyboardInterrupt, SystemExit):
            break


class _WatcherHandle(object):
    """Handle for the SDK-dir watcher subprocess; call ``.stop()`` to end it."""

    def __init__(self, proc):
        self._proc = proc

    def stop(self):
        # type: () -> None
        # Mirror the proxy teardown: terminate, then kill + reap if it lingers so
        # the watcher isn't left a zombie.
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
        except Exception:
            pass


def start_sdk_watcher(home, ca_src_path, proxy_url, app_id, poll_sec=0.5):
    # type: (str, str, str, str, float) -> _WatcherHandle
    """Start a SUBPROCESS that keeps every watched SDK dir under ``home`` wrapped.

    An app may re-download its SDK dir (a fresh, un-wrapped binary appears) on
    demand and across versions, so a one-shot install would miss later downloads;
    ``_run_watcher`` re-installs the wrapper every ``poll_sec`` seconds.

    The watcher runs as its OWN process, not a thread, for the same reason the
    proxy does: the agent ``os.execv``s to replace its process image with the
    task (``worker.py``), and that destroys every thread in the process — a
    watcher thread would die before it ever polled, so the SDK would never get
    wrapped. ``start_new_session=True`` detaches the child from the parent's
    process group, so neither the ``execv`` nor a group-directed signal takes it
    down; it outlives the exec, polls across the task's life, and is stopped
    explicitly at teardown (or dies with the container).

    The child re-resolves the app profile from ``--app-id`` so it wraps exactly
    the watched SDK binaries this app declares. stdout/stderr are inherited (not
    redirected) so the child's ``[snug-app]`` lines reach the task console.
    Returns a handle with ``.stop()``.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "clearml_agent.helper.app_metering",
            "--app-id", app_id,
            "--home", home or "",
            "--ca", ca_src_path,
            "--proxy-url", proxy_url,
            "--poll-sec", str(poll_sec),
        ],
        start_new_session=True,
    )
    return _WatcherHandle(proc)


def _is_wrapped_launcher(path, marker):
    # type: (str, str) -> bool
    """True iff ``path`` is a regular file containing our launcher ``marker``.

    The original launcher may itself be a ``#!/bin/sh`` launcher, so a shebang
    check would false-positive; we look for the distinctive per-app marker our
    wrapper embeds near its top instead. This is to the launcher what ``_is_elf``
    is to the SDK binary: the "already wrapped?" idempotency test. Best-effort:
    any error -> False."""
    try:
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as fh:
            head = fh.read(512)
        return marker.encode("ascii") in head
    except Exception:
        return False


def render_launcher_wrapper(real_binary, proxy_url, ca_path, spki, marker,
                            h2_assumed_host=None, kind="electron_chromium"):
    # type: (str, str, str, str, str, Optional[str], str) -> str
    """Return the ``#!/bin/sh`` wrapper that routes an Electron launcher through
    the SNUG proxy and makes Chromium trust its CA.

    Unlike the SDK wrapper, this shadows a fixed-path launcher whose preserved
    original and CA live in DIFFERENT directories, so it references both by
    absolute path (no ``$(dirname "$0")`` relocatability needed here).

    Only the ``electron_chromium`` recipe is implemented. It exports the proxy
    env for the launched app and appends the three Chromium switches that work on
    hardened builds (CDP / NODE_OPTIONS / asar injection are all fuse-blocked):

      - ``HTTP_PROXY`` / ``HTTPS_PROXY`` — route outbound requests at the local
        proxy (Node/undici honors these; Chromium's own routing comes from
        ``--proxy-server`` below).
      - ``NODE_EXTRA_CA_CERTS`` — APPENDS the proxy CA to the existing root store
        so the Node side trusts the proxy's cert (see render_sdk_wrapper for why not
        ``SSL_CERT_FILE``).
      - ``CLEARML_SNUG_H2_ASSUMED_HOST`` (when given) — the fallback host the
        shim assigns to h2 streams whose HPACK :authority it can't decode, for the
        app's dynamically-linked (shim-hooked) children.
      - ``--proxy-server`` — point Chromium's renderer at the proxy.
      - ``--proxy-bypass-list=<-loopback>`` — Chromium bypasses loopback by
        default; ``<-loopback>`` removes that exception so app traffic still flows
        through the local (loopback) proxy.
      - ``--ignore-certificate-errors-spki-list=<SPKI>`` — pin ONLY our CA's SPKI
        so the renderer accepts the proxy's cert without disabling cert checks
        globally.

    The switches are placed before ``"$@"`` so the caller's own arguments keep
    their trailing position (matching the SDK wrapper's ``exec … "$@"`` shape).
    """
    if kind != "electron_chromium":
        raise ValueError("unsupported launcher kind {!r}".format(kind))
    h2_export = ""
    if h2_assumed_host:
        h2_export = 'export CLEARML_SNUG_H2_ASSUMED_HOST="{}"\n'.format(h2_assumed_host)
    return (
        "#!/bin/sh\n"
        "# {marker} (generated). Routes the app's Electron/Chromium traffic\n"
        "# through the local metering proxy, then execs the real\n"
        "# launcher with the proxy + CA-trust switches appended.\n"
        'export HTTP_PROXY="{proxy}"\n'
        'export HTTPS_PROXY="{proxy}"\n'
        'export NODE_EXTRA_CA_CERTS="{ca}"\n'
        "{h2}"
        'exec "{real}" '
        # Single-quote each switch: --proxy-bypass-list's ``<-loopback>`` value
        # would otherwise be read by /bin/sh as an input redirection (``< file``)
        # and the wrapper would die with "cannot open -loopback" before exec.
        "'--proxy-server={proxy}' "
        "'--proxy-bypass-list=<-loopback>' "
        "'--ignore-certificate-errors-spki-list={spki}' "
        '"$@"\n'
    ).format(marker=marker, proxy=proxy_url, ca=ca_path, real=real_binary,
             spki=spki, h2=h2_export)


def install_launcher_wrapper(
    launcher_path, proxy_url, ca_path, spki, marker,
    h2_assumed_host=None, kind="electron_chromium", real_name=None,
):
    # type: (str, str, str, str, str, Optional[str], str, Optional[str]) -> bool
    """Shadow an Electron launcher with a proxy wrapper.

    Steps (idempotent, CA-race-safe):
      1. Bail WITHOUT touching anything if the CA isn't ready yet: no ``spki``
         (the proxy hasn't written its SPKI file — see read_ca_spki) or the CA
         cert at ``ca_path`` is missing. A one-shot install has no watcher to
         retry, so setup reads the SPKI (which the proxy writes shortly after
         start) before calling this; the guard is defense-in-depth.
      2. If ``launcher_path`` is already our wrapper (contains ``marker``), no-op
         and return False.
      3. Rename the real launcher to ``<name>.real`` (default: basename +
         ``.real``, beside it), then write the ``#!/bin/sh`` wrapper into the
         original path and chmod +x.

    Returns True iff it performed the wrap this call; False if the CA wasn't
    ready, it was already wrapped, the launcher was absent, or on any failure.
    """
    if not spki:
        _cd_log("launcher wrap skipped: SPKI not ready")
        return False
    if not os.path.isfile(ca_path):
        _cd_log("launcher wrap skipped: CA cert not present at {}".format(ca_path))
        return False

    # Already our wrapper -> no-op (idempotent). This must precede the existence
    # check below so a re-run over an already-wrapped launcher is a clean no-op.
    if _is_wrapped_launcher(launcher_path, marker):
        return False

    if not os.path.isfile(launcher_path):
        _cd_log("launcher wrap skipped: no launcher at {}".format(launcher_path))
        return False

    real_name = real_name or _real_name(os.path.basename(launcher_path))
    real_path = os.path.join(os.path.dirname(launcher_path), real_name)

    try:
        # Preserve the real launcher. A stale ``.real`` from a prior crashed run
        # is not authoritative (the live launcher_path is) -> replace it.
        if os.path.exists(real_path):
            os.remove(real_path)
        os.rename(launcher_path, real_path)

        cur = os.stat(real_path).st_mode
        os.chmod(real_path, cur | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        wrapper = render_launcher_wrapper(
            real_path, proxy_url, ca_path, spki, marker,
            h2_assumed_host=h2_assumed_host, kind=kind,
        )
        with open(launcher_path, "w") as fh:
            fh.write(wrapper)
        cur = os.stat(launcher_path).st_mode
        os.chmod(launcher_path, cur | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        _cd_log("wrapped Electron launcher {}".format(launcher_path), debug=True)
        return True
    except Exception as ex:
        _cd_log("launcher wrap failed for {}: {}".format(launcher_path, ex))
        return False


def uninstall_launcher_wrapper(launcher_path, marker, real_name=None):
    # type: (str, str, Optional[str]) -> bool
    """Restore the original Electron launcher shadowed by
    ``install_launcher_wrapper`` (rename ``<name>.real`` back).

    Idempotent and never raises: a no-op unless ``launcher_path`` is currently our
    wrapper AND the preserved ``.real`` exists. Called from teardown so a normal
    agent run isn't left with the launcher altered. Returns True iff it restored.
    """
    real_name = real_name or _real_name(os.path.basename(launcher_path))
    real_path = os.path.join(os.path.dirname(launcher_path), real_name)
    try:
        # Only restore what WE installed: the current launcher must be our wrapper
        # and the preserved original must exist. Otherwise leave the FS untouched.
        if not _is_wrapped_launcher(launcher_path, marker) or not os.path.isfile(real_path):
            return False
        os.remove(launcher_path)
        os.rename(real_path, launcher_path)
        _cd_log("restored Electron launcher {}".format(launcher_path), debug=True)
        return True
    except Exception as ex:
        _cd_log("launcher restore failed for {}: {}".format(launcher_path, ex))
        return False


def _resolve_desktop_ids(home, user=None):
    # type: (str, Optional[str]) -> Optional[tuple]
    """Return ``(uid, gid)`` the NSS DBs must be owned by, or None to leave
    ownership alone.

    None means "this process is already the right user (not root), or we can't
    resolve a desktop user to switch to" -- in either case running ``certutil``
    in-place as the current user (no privilege drop, no chown) is the best we can
    do. A non-None result means THIS process is root while the DBs live under a
    non-root home, so certutil must be demoted and the created files chowned, or
    Chrome (running as the desktop user) can't read the trust store.

    Mirrors how the rest of the module reasons about root-vs-desktop-user: prefer
    the explicit ``user`` name (``pwd`` lookup, as worker.py's RunasArgv does),
    else fall back to the owner of ``home``. Best-effort; any error -> None.
    """
    try:
        if os.geteuid() != 0:
            return None
    except Exception:
        # No geteuid (non-unix) -> not root, run in place.
        return None
    if user:
        try:
            import pwd
            pw = pwd.getpwnam(user)
            return pw.pw_uid, pw.pw_gid
        except Exception:
            pass
    try:
        st = os.stat(home)
        if st.st_uid != 0:
            return st.st_uid, st.st_gid
    except Exception:
        pass
    return None


def _demote_preexec(uid, gid):
    # type: (int, int) -> callable
    """Return a ``preexec_fn`` that drops the child to ``uid``/``gid`` before it
    execs, so ``certutil`` writes DB files owned by the desktop user (not root).
    setgid MUST precede setuid: once uid is dropped the process can no longer
    change its gid."""
    def _pre():
        os.setgid(gid)
        os.setuid(uid)
    return _pre


def _run_certutil(argv, preexec=None, what="", check=True):
    # type: (List[str], Optional[callable], str, bool) -> int
    """Run one ``certutil`` invocation, return its exit code (or -1 if it could
    not be spawned). Output is captured and logged only on an unexpected failure
    (``check``) so a benign "nickname not found" delete doesn't spam the console.
    Never raises -- NSS trust is best-effort and must not break metering setup."""
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec,
        )
        rc = proc.returncode
        if rc != 0 and check:
            out = (proc.stdout or b"").decode("utf-8", "replace").strip()
            _cd_log("NSS trust: certutil {} rc={} out={!r}".format(what, rc, out))
        return rc
    except Exception as ex:
        _cd_log("NSS trust: certutil {} raised {}".format(what, ex))
        return -1


def _chown_tree(path, uid, gid):
    # type: (str, int, int) -> None
    """chown ``path`` and everything under it to ``uid``/``gid``. Best-effort per
    entry; a single failure never aborts the walk."""
    try:
        os.chown(path, uid, gid)
    except Exception:
        pass
    try:
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                try:
                    os.chown(os.path.join(root, name), uid, gid)
                except Exception:
                    pass
    except Exception:
        pass


def _chown_created_chain(home, leaf, uid, gid):
    # type: (str, str, int, int) -> None
    """chown each dir from ``leaf`` up to (but not including) ``home`` that is
    still root-owned to ``uid``/``gid``.

    We create the ``nssdb`` dir chain as root (``os.makedirs``), so those dirs
    are root-owned; a demoted ``certutil`` then needs to write into them. Only
    root-owned links are touched, so a pre-existing user-owned ``~/.local/share``
    is left as-is."""
    try:
        home_real = os.path.realpath(home)
    except Exception:
        return
    cur = os.path.abspath(leaf)
    while True:
        try:
            if os.path.realpath(cur) == home_real:
                break
            st = os.stat(cur)
            if st.st_uid == 0:
                os.chown(cur, uid, gid)
        except Exception:
            pass
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        cur = parent


def _install_ca_into_nss_db(certutil, db_dir, ca_path, ids):
    # type: (str, str, str, Optional[tuple]) -> bool
    """Ensure ``db_dir`` exists + is initialized and holds the proxy CA trusted
    for TLS. Returns True iff the CA is present in this DB afterwards. Graceful:
    logs and returns False on any failure, never raises."""
    preexec = _demote_preexec(*ids) if ids else None
    try:
        os.makedirs(db_dir, exist_ok=True)
    except Exception as ex:
        _cd_log("NSS trust: could not create DB dir {}: {}".format(db_dir, ex))
        return False

    # Hand the created dir chain to the desktop user BEFORE running certutil, so
    # the demoted certutil can write cert9.db/key9.db into a dir it owns.
    if ids:
        _chown_created_chain(home=os.path.dirname(db_dir), leaf=db_dir, uid=ids[0], gid=ids[1])
        # dirname(db_dir) is the chain's stop point; also chown db_dir itself.
        _chown_tree(db_dir, ids[0], ids[1])

    # Initialize the sql: DB if it isn't already. certutil -A can create the DB
    # in some builds, but init explicitly so the add can't fail on a missing DB.
    if not os.path.isfile(os.path.join(db_dir, _NSS_DB_SENTINEL)):
        rc = _run_certutil(
            [certutil, "-N", "--empty-password", "-d", "sql:" + db_dir],
            preexec, "init " + db_dir, check=False,
        )
        if rc != 0:
            _cd_log("NSS trust: certutil -N rc={} for {} (continuing; -A may create it)".format(rc, db_dir))

    # Idempotent add on a stable nickname: delete any existing entry first so a
    # re-run refreshes rather than erroring on a duplicate nickname. The delete
    # is expected to "fail" (nickname absent) on a first install -> check=False.
    _run_certutil(
        [certutil, "-D", "-d", "sql:" + db_dir, "-n", _NSS_CA_NICKNAME],
        preexec, "predelete " + db_dir, check=False,
    )
    rc = _run_certutil(
        [certutil, "-A", "-d", "sql:" + db_dir,
         "-t", "C,,", "-n", _NSS_CA_NICKNAME, "-i", ca_path],
        preexec, "add " + db_dir,
    )

    # Belt-and-suspenders: re-assert ownership of anything certutil just wrote.
    if ids:
        _chown_tree(db_dir, ids[0], ids[1])

    if rc == 0:
        _cd_log("NSS trust: installed proxy CA into {} as '{}'".format(db_dir, _NSS_CA_NICKNAME), debug=True)
        return True
    _cd_log("NSS trust: failed to add CA to {} (rc={})".format(db_dir, rc))
    return False


def install_ca_into_nss(ca_path, home, user=None):
    # type: (str, str, Optional[str]) -> bool
    """Install the proxy CA into the desktop user's NSS trust stores so the
    EXTERNAL Chrome/Chromium an app opens for Google OAuth trusts it.

    The Electron app trusts the proxy via the launcher wrapper's
    ``--ignore-certificate-errors-spki-list`` switch (no NSS needed), but the
    OAuth login opens in the user's REAL browser, which our wrapper never
    launched and which reads trust from NSS -- so without the CA in NSS that
    login page fails with a cert error. Adds it, trusted for TLS
    (``-t "C,,"``), to BOTH NSS DBs (Chromium consults both).

    Ownership: when this process is root but ``home`` belongs to the desktop
    user, ``certutil`` is run demoted to that user and the created DB dirs/files
    are chowned to them, so Chrome (running as that user) can read the store. See
    ``_resolve_desktop_ids``.

    Best-effort by contract: if ``certutil`` isn't on PATH, the CA cert is
    missing, or any step fails, it logs via ``_cd_log`` and returns without
    raising -- NSS trust must NEVER break proxy/metering setup. Returns True iff
    the CA was installed into at least one NSS DB.
    """
    certutil = shutil.which("certutil")
    if not certutil:
        _cd_log(
            "NSS trust skipped: certutil not on PATH; the external browser used "
            "for Google OAuth will not trust the proxy CA"
        )
        return False
    if not ca_path or not os.path.isfile(ca_path):
        _cd_log("NSS trust skipped: CA cert not present at {!r}".format(ca_path))
        return False
    if not home:
        _cd_log("NSS trust skipped: no desktop home dir given")
        return False

    ids = _resolve_desktop_ids(home, user)
    if ids:
        _cd_log("NSS trust: installing as desktop user uid={} gid={} under {}".format(ids[0], ids[1], home), debug=True)

    ok_any = False
    for sub in _NSS_DB_SUBDIRS:
        db_dir = os.path.join(home, sub)
        try:
            if _install_ca_into_nss_db(certutil, db_dir, ca_path, ids):
                ok_any = True
        except Exception as ex:
            # _install_ca_into_nss_db is already no-raise, but keep the per-DB
            # loop resilient so one bad DB can't skip the other.
            _cd_log("NSS trust: unexpected error on {}: {}".format(db_dir, ex))
    return ok_any


def remove_ca_from_nss(home, user=None):
    # type: (str, Optional[str]) -> bool
    """Remove the proxy CA (nickname ``clearml-snug-proxy``) from both NSS DBs
    under ``home``, mirroring the launcher restore at teardown. Idempotent and
    graceful: missing DBs are skipped, a "nickname not found" is not an error,
    and it never raises. Returns True iff the CA was removed from at least one
    DB."""
    certutil = shutil.which("certutil")
    if not certutil or not home:
        return False
    ids = _resolve_desktop_ids(home, user)
    preexec = _demote_preexec(*ids) if ids else None
    removed_any = False
    for sub in _NSS_DB_SUBDIRS:
        db_dir = os.path.join(home, sub)
        if not os.path.isdir(db_dir):
            continue
        rc = _run_certutil(
            [certutil, "-D", "-d", "sql:" + db_dir, "-n", _NSS_CA_NICKNAME],
            preexec, "remove " + db_dir, check=False,
        )
        if rc == 0:
            removed_any = True
            _cd_log("NSS trust: removed proxy CA from {}".format(db_dir), debug=True)
    return removed_any


def _find_free_port():
    # type: () -> tuple
    """Reserve an OS-assigned free TCP port on loopback for the proxy.

    Returns ``(port, reservation_socket)``. The caller must hold the socket open
    until immediately before spawning the proxy, then close it -- closing it any
    earlier reopens the race this exists to close. Under docker --network=host the
    container's 127.0.0.1 IS the host's loopback, so a fixed port lets two
    app-mode sessions landing on the same worker race for the same bind; drawing
    an OS-assigned ephemeral port per session avoids that. A tiny TOCTOU window
    remains between closing this socket and the proxy's own bind; the caller's
    launch retry (a fresh port per attempt) covers it.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1], s


def launch_proxy(
    proxy_bin, cred_b64, ca_path, ca_key_path, spki_path=None,
    port=8888, log_path=None, decrypt_all=True,
    whitelist_b64=None, default_tokenizer="approx",
):
    # type: (str, Optional[str], str, str, Optional[str], int, Optional[str], bool, Optional[str], str) -> tuple
    """Spawn the bundled SNUG forward proxy.

    Returns ``(proc, log_fh)``: the ``Popen`` handle and the log file object we
    opened for the child (or ``None`` when ``log_path`` is not given). The caller
    owns closing ``log_fh`` at teardown so the parent's copy isn't leaked.

    The proxy reads its config from these environment variables (see
    clearml_snug/proxy/src/main.rs):
      - ``CLEARML_SNUG_PROXY_PORT`` — listen port (127.0.0.1 only).
      - ``CLEARML_SNUG_PROXY_CA`` / ``CLEARML_SNUG_PROXY_CA_KEY`` — persistent CA
        cert + key path (persisted so already-running clients keep trusting it
        across proxy restarts).
      - ``CLEARML_SNUG_PROXY_DECRYPT_ALL`` — when ``decrypt_all`` (app-mode),
        decrypt every CONNECT target, not just known providers, so the app's full
        renderer traffic is decrypted + metered.
      - ``CLEARML_SNUG_PROXY_CA_SPKI_FILE`` — where the proxy writes the CA's
        SPKI-SHA256 (base64) shortly after start; the launcher reads it back (see
        read_ca_spki) to pin the CA via ``--ignore-certificate-errors-spki-list``.
      - ``CLEARML_SNUG_CRED`` — base64 credential descriptor; when present the
        proxy runs the in-process reporter, otherwise it is log-only.
      - ``CLEARML_SNUG_WHITELIST`` — when ``whitelist_b64`` is given, the base64
        v1 whitelist JSON the proxy reads for its report gate + per-host
        tokenizer. Omitted when None, so the proxy falls back to its meter-all
        default (``default_action="meter"``, no rules).

    ``CLEARML_SNUG_PARSE_USAGE`` and ``CLEARML_SNUG_DEFAULT_TOKENIZER`` are also
    exported for parity with the LD_PRELOAD shim's env contract and forward-compat,
    but the proxy binary reads NEITHER — they are currently inert for the proxy,
    which always runs the usage scanner on the decrypted provider traffic. They are
    left set because they are harmless. ``default_tokenizer`` comes from the app
    profile so an app whose primary provider isn't Anthropic doesn't inherit the
    Claude tokenizer as its fallback once the proxy honors this env.

    ``log_path`` (when given) captures the proxy's stdout+stderr to a file;
    otherwise they are inherited.
    """
    env = dict(os.environ)
    env["CLEARML_SNUG_PARSE_USAGE"] = "1"
    env["CLEARML_SNUG_PROXY_CA"] = ca_path
    env["CLEARML_SNUG_PROXY_CA_KEY"] = ca_key_path
    env["CLEARML_SNUG_PROXY_PORT"] = str(port)
    env["CLEARML_SNUG_DEFAULT_TOKENIZER"] = default_tokenizer
    if decrypt_all:
        env["CLEARML_SNUG_PROXY_DECRYPT_ALL"] = "1"
    if spki_path:
        env["CLEARML_SNUG_PROXY_CA_SPKI_FILE"] = spki_path
    if cred_b64:
        env["CLEARML_SNUG_CRED"] = cred_b64
    if whitelist_b64:
        env["CLEARML_SNUG_WHITELIST"] = whitelist_b64

    stdio = None
    if log_path:
        # Unbuffered append so a tail -f on the log shows proxy activity live.
        stdio = open(log_path, "ab", buffering=0)

    proc = subprocess.Popen(
        [proxy_bin],
        env=env,
        stdout=stdio if stdio is not None else None,
        stderr=subprocess.STDOUT if stdio is not None else None,
    )
    return proc, stdio


def read_ca_spki(spki_path, timeout=10.0, poll_sec=0.1, proc=None, settle_sec=0.3):
    # type: (str, float, float, Optional[object], float) -> Optional[str]
    """Wait for + read the CA's SPKI-SHA256 (base64) the proxy writes to
    ``spki_path``.

    Race-safe by design: the proxy writes this file a short moment AFTER it starts
    (it must generate/load the CA first), so this polls until the file exists and
    is non-empty, up to ``timeout`` seconds. Returns the stripped SPKI value, or
    None if it never appears — the caller then skips the launcher wrap rather than
    pinning an empty SPKI (which would make Chromium reject every cert).

    The file contains the raw base64 value per the shared contract; a leading
    ``SPKI=`` (the stdout spelling) is tolerated and stripped defensively.

    When ``proc`` (the proxy's ``Popen`` handle) is given, the liveness of the
    proxy gates the result two ways: (1) a value found on disk is confirmed only if
    ``proc`` is still alive ``settle_sec`` later, returning None instead of the
    stale value if it isn't; and (2) while still waiting for the file, if ``proc``
    has already exited we bail immediately rather than waiting out the full
    ``timeout``. The proxy writes its CA + SPKI and starts the reporter/whitelist
    BEFORE its final listen bind (see clearml_snug/proxy/src/main.rs), so under a
    port collision (two app-mode sessions sharing a --network=host worker) a losing
    instance still leaves a fully-formed but orphaned SPKI on disk moments before
    exiting on the bind failure. Without (1) the caller would pin/trust/route
    through a proxy that's already dead — or worse, a DIFFERENT session's live
    proxy answering the same port with a mismatched CA; without (2) the launch
    retry would stall ``timeout`` seconds on every failed attempt. Skipped when
    ``proc`` is None.
    """
    deadline = time.time() + max(0.0, timeout)
    while True:
        try:
            if os.path.isfile(spki_path):
                with open(spki_path, "r") as fh:
                    val = fh.read().strip()
                if val:
                    if val.startswith("SPKI="):
                        val = val[len("SPKI="):].strip()
                    if val:
                        if proc is not None:
                            step = max(0.01, min(poll_sec, 0.05))
                            settle_deadline = time.time() + max(0.0, settle_sec)
                            while proc.poll() is None and time.time() < settle_deadline:
                                try:
                                    time.sleep(step)
                                except (KeyboardInterrupt, SystemExit):
                                    return None
                            if proc.poll() is not None:
                                return None
                        return val
        except Exception:
            pass
        # Fast-bail: a dead proxy will never publish the SPKI, so don't wait out
        # the whole timeout (matters for the caller's per-attempt retry latency).
        if proc is not None and proc.poll() is not None:
            return None
        if time.time() >= deadline:
            return None
        try:
            time.sleep(poll_sec)
        except (KeyboardInterrupt, SystemExit):
            return None


def app_mode_requested(config):
    # type: (object) -> str
    """Return the raw ``agent.snug.app_mode`` name the operator configured, or ""
    when app-mode is off.

    This is deliberately separate from ``resolve_app_profile`` so the caller can
    tell "app-mode off" (return "") apart from "app-mode requested but the name
    doesn't resolve to a known profile" (returns the non-empty name while
    resolve_app_profile returns None). The latter is a misconfiguration that MUST
    fail closed under mandatory metering — otherwise a typo like ``claude-desktop``
    (hyphen) for ``claude_desktop`` would silently run the app un-metered. Same
    config-read idiom as elsewhere; any read error -> "" (treated as off, since a
    config we can't read can't be asserted to have requested app-mode).
    """
    try:
        return str(config.get("agent.snug.app_mode", "") or "").strip()
    except Exception:
        return ""


def resolve_app_profile(config):
    # type: (object) -> Optional[AppProfile]
    """Resolve the opted-in app profile for this agent, or None when app-mode is
    off (so a plain agent is unaffected) OR when the configured name isn't a known
    profile.

    ``agent.snug.app_mode`` names the app profile to enable (e.g.
    ``"claude_desktop"``); unset or "" means off. An ``app_mode`` naming a profile
    that isn't registered logs and returns None — the caller uses
    ``app_mode_requested`` to distinguish that misconfiguration (fail closed) from
    the off case. Follows the existing ``agent.snug.*`` config-read idiom
    (``config.get(key, default)``); any read error -> None.
    """
    name = app_mode_requested(config) or None
    if not name:
        return None
    profile = BUILTIN_PROFILES.get(name)
    if profile is None:
        _cd_log("app-mode {!r} is not a known app profile; metering not started".format(name))
        return None
    return profile


class AppMeteringHandle(object):
    """Bundle of the live app-mode resources (proxy process + wrapper watcher +
    the shadowed Electron launcher(s) + any NSS trust).

    ``.teardown()`` stops the watcher, terminates the proxy, restores the original
    Electron launcher(s), and removes the CA from NSS (so a normal agent run isn't
    left altered); it is safe to call more than once and never raises. The worker
    calls it on both the normal fd-release path and the error/finally path.
    """

    def __init__(self, proxy_proc, watcher, proxy_url, ca_path, log_fh=None,
                 launchers=None, nss_home=None, nss_user=None, metering_active=False):
        self.proxy_proc = proxy_proc
        self.watcher = watcher
        self.proxy_url = proxy_url
        self.ca_path = ca_path
        self.log_fh = log_fh
        # The Electron launchers we shadowed, as (launcher_path, marker,
        # real_name) tuples; teardown renames each ``.real`` back. Empty when no
        # launcher was wrapped (e.g. SPKI never became ready).
        self.launchers = list(launchers or [])
        # When set, the desktop home (and user) whose NSS trust stores we added
        # the proxy CA to; teardown removes it so a normal agent run leaves the
        # user's browser trust unaltered. None when NSS trust was never installed.
        self.nss_home = nss_home
        self.nss_user = nss_user
        # False whenever the proxy wasn't confirmed alive (see setup_app_metering's
        # proxy_ready check) -- the process is still spawned and this handle still
        # owns tearing it down, but nothing was wrapped/trusted/watched, so the
        # caller must not report metering as active.
        self.metering_active = metering_active
        self._torn_down = False

    def teardown(self):
        # type: () -> None
        if self._torn_down:
            return
        self._torn_down = True
        # Restore the shadowed launcher(s) first: pure file renames that must
        # happen regardless of whether the proxy/watcher teardown below throws.
        for launcher_path, marker, real_name in self.launchers:
            try:
                uninstall_launcher_wrapper(launcher_path, marker, real_name=real_name)
            except Exception:
                pass
        # Remove the proxy CA from the desktop user's NSS trust stores (mirrors
        # the launcher restore). Best-effort; must not break the rest of teardown.
        if self.nss_home is not None:
            try:
                remove_ca_from_nss(self.nss_home, user=self.nss_user)
            except Exception:
                pass
        if self.watcher is not None:
            try:
                self.watcher.stop()
            except Exception:
                pass
        if self.proxy_proc is not None:
            try:
                self.proxy_proc.terminate()
                try:
                    self.proxy_proc.wait(timeout=5)
                except Exception:
                    # terminate() didn't land in time; kill and then reap so the
                    # proxy isn't left a zombie holding the listen port.
                    self.proxy_proc.kill()
                    try:
                        self.proxy_proc.wait(timeout=5)
                    except Exception:
                        pass
            except Exception:
                pass
        # Close our copy of the child's log file (Popen doesn't own file objects
        # passed as stdout), so the long-lived agent doesn't leak one fd per task.
        if self.log_fh is not None:
            try:
                self.log_fh.close()
            except Exception:
                pass


def setup_app_metering(
    profile, session, task_id, project, home, proxy_bin, config,
    port=None, log_path=None, worker_id="", user="", spki_timeout=10.0,
    launch_attempts=3,
):
    # type: (AppProfile, object, str, str, str, str, object, Optional[int], Optional[str], str, str, float, int) -> AppMeteringHandle
    """Orchestrate proxy metering for one task launch of ``profile``'s app.

    The returned handle's ``.metering_active`` is the single source of truth the
    caller uses to decide whether the app may run: it is True ONLY when the proxy
    came up live AND (for a launcher-based app) every launcher was actually
    wrapped to route through it. A live proxy that nothing routes through is NOT
    metering, so it does not count. The caller treats ``metering_active`` False as
    a hard failure (the task must not run un-metered) — see the worker.

    Wiring (all driven by the app profile; the generic mechanism stays in
    snug.py):
      1. Build the base64 credential descriptor (``build_shim_descriptor_b64``)
         for the proxy's in-process reporter — the fd channel doesn't survive the
         Electron/bwrap spawn chain, so the proxy takes creds via env.
      2. Launch the bundled proxy on ``127.0.0.1:<port>`` with a persistent CA,
         the profile's decrypt-all policy + tokenizer, and the SPKI-file handoff.
         ``port=None`` (the default) picks a free loopback port via a held
         reservation socket (see ``_find_free_port``) instead of a fixed one, so
         two app-mode sessions sharing a --network=host worker never race for the
         same bind. The launch is retried up to ``launch_attempts`` times, each
         attempt drawing a FRESH free port, so the small TOCTOU window between
         releasing the reservation and the proxy's own bind (a rare ephemeral-port
         reuse) self-heals instead of failing the session. An explicit ``port``
         pins that port and forces a single attempt (retrying a fixed port cannot
         clear an in-use bind).
      3. Read back the CA SPKI the proxy writes (race-safe wait, additionally
         gated on the proxy still being alive — see ``read_ca_spki``) and wrap each
         of the profile's Electron launchers so Chromium routes through the proxy
         and pins the CA. One-shot (no watcher): launchers are installed once in
         the image and not re-fetched at runtime, unlike SDK dirs.
      4. Install the CA into NSS when the profile opens an external OAuth browser.
      5. Start the SDK-dir watcher when the profile has a watched SDK, so every
         (re-downloaded) SDK binary is wrapped to route through the proxy.
         Steps 3-5 run only once a live proxy is confirmed (step 2's retry loop);
         a proxy that never came up leaves ``metering_active`` False and the app
         is not launched. The launcher wrap (step 3) is REQUIRED for metering on a
         launcher-based app: an unwrapped Electron shell bypasses the proxy
         entirely, so if any launcher fails to wrap, ``metering_active`` is False.
         NSS trust (step 4) is best-effort and does not gate metering.

    It deliberately does NOT touch the task-wide environment: the SDK-only
    ``HTTPS_PROXY`` lives INSIDE the SDK wrapper and the Chromium switches/env
    (incl. the h2 assumed-host) live INSIDE the launcher wrapper — never in
    ``_get_job_os_envs``. Returns a handle whose ``.teardown()`` stops the
    watcher, terminates the proxy, and restores the original launcher(s).
    """
    cred_b64 = build_shim_descriptor_b64(
        session=session,
        task_id=task_id,
        worker_id=worker_id or "",
        user=user or "",
        project=project or "",
    )

    # The proxy persists its CA cert + key beside itself under the home dir so
    # the wrapper watcher can copy the cert into each SDK dir. 127.0.0.1 only.
    ca_dir = os.path.join(home, ".clearml_snug")
    try:
        os.makedirs(ca_dir, exist_ok=True)
    except Exception:
        ca_dir = home
    ca_path = os.path.join(ca_dir, "snug_proxy_ca.pem")
    ca_key_path = os.path.join(ca_dir, "snug_proxy_ca.key.pem")
    spki_path = os.path.join(ca_dir, _CA_SPKI_FILENAME)

    # The operator-controlled whitelist (base64 v1 JSON) the proxy reads for its
    # report gate + per-host tokenizer, extended with this app's own rows (merged
    # as additions: admin rules still win on a host collision). A build failure
    # must never abort setup: fall back to None so the proxy uses its meter-all
    # default.
    try:
        whitelist_b64 = build_whitelist_env(
            session, extra_rules=profile.whitelist_contribution)
    except Exception as ex:
        _cd_log("whitelist build failed ({}); proxy will meter-all".format(ex))
        whitelist_b64 = None

    # The uid/euid + resolved home are the crux of the prod wrap failure: the
    # watcher walks THIS process's ``home``, so if the agent runs as root while
    # the app downloads its SDK under the desktop user's home, the walk finds
    # nothing and the SDK is never wrapped.
    try:
        _uid, _euid = os.getuid(), os.geteuid()
    except Exception:
        _uid = _euid = None
    _cd_log(
        "setup app={} home={!r} roots={} uid={} euid={} proxy_bin={!r} ca_path={!r}".format(
            profile.app_id, home, candidate_home_roots(home), _uid, _euid, proxy_bin, ca_path
        ),
        debug=True,
    )

    # Bring up a live proxy, retrying with a FRESH free port each attempt. Under
    # docker --network=host the container's 127.0.0.1 is the HOST's loopback, so
    # two concurrent sessions on one worker each need their own port; a fixed port
    # (and an explicit ``port`` here) can't self-heal a collision, so we only
    # retry when picking the port ourselves. The dominant failure — a losing
    # racer whose proxy exits(1) on bind — is detected fast by read_ca_spki's
    # liveness gate (the loser wrote a real-looking SPKI before dying), so a
    # retry costs little. ``handle`` holds the live proxy on success.
    explicit_port = port is not None
    attempts = 1 if explicit_port else max(1, launch_attempts)
    handle = None
    spki = None
    for attempt in range(1, attempts + 1):
        # A fresh OS-assigned free port per attempt is what makes the retry
        # self-heal: the reservation socket holds it until right before Popen so
        # nothing else on this host can grab it in between.
        if explicit_port:
            bind_port, port_reservation = port, None
        else:
            bind_port, port_reservation = _find_free_port()
        proxy_url = "http://127.0.0.1:{}".format(bind_port)
        if port_reservation is not None:
            port_reservation.close()

        proxy_proc, log_fh = launch_proxy(
            proxy_bin=proxy_bin,
            cred_b64=cred_b64,
            ca_path=ca_path,
            ca_key_path=ca_key_path,
            spki_path=spki_path,
            port=bind_port,
            log_path=log_path,
            decrypt_all=profile.decrypt_all,
            whitelist_b64=whitelist_b64,
            default_tokenizer=profile.default_tokenizer,
        )
        attempt_handle = AppMeteringHandle(
            proxy_proc=proxy_proc,
            watcher=None,
            proxy_url=proxy_url,
            ca_path=ca_path,
            log_fh=log_fh,
        )

        # Read back the CA SPKI the proxy publishes shortly after start, gated on
        # the proxy still being alive (see read_ca_spki's proc/settle_sec): the
        # proxy writes CA + SPKI BEFORE its listen bind, so a losing racer leaves
        # a fully-formed but orphaned SPKI on disk moments before exiting(1).
        spki = read_ca_spki(spki_path, timeout=spki_timeout, proc=proxy_proc)
        if spki:
            handle = attempt_handle
            break

        # Failed attempt: tear the (dead or unresponsive) proxy down before the
        # next try so it can't orphan holding its port, then back off briefly.
        exit_code = proxy_proc.poll()
        _cd_log(
            "proxy launch attempt {}/{} on port {} failed (rc={}); {}".format(
                attempt, attempts, bind_port, exit_code,
                "retrying with a fresh port" if attempt < attempts else "no attempts left",
            )
        )
        attempt_handle.teardown()
        if attempt < attempts:
            try:
                time.sleep(min(0.5, 0.2 * attempt))
            except (KeyboardInterrupt, SystemExit):
                break

    if handle is None:
        # Every attempt failed: no live proxy. Return an inactive handle (the
        # last proxy is already torn down) so the caller fails the task rather
        # than launching the app un-metered. metering_active defaults False.
        _cd_log(
            "app-mode metering NOT active: no live proxy after {} attempt(s)".format(attempts)
        )
        return AppMeteringHandle(
            proxy_proc=None, watcher=None, proxy_url=proxy_url, ca_path=ca_path,
        )

    # Proxy is live + confirmed. Wrap the launcher(s), install NSS trust, and
    # start the watcher. If anything here raises, tear the live proxy down (and
    # restore any launcher we wrapped) instead of orphaning it.
    try:
        # Wrap the Electron launcher(s) so Chromium routes through the proxy and
        # pins the CA. For a launcher-based app this is REQUIRED for metering: an
        # unwrapped shell talks to the LLM directly, bypassing the proxy, so a
        # wrap failure means this session would run un-metered — metering_active
        # stays False and the caller fails the task.
        launchers_wrapped = True
        for launcher in profile.launchers or ():
            marker = _launcher_marker(profile.app_id)
            # Clear any stale wrapper left by a prior run FIRST: it points at a
            # now-dead proxy on an old (ephemeral) port, and install_launcher_wrapper
            # treats an already-wrapped launcher as a no-op (returns False). Without
            # this restore the stale wrapper would be read as a wrap failure and the
            # session would fail-hard forever on the same filesystem. Restoring it
            # here lets the install below re-point it at THIS proxy. A no-op when the
            # launcher isn't wrapped (the normal first-run case).
            uninstall_launcher_wrapper(launcher.path, marker)
            wrapped = install_launcher_wrapper(
                launcher_path=launcher.path,
                proxy_url=handle.proxy_url,
                ca_path=ca_path,
                spki=spki,
                marker=marker,
                h2_assumed_host=profile.h2_assumed_host,
                kind=launcher.kind,
            )
            if wrapped:
                handle.launchers.append((launcher.path, marker, None))
            else:
                launchers_wrapped = False
                _cd_log(
                    "launcher wrap FAILED for {!r}; app would bypass the proxy".format(
                        launcher.path
                    )
                )

        handle.metering_active = launchers_wrapped
        if not launchers_wrapped:
            return handle

        # Bake the proxy CA into the desktop user's NSS trust stores so the
        # EXTERNAL browser the app opens for Google OAuth trusts the proxy CA.
        # Strictly best-effort and NOT a metering gate: wrapped in its own guard
        # so a certutil/NSS failure is logged but never fails the session. Record
        # nss_home only on success so teardown removes exactly what we installed.
        if profile.external_oauth_browser:
            try:
                if os.path.isfile(ca_path):
                    if install_ca_into_nss(ca_path=ca_path, home=home, user=user):
                        handle.nss_home = home
                        handle.nss_user = user
                else:
                    _cd_log("NSS trust skipped: CA cert not present at {}".format(ca_path))
            except Exception as ex:
                _cd_log("NSS trust: install step errored (ignored): {}".format(ex))

        # Start the SDK-dir watcher only when the profile has a watched SDK.
        if any(getattr(s, "watched", False) for s in (profile.sdk_binaries or ())):
            handle.watcher = start_sdk_watcher(
                home=home,
                ca_src_path=ca_path,
                proxy_url=handle.proxy_url,
                app_id=profile.app_id,
            )
    except Exception:
        handle.teardown()
        raise

    return handle


if __name__ == "__main__":
    # CLI entry so the watcher can run as its own process (see start_sdk_watcher
    # for why it must outlive the agent's os.execv). It re-resolves the app
    # profile from --app-id and watches that app's watched SDK binaries.
    import argparse

    _parser = argparse.ArgumentParser(
        description="SNUG SDK-wrapper watcher — re-installs the app's SDK wrapper "
                    "under --home so the SDK's traffic keeps routing through the proxy.",
    )
    _parser.add_argument("--app-id", required=True)
    _parser.add_argument("--home", required=True)
    _parser.add_argument("--ca", required=True)
    _parser.add_argument("--proxy-url", required=True)
    _parser.add_argument("--poll-sec", type=float, default=0.5)
    _args = _parser.parse_args()
    _profile = BUILTIN_PROFILES.get(_args.app_id)
    if _profile is None:
        _cd_log("watcher: unknown app-id {!r}; nothing to watch".format(_args.app_id))
        sys.exit(0)
    _watched = [s for s in (_profile.sdk_binaries or ()) if getattr(s, "watched", False)]
    _run_watcher(_args.home, _args.ca, _args.proxy_url, _watched, _args.poll_sec)
