"""
clearml-agent install-bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Downloads the latest .whl release from the clearml-agent-bootstrap GitHub
repository and installs it into the currently running Python environment.

Usage:
    clearml-agent install-bootstrap
    clearml-agent install-bootstrap --check        # only print what would be installed
    clearml-agent install-bootstrap --version TAG   # install a specific release
    clearml-agent install-bootstrap --from-file X   # install a local .whl
    clearml-agent install-bootstrap --force         # force reinstall
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


BOOTSTRAP_REPO = "clearml/clearml-agent-bootstrap"
GITHUB_API_VERSION = "2022-11-28"
RELEASES_API = "https://api.github.com/repos/{repo}/releases".format(
    repo=BOOTSTRAP_REPO
)
ASSET_API = "https://api.github.com/repos/{repo}/releases/assets/{{asset_id}}".format(
    repo=BOOTSTRAP_REPO
)

CHUNK_SIZE = 64 * 1024  # 64 KiB


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------


def _api_headers(accept: str = "application/vnd.github+json") -> dict:
    """Return standard headers for GitHub API requests."""
    return {
        "Accept": accept,
        "User-Agent": "clearml-agent",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }


def _fetch_json(url: str) -> object:
    """Fetch a URL and return the parsed JSON response."""
    req = urllib.request.Request(url, headers=_api_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _get_installed_version() -> str | None:
    """Get the currently installed version of clearml-agent-bootstrap, or None."""
    try:
        from clearml_agent_bootstrap.version import __version__

        return __version__
    except ImportError:
        return None
    except Exception:
        return None


def _find_whl(version: str | None = None) -> tuple[str, int, str]:
    """Return ``(tag, asset_id, filename)`` for a release with a .whl asset.

    When *version* is None the ``/releases/latest`` endpoint is used (returns
    the most recent non-prerelease, non-draft release sorted by created_at).

    When a version string is given the release whose tag matches is returned.
    The lookup is format-agnostic: the string is tried verbatim, then with a
    ``v`` prefix added or removed.
    """
    if version:
        candidates = [version]
        if version.startswith("v"):
            candidates.append(version[1:])
        else:
            candidates.append("v" + version)

        release = None
        for tag in candidates:
            url = "{}/tags/{}".format(RELEASES_API, tag)
            try:
                release = _fetch_json(url)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise

        if not release:
            raise RuntimeError(
                "No release found matching version '{}'.".format(version)
            )
    else:
        url = "{}/latest".format(RELEASES_API)
        try:
            release = _fetch_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError("No releases found in {}.".format(BOOTSTRAP_REPO))
            raise

    tag: str = release.get("tag_name", "?")
    for asset in release.get("assets", []):
        name: str = asset.get("name", "")
        if name.endswith(".whl"):
            return tag, int(asset["id"]), name

    raise RuntimeError(
        "Release '{}' in {} has no .whl asset attached.".format(tag, BOOTSTRAP_REPO)
    )


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download(asset_id: int, filename: str, dest: str) -> None:
    """
    Download a release asset via the GitHub API using the asset ID.

    Composes the target path and checks writability before making any network
    request.  Uses Accept: application/octet-stream so the API either streams
    the binary directly or issues a redirect — both cases are handled by
    urllib's redirect handler.  Streams in chunks to avoid loading the full
    file into memory.
    """
    target_dir = os.path.dirname(dest)
    if not os.access(target_dir, os.W_OK):
        raise OSError("Target directory is not writable: {}".format(target_dir))

    url = ASSET_API.format(asset_id=asset_id)
    print("Downloading: {} (asset id={})".format(filename, asset_id))
    req = urllib.request.Request(url, headers=_api_headers("application/octet-stream"))
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        written = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = min(written * 100 // total, 100)
                    print("\r  {}%".format(pct), end="", flush=True)
    if total:
        print()


# ---------------------------------------------------------------------------
# Package-manager detection
# ---------------------------------------------------------------------------


def _find_installer() -> tuple[list[str], str]:
    """
    Return ``(command_prefix, force_flag)`` for installing a wheel,
    probing in priority order:
      1. pip  (via the current interpreter — guarantees correct env)
      2. pip3 (standalone binary)
      3. uv   (uv pip install)
      4. pipx (pipx install)
    """
    candidates = [
        (
            "pip",
            [sys.executable, "-m", "pip", "--version"],
            [sys.executable, "-m", "pip", "install"],
            "--force-reinstall",
        ),
        ("pip3", ["pip3", "--version"], ["pip3", "install"], "--force-reinstall"),
        ("uv", ["uv", "--version"], ["uv", "pip", "install"], "--force-reinstall"),
        ("pipx", ["pipx", "--version"], ["pipx", "install"], "--force"),
    ]
    for name, probe, prefix, force_flag in candidates:
        try:
            subprocess.run(
                probe,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Using package manager: {}".format(name))
            return prefix, force_flag
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    raise RuntimeError(
        "No supported package manager found. "
        "Please install pip, uv, or pipx and try again."
    )


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------


def _install_and_verify(
    wheel_path: str, current_version: str | None, force: bool = False
) -> None:
    """
    Install the wheel, flush the import cache, and print whether the
    version changed.  Exits with a non-zero status on failure.
    """
    cmd, force_flag = _find_installer()
    if force:
        cmd = cmd + [force_flag]
    cmd = cmd + [wheel_path]
    print("Running: {}".format(" ".join(cmd)))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "ERROR: Installation failed (exit code {}).".format(result.returncode),
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    # invalidate cached imports so we can re-check the version
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("clearml_agent_bootstrap"):
            del sys.modules[mod_name]

    new_version = _get_installed_version()
    if new_version:
        if current_version and current_version == new_version and not force:
            print(
                "clearml-agent-bootstrap v{} is already up-to-date".format(new_version)
            )
        else:
            print(
                "Successfully installed clearml-agent-bootstrap v{}".format(new_version)
            )
    else:
        print("WARNING: installation completed but package could not be imported")


# ---------------------------------------------------------------------------
# Public entry point called by __main__.py
# ---------------------------------------------------------------------------


def install_bootstrap(
    check: bool = False,
    force: bool = False,
    version: str | None = None,
    from_file: str | None = None,
    **kwargs,
) -> None:
    """
    Install or update the clearml-agent-bootstrap package.

    By default the wheel is fetched from GitHub Releases.  Use
    ``--from-file`` to install a local ``.whl`` instead (useful for
    testing internal builds that haven't been published yet).

    This command does not require a ClearML server connection and can be
    used before ``clearml-agent init`` has been run.
    """
    current_version = _get_installed_version()

    if current_version:
        print(
            "Currently installed: clearml-agent-bootstrap v{}".format(current_version)
        )
    else:
        print("clearml-agent-bootstrap is not currently installed")

    # ---- local wheel path ----
    if from_file:
        wheel_path = os.path.expanduser(from_file)
        if not os.path.isfile(wheel_path):
            print("ERROR: file not found: {}".format(wheel_path), file=sys.stderr)
            sys.exit(1)
        if not wheel_path.endswith(".whl"):
            print(
                "ERROR: expected a .whl file, got: {}".format(wheel_path),
                file=sys.stderr,
            )
            sys.exit(1)

        print("Using local wheel: {}".format(wheel_path))

        if check:
            print("(--check mode: no download or installation performed)")
            return

        _install_and_verify(wheel_path, current_version, force=force)
        return

    # ---- locate the release on GitHub ----
    try:
        tag, asset_id, filename = _find_whl(version=version)
    except urllib.error.URLError as exc:
        print("ERROR: Could not reach GitHub API: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    print("Latest bootstrap release : {}".format(tag))
    print("Asset                    : {}".format(filename))

    if check:
        print("(--check mode: no download or installation performed)")
        return

    # ---- download & install ----
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, filename)
        try:
            _download(asset_id, filename, dest)
        except urllib.error.URLError as exc:
            print("ERROR: Download failed: {}".format(exc), file=sys.stderr)
            sys.exit(1)

        _install_and_verify(dest, current_version, force=force)

    print("clearml-agent-bootstrap installed successfully.")
