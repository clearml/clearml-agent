from pathlib import Path

import pytest

from clearml_agent.helper.package import uv_api


@pytest.mark.parametrize(
    "windows, version, expected",
    (
        (False, None, Path("venv", "bin", "python")),
        (False, "3.10", Path("venv", "bin", "python3.10")),
        (True, None, Path("venv", "Scripts", "python.exe")),
        (True, "3.10", Path("venv", "Scripts", "python.exe")),
    ),
)
def test_venv_python_path(monkeypatch, windows, version, expected):
    monkeypatch.setattr(uv_api, "is_windows_platform", lambda: windows)
    assert Path(str(uv_api.venv_python_path("venv", version))) == expected


@pytest.mark.parametrize(
    "windows, expected",
    (
        (False, Path("venv", "bin", "uv")),
        (True, Path("venv", "Scripts", "uv.exe")),
    ),
)
def test_venv_uv_path(monkeypatch, windows, expected):
    monkeypatch.setattr(uv_api, "is_windows_platform", lambda: windows)
    assert Path(str(uv_api.venv_uv_path("venv"))) == expected
