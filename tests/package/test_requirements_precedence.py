from types import SimpleNamespace
from unittest.mock import Mock

from clearml_agent.commands.worker import Worker


class _Config(object):
    def get(self, key, default=None):
        return default


def test_task_requirements_take_precedence_over_repo_requirements(tmp_path):
    worker = Worker.__new__(Worker)
    worker._session = SimpleNamespace(config=_Config(), debug_mode=False)
    worker.package_api = None
    worker.global_package_api = None
    worker.is_conda = False
    worker._install_poetry_requirements = Mock(return_value=None)
    worker._install_uv_requirements = Mock(return_value=None)
    worker._install_patched_python_requirements = Mock(return_value=True)

    package_api = SimpleNamespace(
        cwd=None,
        upgrade_pip=Mock(),
        set_selected_package_manager=Mock(),
        out_of_scope_install_package=Mock(),
        install_from_file=Mock(),
    )
    requirements_manager = SimpleNamespace(
        set_cwd=Mock(),
        replace=Mock(side_effect=lambda text: text),
        post_install=Mock(),
    )
    execution = SimpleNamespace(working_dir=".", entry_point="train.py")
    repo_info = SimpleNamespace(root=tmp_path)
    (tmp_path / "requirements.txt").write_text("repo-only-package==1.0\n")
    cached_requirements = {"pip": "-r subdir/step_requirements.txt"}

    worker.install_requirements_for_package_api(
        execution=execution,
        repo_info=repo_info,
        requirements_manager=requirements_manager,
        cached_requirements=cached_requirements,
        cwd=tmp_path.as_posix(),
        package_api=package_api,
    )

    worker._install_patched_python_requirements.assert_called_once_with(
        execution, package_api, cached_requirements
    )
    package_api.install_from_file.assert_not_called()
