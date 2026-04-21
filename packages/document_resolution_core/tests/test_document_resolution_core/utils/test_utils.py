from pathlib import Path
from unittest import mock

import document_resolution_core.utils.utils as utils
import pytest
from document_resolution_core.utils.utils import (
    find_project_root,
    get_environment,
    is_integration_env,
    is_local_env,
    is_test_env,
    is_valid_environment,
)


class TestLoadEnvIfLocal:
    @pytest.mark.parametrize("env_value", ["LOCAL", "LOCAL_PROD", "local", "local_prod"])
    def test_calls_load_dotenv_for_local_variants(self, monkeypatch, env_value):
        monkeypatch.setenv("ENVIRONMENT", env_value)
        m = mock.Mock()
        # Patch the symbol in the defining module
        monkeypatch.setattr(utils, "load_dotenv", m)

        rv = utils.load_env_if_local()

        assert rv is None
        m.assert_called_once_with()

    @pytest.mark.parametrize("env_value", [None, "", "TEST", "DEV", "PROD", "STAGING"])
    def test_does_not_call_load_dotenv_for_other_envs(self, monkeypatch, env_value):
        if env_value is None:
            monkeypatch.delenv("ENVIRONMENT", raising=False)
        else:
            monkeypatch.setenv("ENVIRONMENT", env_value)

        m = mock.Mock()
        monkeypatch.setattr(utils, "load_dotenv", m)

        rv = utils.load_env_if_local()

        assert rv is None
        m.assert_not_called()


class TestEnvironmentFuncs:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("LOCAL", True),
            ("INTEGRATION", True),
            ("PROD", True),
            ("LOCAL_PROD", True),
            ("TEST", True),
            ("STAGING", True),
            ("UNKNOWN", False),
            ("", False),
            (None, False),
        ],
    )
    def test_is_valid_environment(self, value, expected):
        assert is_valid_environment(value) == expected

    def test_get_environment_valid(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "LOCAL")
        assert get_environment() == "LOCAL"

    def test_get_environment_invalid_raises(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)  # no variable set
        with pytest.raises(ValueError):
            get_environment()

        monkeypatch.setenv("ENVIRONMENT", "INVALID")
        with pytest.raises(ValueError):
            get_environment()

    @pytest.mark.parametrize(
        "env_value, func, expected",
        [
            ("LOCAL", is_local_env, True),
            ("TEST", is_test_env, True),
            ("INTEGRATION", is_integration_env, True),
            ("PROD", is_local_env, False),
            ("STAGING", is_test_env, False),
        ],
    )
    def test_environment_helpers(self, monkeypatch, env_value, func, expected):
        monkeypatch.setenv("ENVIRONMENT", env_value)
        assert func() is expected

class TestFindProjectRoot:

    def test_returns_nearest_ancestor_with_marker_file(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        utils_dir = repo / "src" / "utils"

        utils_dir.mkdir(parents=True)
        (repo / ".document-resolution-root").touch()

        root = find_project_root(start=utils_dir)

        assert root == repo

    def test_accepts_string_start_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        utils_dir = repo / "src" / "utils"

        utils_dir.mkdir(parents=True)
        (repo / ".document-resolution-root").touch()

        root = find_project_root(start=str(utils_dir))

        assert root == repo

    def test_uses_parent_when_start_is_file(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        file_path = repo / "src" / "utils" / "thing.py"

        file_path.parent.mkdir(parents=True)
        file_path.touch()
        (repo / ".document-resolution-root").touch()

        root = find_project_root(start=file_path)

        assert root == repo

    def test_returns_nearest_marker_not_topmost(self, tmp_path: Path) -> None:
        top = tmp_path / "top"
        sub = top / "subproj"
        deep = sub / "src" / "utils"

        deep.mkdir(parents=True)
        (top / ".document-resolution-root").touch()
        (sub / ".document-resolution-root").touch()

        root = find_project_root(start=deep)

        assert root == sub

    def test_raises_when_no_marker_is_found(self, tmp_path: Path, monkeypatch) -> None:
        leaf = tmp_path / "a" / "b" / "c"
        leaf.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Could not find"):
            find_project_root(start=leaf)

        with pytest.raises(FileNotFoundError, match="Could not find"):
            find_project_root(start=str(leaf))

        monkeypatch.chdir(leaf)
        with pytest.raises(FileNotFoundError, match="Could not find"):
            find_project_root()
