from pathlib import Path
from unittest import mock

import plainera_core.utils.utils as utils
import pytest
from plainera_core.utils.utils import (
    find_project_root,
    get_environment,
    get_project_path,
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


@pytest.fixture(autouse=True)
def _clear_cache():
    # Ensure LRU cache doesn't leak between tests
    find_project_root.cache_clear()
    yield
    find_project_root.cache_clear()


class TestFindProjectRoot:

    def test_returns_repo_root_when_git_present(self, tmp_path: Path):
        """
        Start inside src/utils; repo root has .git → should return repo root.
        """
        repo = tmp_path / "repo"
        git = repo / ".git"
        utils = repo / "src" / "utils"
        git.mkdir(parents=True)
        utils.mkdir(parents=True)

        root = find_project_root(start=utils)
        assert root == repo

    def test_biases_to_git_when_multiple_markers(self, tmp_path: Path):
        """
        If both a lower pyproject and an upper .git exist, prefer the .git
        root.
        """
        repo = tmp_path / "repo"
        pkg = repo / "packages" / "pkg"
        utils = pkg / "src" / "utils"

        # markers
        (repo / ".git").mkdir(parents=True)
        utils.mkdir(parents=True)
        (pkg / "pyproject.toml").touch()

        root = find_project_root(start=utils)
        assert root == repo  # prefer repo due to .git

    def test_top_most_when_no_git(self, tmp_path: Path):
        """
        With multiple pyproject markers but no .git, choose the top-most
        candidate.
        """
        top = tmp_path / "top"
        sub = top / "subproj"
        deep = sub / "src" / "utils"
        deep.mkdir(parents=True)
        (top / "pyproject.toml").touch()
        (sub / "pyproject.toml").touch()

        root = find_project_root(start=deep)
        assert root == top  # top-most candidate wins

    def test_returns_start_when_no_markers(self, tmp_path: Path, monkeypatch):
        """
        If no markers found, return the resolved start (or CWD if start=None).
        """
        leaf = tmp_path / "a" / "b" / "c"
        leaf.mkdir(parents=True)

        # Explicit start (Path)
        root = find_project_root(start=leaf)
        assert root == leaf.resolve()

        # Explicit start (str)
        root2 = find_project_root(start=str(leaf))
        assert root2 == leaf.resolve()

        # start=None → uses CWD
        monkeypatch.chdir(leaf)
        root3 = find_project_root()
        assert root3 == leaf.resolve()

    def test_handles_marker_tuple_override(self, tmp_path: Path):
        """
        Verify custom markers tuple is respected.
        """
        repo = tmp_path / "repo"
        utils = repo / "src" / "utils"
        utils.mkdir(parents=True)
        # Only write a .gitlab-ci.yml marker
        (repo / ".gitlab-ci.yml").touch()

        root = find_project_root(start=utils, markers=("pyproject.toml", ".git", ".gitlab-ci.yml"))
        assert root == repo


class TestGetProjectPath:
    @pytest.fixture
    def mock_project_root(self):
        with mock.patch('plainera_core.utils.utils.find_project_root') as mock_root:
            mock_root.return_value = '/home/user/project'
            yield mock_root

    def test_get_project_path_valid(self, mock_project_root):
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            relative_path = 'src/utils/file.txt'
            expected_path = '/home/user/project/src/utils/file.txt'

            result = get_project_path(relative_path)

            assert result == expected_path
            mock_project_root.assert_called_once()
            mock_exists.assert_called_once_with(expected_path)

    def test_get_project_path_invalid_raise_error(self, mock_project_root):
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            relative_path = 'src/utils/non_existent_file.txt'
            expected_path = '/home/user/project/src/utils/non_existent_file.txt'

            with pytest.raises(FileNotFoundError):
                get_project_path(relative_path)

            mock_project_root.assert_called_once()
            mock_exists.assert_called_once_with(expected_path)

    def test_get_project_path_invalid_no_raise_error(self, mock_project_root):
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            relative_path = 'src/utils/non_existent_file.txt'
            expected_path = '/home/user/project/src/utils/non_existent_file.txt'

            result = get_project_path(relative_path, raise_error=False)

            assert result is False
            mock_project_root.assert_called_once()
            mock_exists.assert_called_once_with(expected_path)

    def test_get_project_path_missing_return_path_true(self, mock_project_root):
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            relative_path = 'src/utils/non_existent_file.txt'
            expected_path = '/home/user/project/src/utils/non_existent_file.txt'

            # return_path=True should return the computed absolute path (no exception)
            result = get_project_path(relative_path, return_path=True)
            assert result == expected_path

            mock_project_root.assert_called_once()
            mock_exists.assert_called_once_with(expected_path)

    def test_get_project_path_missing_return_path_true_overrides_raise(self, mock_project_root):
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            relative_path = 'src/utils/non_existent_file.txt'
            expected_path = '/home/user/project/src/utils/non_existent_file.txt'

            # Even with raise_error=True, return_path=True should still return the path
            result = get_project_path(relative_path, raise_error=True, return_path=True)
            assert result == expected_path
