import os
from unittest import mock

import plainera_core.utils.utils as utils
import pytest
from plainera_core.utils.utils import (
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


class TestGetProjectRoot:
    def test_returns_expected_path_from_fake_file(self, monkeypatch):

        fake_file = os.path.join(
            os.sep, "a", "b", "c", "src", "utils", "foo", "bar", "utils.py"
        )
        monkeypatch.setattr(utils, "__file__", fake_file)

        root = utils.get_project_root()

        # 6 levels up from fake_file
        expected = os.path.abspath(os.path.join(fake_file, "..", "..", "..", "..", "..", ".."))
        assert root == expected
        # should be an absolute path
        assert os.path.isabs(root)


class TestGetProjectPath:
    @pytest.fixture
    def mock_project_root(self):
        with mock.patch('plainera_core.utils.utils.get_project_root') as mock_root:
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
