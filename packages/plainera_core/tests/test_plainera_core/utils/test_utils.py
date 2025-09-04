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


@pytest.mark.unit
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


@pytest.mark.unit
class TestGetProjectRoot:
    def test_returns_expected_path_from_fake_file(self, monkeypatch):
        # Suppose our module file is at /a/b/c/src/utils/foo/bar/utils.py
        fake_file = os.path.join(
            os.sep, "a", "b", "c", "src", "utils", "foo", "bar", "utils.py"
        )
        monkeypatch.setattr(utils, "__file__", fake_file)

        root = utils.get_project_root()

        # 5 levels up from fake_file
        expected = os.path.abspath(os.path.join(fake_file, "..", "..", "..", "..", ".."))
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
