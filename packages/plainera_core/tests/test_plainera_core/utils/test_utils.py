from unittest import mock

import pytest

from utils.utils import get_project_path


class TestGetProjectPath:
    @pytest.fixture
    def mock_project_root(self):
        with mock.patch('utils.utils.get_project_root') as mock_root:
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
