import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, overload

from dotenv import load_dotenv


def load_env_if_local() -> None:
    if (os.getenv("ENVIRONMENT") or "").upper() in {"LOCAL", "LOCAL_PROD"}:
        load_dotenv()

load_env_if_local()

def is_valid_environment(env_value: Optional[str]) -> bool:
    """Validate that the environment value is one of the allowed values.

    Args:
        env_value (str): The environment value to validate.
    Returns:
        bool: True if the environment value is one of the allowed values.
    """
    allowed_environments = {'LOCAL', 'INTEGRATION', 'PROD', 'LOCAL_PROD', 'TEST', 'STAGING'}
    return env_value in allowed_environments


def get_environment() -> str:
    """Get and validate the current environment setting.

    Retrieves the value of the ``ENVIRONMENT`` environment variable and
    validates it using ``is_valid_environment``. If the value is invalid,
    a ``ValueError`` is raised.

    Returns:
        str: The validated environment value.

    Raises:
        ValueError: If the ``ENVIRONMENT`` variable is missing or invalid.
    """
    env_value = os.getenv('ENVIRONMENT')
    if env_value is None or not is_valid_environment(env_value):
        raise ValueError(f"Invalid environment value: {env_value}")
    return env_value



def is_local_env() -> bool:
    """Checks if the environment is local or not.

    Returns:
         bool: True if it is local or not.
    """
    return get_environment() == 'LOCAL'


def is_test_env() -> bool:
    """Checks if the environment is test env or not.

    Returns:
         bool: True if it is test env or not.
    """
    return get_environment() == 'TEST'

def is_integration_env() -> bool:
    """Checks if the environment is INTEGRATION env or not.

    Returns:
         bool: True if it is INTEGRATION env or not.
    """
    return get_environment() == 'INTEGRATION'


@lru_cache(maxsize=1)
def find_project_root(start: str | Path | None = None,
                      markers: tuple[str, ...] = ("pyproject.toml", ".git", ".gitlab-ci.yml")) -> Path:
    """Returns the project root directory.

    This function determines the project root directory by assuming the file is
    located in the 'src/utils/' directory and then traversing up the directory
    structure.

    Returns:
        str: The absolute path to the project root directory.
    """
    p = Path(start).resolve() if start else Path.cwd().resolve()
    parents = [p, *p.parents]
    candidates: list[Path] = []
    for q in parents:
        if any((q / m).exists() for m in markers):
            candidates.append(q)

    if not candidates:
        return p  # fallback: no markers found

    # Prefer the top-most (repo root). If multiple, bias to the one that has .git.
    top = candidates[-1]
    for q in reversed(candidates):
        if (q / ".git").exists():
            return q
    return top



@overload
def get_project_path(
    relative_path: str, *, raise_error: Literal[True] = ..., return_path: bool = ...
) -> str: ...
    # no code, no docstring — just a signature

@overload
def get_project_path(
    relative_path: str, *, raise_error: Literal[False], return_path: Literal[True]
) -> str: ...
    # this says: if call with raise_error=False and return_path=True,
    # also always get a str.

@overload
def get_project_path(
    relative_path: str, *, raise_error: Literal[False] = ..., return_path: Literal[False] = ...
) -> str | bool: ...
    # the “other cases” — could be str or bool


def get_project_path(
    relative_path: str, *, raise_error: bool = True, return_path: bool = False
) -> str | bool:
    """Returns the absolute path relative to the project root directory.

    This function combines the project root directory with a relative path
    to provide the absolute path to any file within the project structure.

    Args:
        relative_path (str): The relative path from the project root to the target file.
        raise_error (bool): If True (default), raise FileNotFoundError if not found.
        return_path (bool): If True, return the computed path even if missing.
    Returns:
        str | bool: The absolute path to the target file, or False if it does not exist
        (and return_path is False).
    Raises:
        FileNotFoundError: If the path does not exist and raise_error=True.
    """
    project_root = find_project_root()
    absolute_path = os.path.join(project_root, relative_path)
    if not os.path.exists(absolute_path):
        if return_path:
            return absolute_path
        elif raise_error:
            raise FileNotFoundError(f"File '{absolute_path}' does not exist.")
        else:
            return False
    return absolute_path
