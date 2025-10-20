import pytest


pytest_plugins = ["test_kit.fixtures"]


@pytest.fixture
def anyio_backend():
    # Force AnyIO tests to run on asyncio so we don't need Trio installed.
    return "asyncio"
