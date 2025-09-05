import pytest


pytest_plugins = ["test_kit.fixtures"]


@pytest.fixture
def anyio_backend():
    # Force AnyIO tests to run on asyncio so you don't need Trio installed.
    return "asyncio"
