import pytest


pytest_plugins = ["test_kit.fixtures"]

from test_kit.fixtures import (  # noqa: F401
    TEST_DB_URL,
    apply_migrations_once,
    db_ready,
    dbm,
    engine_factory,
    session_factory,
)


@pytest.fixture
def anyio_backend():
    # Force AnyIO tests to run on asyncio so we don't need Trio installed.
    return "asyncio"
