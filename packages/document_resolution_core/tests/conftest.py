"""Fixture bridge for this test tree.

pytest_plugins = ["test_kit.fixtures"] -- should work but has no effect
from root It’s a shared fixture module and must be loaded locally
because pytest rootdir changes under `make -C`.
"""
from test_kit.fixtures import (  # noqa: F401
    TEST_DB_URL,
    apply_migrations_once,
    db_ready,
    dbm,
    engine_factory,
    session_factory,
)
