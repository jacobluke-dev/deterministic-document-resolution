import pytest
from document_resolution_core.core.domain import DefinitionCandidate
from document_resolution_core.core.services.resolver import AcronymResolver

# It’s a shared fixture module and must be loaded locally because pytest rootdir changes under `make -C`.
from test_kit.fixtures import (  # noqa: F401
    TEST_DB_URL,
    apply_migrations_once,
    db_ready,
    dbm,
    engine_factory,
    session_factory,
)


@pytest.fixture
def mock_lookup():
    """
    Fixture to mock the lookup function.
    """
    def _lookup(query):
        return [
            DefinitionCandidate(text="Definition 1", score=10.0),
            DefinitionCandidate(text="Definition 2", score=5.0),
            DefinitionCandidate(text="Definition 3", score=8.0),
        ]
    return _lookup


@pytest.fixture
def acronym_resolver(mock_lookup):
    """
    Fixture for AcronymResolver with mocked lookup.
    """
    return AcronymResolver(mock_lookup)
