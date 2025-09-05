import pytest
from plainera_core.core.domain import DefinitionCandidate
from plainera_core.core.services.resolver import AcronymResolver

pytest_plugins = (
    "test_kit.fixtures",
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
