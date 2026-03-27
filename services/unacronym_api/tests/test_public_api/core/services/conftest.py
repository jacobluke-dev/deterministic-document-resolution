from unittest.mock import Mock

import pytest

from public_api.core.services.resolve_service import ResolveService
from public_api.schemas.resolve import ResolveOptions


@pytest.fixture
def service_factory():
    def make(*, meanings: list[dict] | None = None, semaphore=None) -> tuple[ResolveService, Mock]:
        repo = Mock()
        repo.list_meanings.return_value = meanings or []
        from plainera_unacronym.orchestration import PipelineRegistry
        svc = ResolveService(
            glossary_repo=repo,
            semaphore=semaphore,
            request_timeout_ms=1000,
            tier2_model=None,
            pipeline_registry=PipelineRegistry()
        )
        return svc, repo

    return make


@pytest.fixture
def opts_factory():
    def make(**overrides) -> ResolveOptions:
        return ResolveOptions.model_validate(overrides)

    return make

class DummyGlossaryRepo:
    def __init__(self, meanings_by_acronym):
        self._meanings_by_acronym = meanings_by_acronym

    def list_meanings(self, *, acronym: str):
        return self._meanings_by_acronym.get(acronym, [])
