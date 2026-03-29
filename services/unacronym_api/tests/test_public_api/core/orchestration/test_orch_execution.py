from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineRunResult,
)
from plainera_unacronym.orchestration.state import PipelineErrorCode
from public_api.core.errors import ResolveError
from public_api.core.orchestration import Orchestrator
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolutionMode, ResolveOptions


class _FakeExecutor:
    def __init__(
        self,
        *,
        result: PipelineRunResult | None = None,
        exc: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._result = result
        self._exc = exc
        self._delay_s = delay_s

    async def execute(
        self,
        *,
        request: Any,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        if self._delay_s:
            await anyio.sleep(self._delay_s)

        if self._exc is not None:
            raise self._exc

        assert self._result is not None
        return self._result


@pytest.fixture
def resolve_options_factory():
    def make(**overrides) -> ResolveOptions:
        return ResolveOptions(
            locale="en-GB",
            window_chars=120,
            max_definitions_per_acronym=5,
            include_glossary_enrichment=True,
            return_occurrences=True,
            min_confidence=0.0,
            **overrides,
        )
    return make

def _make_request(*targets: str, partial_success: bool) -> Any:
    return SimpleNamespace(
        text="dummy text",
        targets=tuple(targets),
        execution_options=SimpleNamespace(partial_success=partial_success),
    )


def _make_orchestrator() -> Orchestrator:
    return Orchestrator(
        pipeline_registry=cast(Any, object()),
        glossary_repo=cast(Any, object()),
        request_timeout_ms=1_000,
        tier2_model=None,
    )


class TestOrchestrator:
    @pytest.mark.anyio
    async def test_execute_orchestration_request_records_successes_in_registry_order(self, monkeypatch,
                                                                                     resolve_options_factory):
        orchestrator = _make_orchestrator()
        monkeypatch.setattr(
            orchestrator,
            "_registry_order_targets",
            lambda _targets: (PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS),
        )

        orchestrator._executors = {
            PIPELINE_ACRONYMS: _FakeExecutor(
                result=PipelineRunResult(
                    pipeline=PIPELINE_ACRONYMS,
                    payload={"name": "acronyms"},
                ),
                delay_s=0.02,
            ),
            PIPELINE_DEFINED_TERMS: _FakeExecutor(
                result=PipelineRunResult(
                    pipeline=PIPELINE_DEFINED_TERMS,
                    payload={"name": "defined_terms"},
                ),
                delay_s=0.0,
            ),
        }

        state = await orchestrator.execute_orchestration_request(
            request=_make_request(
                PIPELINE_ACRONYMS,
                PIPELINE_DEFINED_TERMS,
                partial_success=True,
            ),
            opts=resolve_options_factory(),
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )

        assert state.requested_targets == (PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS)
        assert state.completed_targets == (PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS)
        assert state.failed_targets == ()
        assert list(state.results_by_pipeline) == [PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS]
        assert state.results_by_pipeline[PIPELINE_ACRONYMS].payload == {"name": "acronyms"}
        assert state.results_by_pipeline[PIPELINE_DEFINED_TERMS].payload == {"name": "defined_terms"}
        assert state.errors_by_pipeline == {}
        assert state.metadata.finished_at_monotonic is not None

    @pytest.mark.anyio
    async def test_execute_orchestration_request_records_failure_when_partial_success_enabled(self,
                                                                                              monkeypatch,
                                                                                              resolve_options_factory):
        orchestrator = _make_orchestrator()
        monkeypatch.setattr(
            orchestrator,
            "_registry_order_targets",
            lambda _targets: (PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS),
        )

        orchestrator._executors = {
            PIPELINE_ACRONYMS: _FakeExecutor(
                result=PipelineRunResult(
                    pipeline=PIPELINE_ACRONYMS,
                    payload={"name": "acronyms"},
                ),
            ),
            PIPELINE_DEFINED_TERMS: _FakeExecutor(exc=ValueError("bad options")),
        }

        state = await orchestrator.execute_orchestration_request(
            request=_make_request(
                PIPELINE_ACRONYMS,
                PIPELINE_DEFINED_TERMS,
                partial_success=True,
            ),
            opts=resolve_options_factory(),
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )

        assert state.completed_targets == (PIPELINE_ACRONYMS,)
        assert state.failed_targets == (PIPELINE_DEFINED_TERMS,)
        assert list(state.results_by_pipeline) == [PIPELINE_ACRONYMS]
        assert len(state.errors_by_pipeline) == 1
        print(state.errors_by_pipeline)
        assert state.errors_by_pipeline['defined_terms'].pipeline == PIPELINE_DEFINED_TERMS
        assert state.errors_by_pipeline['defined_terms'].code == PipelineErrorCode.PIPELINE_INVALID_OPTIONS
        assert state.errors_by_pipeline['defined_terms'].message == "bad options"
        assert state.metadata.finished_at_monotonic is not None

    @pytest.mark.anyio
    async def test_execute_orchestration_request_reraises_when_partial_success_disabled(self,
                                                                                        monkeypatch,
                                                                                        resolve_options_factory):
        orchestrator = _make_orchestrator()
        monkeypatch.setattr(
            orchestrator,
            "_registry_order_targets",
            lambda _targets: (PIPELINE_ACRONYMS,),
        )

        orchestrator._executors = {
            PIPELINE_ACRONYMS: _FakeExecutor(exc=RuntimeError("boom")),
        }

        with pytest.raises(ExceptionGroup) as exc:
            await orchestrator.execute_orchestration_request(
                request=_make_request(
                    PIPELINE_ACRONYMS,
                    partial_success=False,
                ),
                opts=resolve_options_factory(),
                lang="en",
                resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            )

        assert len(exc.value.exceptions) == 1
        assert isinstance(exc.value.exceptions[0], RuntimeError)
        assert str(exc.value.exceptions[0]) == "boom"

    @pytest.mark.parametrize(
        ("exc", "expected_code", "expected_message"),
        [
            (
            ResolveError(message="Resolution timed out.",
                         details={"timeout_ms": 1_000},
                         http_status=503,
                         code=ErrorCode.BAD_REQUEST),
                PipelineErrorCode.PIPELINE_TIMEOUT,
                "Resolution timed out.",
            ),
            (
            ResolveError(message="pipeline exploded",
                         details={"foo": "bar"},
                         http_status=500,
                         code=ErrorCode.BAD_REQUEST),
                PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                "pipeline exploded",
            ),
            (
                TimeoutError(),
                PipelineErrorCode.PIPELINE_TIMEOUT,
                "Pipeline execution failed.",
            ),
            (
                ValueError("bad options"),
                PipelineErrorCode.PIPELINE_INVALID_OPTIONS,
                "bad options",
            ),
            (
                RuntimeError("boom"),
                PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                "boom",
            ),
        ],
    )
    def test_map_pipeline_exception_maps_expected_codes(
        self,
        exc: Exception,
        expected_code: PipelineErrorCode,
        expected_message: str,
    ):
        error = Orchestrator._map_pipeline_exception(PIPELINE_STRUCTURAL_REFERENCES, exc)

        assert error.pipeline == PIPELINE_STRUCTURAL_REFERENCES
        assert error.code == expected_code
        assert error.message == expected_message
        assert error.error_type == type(exc).__name__

    @pytest.mark.anyio
    async def test_execute_pipeline_raises_for_unknown_pipeline(self, resolve_options_factory):
        orchestrator = _make_orchestrator()
        orchestrator._executors = {}

        with pytest.raises(ValueError, match="No executor configured"):
            await orchestrator._execute_pipeline(
                pipeline=PIPELINE_ACRONYMS,
                request=_make_request(PIPELINE_ACRONYMS, partial_success=True),
                opts=resolve_options_factory(),
                lang="en",
                resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            )
