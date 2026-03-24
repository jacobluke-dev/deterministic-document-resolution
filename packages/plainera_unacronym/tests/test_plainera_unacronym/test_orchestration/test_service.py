import anyio
import pytest

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    OrchestrationRequest,
    PipelineRequest,
    PipelineRunResult,
    PipelineRunner,
)
from plainera_unacronym.orchestration.registry import PipelineRegistry
from plainera_unacronym.orchestration.service import run_selected_pipelines


class _StubRunner(PipelineRunner):
    def __init__(
        self,
        key: str,
        *,
        seen: list[tuple[str, PipelineRequest]],
        delay: float = 0.0,
    ) -> None:
        self.key = key
        self._seen = seen
        self._delay = delay

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        if self._delay:
            import time

            time.sleep(self._delay)

        self._seen.append((self.key, request))
        return PipelineRunResult(
            pipeline=self.key,
            payload={"text": request.text, "options": dict(request.options)},
        )


@pytest.mark.anyio
async def test_run_selected_pipelines_runs_requested_subset() -> None:
    seen: list[tuple[str, PipelineRequest]] = []
    registry = PipelineRegistry()
    registry.register(_StubRunner(PIPELINE_ACRONYMS, seen=seen))
    registry.register(_StubRunner(PIPELINE_DEFINED_TERMS, seen=seen))
    registry.register(_StubRunner(PIPELINE_STRUCTURAL_REFERENCES, seen=seen))

    request = OrchestrationRequest(
        text="Example text",
        targets=(PIPELINE_ACRONYMS, PIPELINE_STRUCTURAL_REFERENCES),
    )

    results = await run_selected_pipelines(registry, request)

    assert tuple(result.pipeline for result in results) == (
        PIPELINE_ACRONYMS,
        PIPELINE_STRUCTURAL_REFERENCES,
    )
    assert [key for key, _ in seen] == (
        [PIPELINE_ACRONYMS, PIPELINE_STRUCTURAL_REFERENCES]
    )


@pytest.mark.anyio
async def test_run_selected_pipelines_returns_results_in_registry_order() -> None:
    seen: list[tuple[str, PipelineRequest]] = []
    registry = PipelineRegistry()
    registry.register(
        _StubRunner(
            PIPELINE_ACRONYMS,
            seen=seen,
            delay=0.2,
        )
    )
    registry.register(
        _StubRunner(
            PIPELINE_DEFINED_TERMS,
            seen=seen,
            delay=0.0,
        )
    )
    registry.register(
        _StubRunner(
            PIPELINE_STRUCTURAL_REFERENCES,
            seen=seen,
            delay=0.1,
        )
    )

    request = OrchestrationRequest(
        text="Example text",
        targets=(
            PIPELINE_STRUCTURAL_REFERENCES,
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
        ),
    )

    results = await run_selected_pipelines(registry, request)

    assert tuple(result.pipeline for result in results) == (
        PIPELINE_ACRONYMS,
        PIPELINE_DEFINED_TERMS,
        PIPELINE_STRUCTURAL_REFERENCES,
    )


@pytest.mark.anyio
async def test_run_selected_pipelines_passes_per_pipeline_options() -> None:
    seen: list[tuple[str, PipelineRequest]] = []
    registry = PipelineRegistry()
    registry.register(_StubRunner(PIPELINE_ACRONYMS, seen=seen))
    registry.register(_StubRunner(PIPELINE_DEFINED_TERMS, seen=seen))

    request = OrchestrationRequest(
        text="Example text",
        targets=(PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS),
        pipeline_options={
            PIPELINE_ACRONYMS: {
                "window_left": 400,
                "trace": True,
            },
            PIPELINE_DEFINED_TERMS: {
                "disambig_margin_threshold": 0.2,
            },
        },
    )

    results = await run_selected_pipelines(registry, request)

    assert tuple(result.pipeline for result in results) == (
        PIPELINE_ACRONYMS,
        PIPELINE_DEFINED_TERMS,
    )
    assert seen[0][0] == PIPELINE_ACRONYMS
    assert seen[0][1] == PipelineRequest(
        text="Example text",
        options={"window_left": 400, "trace": True},
    )
    assert seen[1][0] == PIPELINE_DEFINED_TERMS
    assert seen[1][1] == PipelineRequest(
        text="Example text",
        options={"disambig_margin_threshold": 0.2},
    )
