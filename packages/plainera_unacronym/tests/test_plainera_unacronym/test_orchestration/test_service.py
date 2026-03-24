import pytest
from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    OrchestrationRequest,
    PipelineRequest,
    PipelineRunner,
    PipelineRunResult,
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


class _FailingRunner(PipelineRunner):
    def __init__(
        self,
        key: str,
        *,
        seen: list[tuple[str, PipelineRequest]],
        message: str = "boom",
    ) -> None:
        self.key = key
        self._seen = seen
        self._message = message

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        self._seen.append((self.key, request))
        raise RuntimeError(self._message)

class TestRunSelectedPipelines:
    @pytest.mark.anyio
    async def test_runs_requested_subset_and_records_completed_targets(self) -> None:
        seen: list[tuple[str, PipelineRequest]] = []
        registry = PipelineRegistry()
        registry.register(_StubRunner(PIPELINE_ACRONYMS, seen=seen))
        registry.register(_StubRunner(PIPELINE_DEFINED_TERMS, seen=seen))
        registry.register(_StubRunner(PIPELINE_STRUCTURAL_REFERENCES, seen=seen))

        request = OrchestrationRequest(
            text="Example text",
            targets=(PIPELINE_ACRONYMS, PIPELINE_STRUCTURAL_REFERENCES),
        )

        state = await run_selected_pipelines(registry, request)

        assert state.requested_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.completed_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.failed_targets == ()
        assert tuple(state.results_by_pipeline) == (
            PIPELINE_ACRONYMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.errors_by_pipeline == {}
        assert [key for key, _ in seen] == [
            PIPELINE_ACRONYMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        ]

    @pytest.mark.anyio
    async def test_records_results_in_registry_order_not_completion_order(self) -> None:
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

        state = await run_selected_pipelines(registry, request)

        assert state.requested_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.completed_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.failed_targets == ()
        assert tuple(state.results_by_pipeline) == (
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )

    @pytest.mark.anyio
    async def test_passes_per_pipeline_options_into_pipeline_requests(self) -> None:
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

        state = await run_selected_pipelines(registry, request)

        assert state.completed_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
        )
        assert state.failed_targets == ()
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

    @pytest.mark.anyio
    async def test_records_mixed_success_and_failure_in_registry_order(self) -> None:
        seen: list[tuple[str, PipelineRequest]] = []
        registry = PipelineRegistry()
        registry.register(_StubRunner(PIPELINE_ACRONYMS, seen=seen))
        registry.register(_FailingRunner(PIPELINE_DEFINED_TERMS, seen=seen))
        registry.register(_StubRunner(PIPELINE_STRUCTURAL_REFERENCES, seen=seen))

        request = OrchestrationRequest(
            text="Example text",
            targets=(
                PIPELINE_STRUCTURAL_REFERENCES,
                PIPELINE_DEFINED_TERMS,
                PIPELINE_ACRONYMS,
            ),
        )

        state = await run_selected_pipelines(registry, request)

        assert state.requested_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.completed_targets == (
            PIPELINE_ACRONYMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert state.failed_targets == (PIPELINE_DEFINED_TERMS,)

        assert tuple(state.results_by_pipeline) == (
            PIPELINE_ACRONYMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        )
        assert tuple(state.errors_by_pipeline) == (PIPELINE_DEFINED_TERMS,)

        error = state.errors_by_pipeline[PIPELINE_DEFINED_TERMS]
        assert error.pipeline == PIPELINE_DEFINED_TERMS
        assert error.error_type == "RuntimeError"
        assert error.message == "boom"

        assert [key for key, _ in seen] == [
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        ]
