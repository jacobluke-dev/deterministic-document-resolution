from plainera_unacronym.orchestration.interface import PipelineRunResult
from plainera_unacronym.orchestration.state import (
    OrchestrationPipelineError,
    OrchestrationState,
)
from public_api.core.orchestration.mapper import map_orchestration_state


class TestMapOrchestrationState:
    def test_maps_all_success_metadata_and_no_errors(self) -> None:
        state = OrchestrationState.from_requested_targets(("acronyms", "defined_terms"))
        state.record_success(PipelineRunResult(pipeline="acronyms", payload={}, metadata={}))
        state.record_success(PipelineRunResult(pipeline="defined_terms", payload={}, metadata={}))
        state.finish()

        meta, errors = map_orchestration_state(state)

        assert meta.requested == ["acronyms", "defined_terms"]
        assert meta.completed == ["acronyms", "defined_terms"]
        assert meta.failed == []
        assert errors == []

    def test_maps_partial_success_metadata_and_errors_in_failed_order(self) -> None:
        state = OrchestrationState.from_requested_targets(
            ("acronyms", "defined_terms", "structural_references")
        )
        state.record_success(PipelineRunResult(pipeline="acronyms", payload={}, metadata={}))
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="defined_terms",
                code="PIPELINE_EXECUTION_FAILED",
                message="boom",
                error_type="RuntimeError",
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="structural_references",
                code="PIPELINE_TIMEOUT",
                message="timed out",
                error_type="TimeoutError",
            )
        )
        state.finish()

        meta, errors = map_orchestration_state(state)

        assert meta.requested == ["acronyms", "defined_terms", "structural_references"]
        assert meta.completed == ["acronyms"]
        assert meta.failed == ["defined_terms", "structural_references"]
        assert [error.pipeline for error in errors] == [
            "defined_terms",
            "structural_references",
        ]
        assert [error.code for error in errors] == [
            "PIPELINE_EXECUTION_FAILED",
            "PIPELINE_TIMEOUT",
        ]

    def test_maps_all_failure_metadata(self) -> None:
        state = OrchestrationState.from_requested_targets(("acronyms", "defined_terms"))
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="acronyms",
                code="PIPELINE_EXECUTION_FAILED",
                message="boom",
                error_type="RuntimeError",
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="defined_terms",
                code="PIPELINE_INVALID_OPTIONS",
                message="bad config",
                error_type="ValueError",
            )
        )
        state.finish()

        meta, errors = map_orchestration_state(state)

        assert meta.requested == ["acronyms", "defined_terms"]
        assert meta.completed == []
        assert meta.failed == ["acronyms", "defined_terms"]
        assert [error.code for error in errors] == [
            "PIPELINE_EXECUTION_FAILED",
            "PIPELINE_INVALID_OPTIONS",
        ]
