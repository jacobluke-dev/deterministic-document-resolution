from __future__ import annotations

import pytest
from plainera_unacronym.orchestration.interface import PipelineRunResult
from plainera_unacronym.orchestration.state import (
    OrchestrationPipelineError,
    OrchestrationState,
)


class TestOrchestrationState:
    def test_initialises_with_requested_targets(self):
        state = OrchestrationState.from_requested_targets(
            ("acronyms", "defined_terms", "structural_references")
        )

        assert state.requested_targets == (
            "acronyms",
            "defined_terms",
            "structural_references",
        )
        assert state.completed_targets == ()
        assert state.failed_targets == ()
        assert state.results_by_pipeline == {}
        assert state.errors_by_pipeline == {}
        assert state.metadata.finished_at_monotonic is None
        assert state.metadata.duration_ms is None

    def test_record_success_stores_result_and_marks_pipeline_completed(self):
        state = OrchestrationState.from_requested_targets(("acronyms",))
        result = PipelineRunResult(
            pipeline="acronyms",
            payload={"items": []},
            metadata={"source": "test"},
        )

        state.record_success(result)

        assert state.completed_targets == ("acronyms",)
        assert state.failed_targets == ()
        assert state.results_by_pipeline == {"acronyms": result}
        assert state.errors_by_pipeline == {}

    def test_record_failure_stores_error_and_marks_pipeline_failed(self):
        state = OrchestrationState.from_requested_targets(("defined_terms",))
        error = OrchestrationPipelineError(
            pipeline="defined_terms",
            error_type="RuntimeError",
            message="boom",
        )

        state.record_failure(error)

        assert state.completed_targets == ()
        assert state.failed_targets == ("defined_terms",)
        assert state.results_by_pipeline == {}
        assert state.errors_by_pipeline == {"defined_terms": error}

    def test_success_and_failure_are_tracked_independently(self):
        state = OrchestrationState.from_requested_targets(("acronyms", "defined_terms"))

        state.record_success(
            PipelineRunResult(
                pipeline="acronyms",
                payload={"items": ["PDF"]},
                metadata={},
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="defined_terms",
                error_type="ValueError",
                message="invalid config",
            )
        )

        assert state.completed_targets == ("acronyms",)
        assert state.failed_targets == ("defined_terms",)
        assert set(state.results_by_pipeline) == {"acronyms"}
        assert set(state.errors_by_pipeline) == {"defined_terms"}

    def test_record_success_rejects_unrequested_pipeline(self):
        state = OrchestrationState.from_requested_targets(("acronyms",))

        with pytest.raises(ValueError, match="unrequested pipeline"):
            state.record_success(
                PipelineRunResult(
                    pipeline="defined_terms",
                    payload={},
                    metadata={},
                )
            )

    def test_record_failure_rejects_duplicate_failure_for_same_pipeline(self):
        state = OrchestrationState.from_requested_targets(("acronyms",))
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="acronyms",
                error_type="RuntimeError",
                message="boom",
            )
        )

        with pytest.raises(ValueError, match="already recorded as failed"):
            state.record_failure(
                OrchestrationPipelineError(
                    pipeline="acronyms",
                    error_type="RuntimeError",
                    message="boom again",
                )
            )

    def test_finish_sets_finished_metadata(self):
        state = OrchestrationState.from_requested_targets(("acronyms",))

        state.finish()

        assert state.metadata.finished_at_monotonic is not None
        assert state.metadata.duration_ms is not None
        assert state.metadata.duration_ms >= 0
