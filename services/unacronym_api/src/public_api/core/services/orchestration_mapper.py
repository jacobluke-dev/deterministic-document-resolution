from __future__ import annotations

from plainera_unacronym.orchestration.state import OrchestrationState

from public_api.schemas.resolve import OrchestrationMeta, PipelineError


def map_orchestration_state(
    state: OrchestrationState,
) -> tuple[OrchestrationMeta, list[PipelineError]]:
    """Map internal orchestration state to public metadata and pipeline errors."""
    meta = OrchestrationMeta(
        requested=list(state.requested_targets),
        completed=list(state.completed_targets),
        failed=list(state.failed_targets),
    )

    errors = [
        PipelineError(
            pipeline=pipeline,
            code=state.errors_by_pipeline[pipeline].code,
            message=state.errors_by_pipeline[pipeline].message,
        )
        for pipeline in state.failed_targets
    ]

    return meta, errors
