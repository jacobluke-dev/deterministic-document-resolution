from __future__ import annotations

from typing import Any

from plainera_unacronym.orchestration import PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS, PIPELINE_STRUCTURAL_REFERENCES
from plainera_unacronym.orchestration.state import OrchestrationState
from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import map_pipeline_to_blocks
from public_api.db.repos import GlossaryRepository

from public_api.schemas.resolve import OrchestrationMeta, PipelineError, ResolveOptions, ResolutionMode


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


def compose_sections(
    state: OrchestrationState,
    *,
    opts: ResolveOptions,
    lang: str,
    resolution_mode: ResolutionMode,
    glossary_repo: GlossaryRepository,
) -> dict[str, Any]:
    sections: dict[str, Any] = {
        "acronyms": [],
        "defined_terms": [],
        "structural_references": [],
    }

    if PIPELINE_ACRONYMS in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_ACRONYMS].payload

        if isinstance(payload, list):
            sections["acronyms"] = payload
        elif isinstance(payload, tuple) and len(payload) == 2:
            det_res, extr = payload
            blocks = map_pipeline_to_blocks(
                det_res=det_res,
                extr=extr,
                opts=opts,
                lang=lang,
                glossary_repo=glossary_repo,
            )
            sections["acronyms"] = attach_resolution_metadata(
                blocks=blocks,
                opts=opts,
                resolution_mode=resolution_mode,
                glossary_repo=glossary_repo,
            )
        else:
            raise ValueError("Unsupported acronym pipeline payload shape.")

    if PIPELINE_DEFINED_TERMS in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_DEFINED_TERMS].payload
        if isinstance(payload, list):
            sections["defined_terms"] = payload

    if PIPELINE_STRUCTURAL_REFERENCES in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_STRUCTURAL_REFERENCES].payload
        if isinstance(payload, list):
            sections["structural_references"] = payload

    return sections
