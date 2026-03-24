from __future__ import annotations

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    OrchestrationRequest,
    PipelineKey,
)

from public_api.schemas.resolve import ResolveOptions, ResolveRequest


def build_orchestration_request(
    payload: ResolveRequest,
    *,
    targets: tuple[PipelineKey, ...],
    tier2_model: object | None,
) -> OrchestrationRequest:
    pipeline_options: dict[PipelineKey, dict[str, object]] = {}

    if PIPELINE_ACRONYMS in targets:
        pipeline_options[PIPELINE_ACRONYMS] = _build_acronym_options(
            payload,
            tier2_model=tier2_model,
        )

    if PIPELINE_DEFINED_TERMS in targets:
        pipeline_options[PIPELINE_DEFINED_TERMS] = _build_defined_term_options(payload)

    if PIPELINE_STRUCTURAL_REFERENCES in targets:
        pipeline_options[PIPELINE_STRUCTURAL_REFERENCES] = _build_structural_reference_options(payload)

    return OrchestrationRequest(
        text=payload.text,
        targets=targets,
        pipeline_options=pipeline_options,
    )


def _build_acronym_options(
    payload: ResolveRequest,
    *,
    tier2_model: object | None,
) -> dict[str, object]:
    opts = payload.options or ResolveOptions.model_validate({})
    return {
        "window_left": int(opts.window_chars),
        "window_right": int(opts.window_chars),
        "trace": False,
        "return_reports": False,
        "return_state": False,
        "det_cfg": None,
        "ext_cfg": None,
        "tier2_model": tier2_model,
    }


def _build_defined_term_options(payload: ResolveRequest) -> dict[str, object]:
    return {}


def _build_structural_reference_options(payload: ResolveRequest) -> dict[str, object]:
    return {}
