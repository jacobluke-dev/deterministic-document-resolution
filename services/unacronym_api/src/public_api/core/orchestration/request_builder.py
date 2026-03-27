from __future__ import annotations

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    OrchestrationRequest,
    PipelineKey,
)

from public_api.core.settings import app_settings
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


def _chunking_options(
    *,
    enabled: bool,
    threshold_chars: int,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> dict[str, object]:
    return {
        "chunking_enabled": bool(enabled),
        "chunk_threshold_chars": int(threshold_chars),
        "chunk_size_chars": int(chunk_size_chars),
        "chunk_overlap_chars": int(chunk_overlap_chars),
    }

def _base_pipeline_options() -> dict[str, object]:
    options: dict[str, object] = {
        "det_cfg": None,
        "ext_cfg": None,
        "return_reports": False,
        "return_state": False,
        "trace": False,
        "trace_filter": None
    }
    return options

def _default_chunking_options() -> dict[str, object]:
    return _chunking_options(
        enabled=app_settings.CHUNKING_ENABLED,
        threshold_chars=app_settings.CHUNK_THRESHOLD_CHARS,
        chunk_size_chars=app_settings.CHUNK_SIZE_CHARS,
        chunk_overlap_chars=app_settings.CHUNK_OVERLAP_CHARS,
    )

def _build_acronym_options(
    payload: ResolveRequest,
    *,
    tier2_model: object | None,
) -> dict[str, object]:
    opts = payload.options or ResolveOptions.model_validate({})
    return {
        **_base_pipeline_options(),
        "window_left": int(opts.window_chars),
        "window_right": int(opts.window_chars),
        "tier2_model": tier2_model,
        **_default_chunking_options(),
    }


def _build_defined_term_options(payload: ResolveRequest) -> dict[str, object]:
    return {
        **_base_pipeline_options(),
        **_default_chunking_options(),
    }

def _build_structural_reference_options(payload: ResolveRequest) -> dict[str, object]:
    return {
        **_base_pipeline_options(),
        **_default_chunking_options(),
    }
