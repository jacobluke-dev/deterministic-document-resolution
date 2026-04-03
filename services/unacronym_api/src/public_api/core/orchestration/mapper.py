from __future__ import annotations

from typing import Any

from plainera_unacronym.nlp.common.types import AcronymDetectorResult, AcronymPipelineResult, ExtractionResult
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult
from plainera_unacronym.nlp.extraction.structural.types import StructuralReferenceResolutionResult
from plainera_unacronym.orchestration import PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS, PIPELINE_STRUCTURAL_REFERENCES
from plainera_unacronym.orchestration.state import OrchestrationState

from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import (
    map_acronym_pipeline_to_blocks,
    map_defined_term_blocks,
    map_structural_blocks,
)
from public_api.db.repos import GlossaryRepository
from public_api.schemas.resolve import OrchestrationMeta, PipelineError, ResolutionMode, ResolveOptions


def _resolve_defined_term_payload(
    payload: object,
    *,
    error_message: str,
    allow_prebuilt_blocks: bool = True,
) -> TermResolutionResult | list[Any]:
    """Normalize a defined-term pipeline payload.

    Supported payload shapes are:

    * a prebuilt list of public response blocks
    * a ``(detector_result, resolution_result)`` tuple
    * a direct ``TermResolutionResult``

    Args:
        payload: Raw pipeline payload stored on orchestration state.
        error_message: Error message raised when the payload shape is unsupported.
        allow_prebuilt_blocks: Whether a prebuilt block list is accepted.

    Returns:
        Either the prebuilt list of blocks, or the validated resolution result.

    Raises:
        ValueError: If the payload does not match a supported shape.
    """
    if allow_prebuilt_blocks and isinstance(payload, list):
        return payload
    if isinstance(payload, tuple) and len(payload) == 2:
        _, resolution = payload
        if not isinstance(resolution, TermResolutionResult):
            raise ValueError(error_message)
        return resolution
    if isinstance(payload, TermResolutionResult):
        return payload
    raise ValueError(error_message)

def _resolve_structural_payload(
    payload: object,
    *,
    error_message: str,
    allow_prebuilt_blocks: bool = True,
) -> StructuralReferenceResolutionResult | list[Any]:
    """Normalize a pipeline payload to either a result object or prebuilt blocks.

    Pipelines may return one of three shapes:

    * a prebuilt list of public response blocks
    * a ``(detector_result, resolution_result)`` tuple, where the second element
      is the composition-ready resolution object
    * a direct resolution object

    This helper extracts the resolution object when needed and validates that it
    matches the expected result type.

    Args:
        payload: Raw pipeline payload stored on orchestration state.
        error_message: Error message raised when the payload shape is unsupported.
        allow_prebuilt_blocks: Whether a prebuilt block list is accepted.

    Returns:
        Either the prebuilt list of blocks, or the validated resolution result.

    Raises:
        ValueError: If the payload does not match a supported shape for the
            requested pipeline.
    """
    if allow_prebuilt_blocks and isinstance(payload, list):
        return payload
    if isinstance(payload, tuple) and len(payload) == 2:
        _, resolution = payload
        if not isinstance(resolution, StructuralReferenceResolutionResult):
            raise ValueError(error_message)
        return resolution
    if isinstance(payload, StructuralReferenceResolutionResult):
        return payload
    raise ValueError(error_message)


def map_orchestration_state(
    state: OrchestrationState,
) -> tuple[OrchestrationMeta, list[PipelineError]]:
    """Map internal orchestration state to public metadata and pipeline errors.

    Args:
        state: Finished orchestration state containing requested targets,
            completed targets, failed targets, and per-pipeline errors.

    Returns:
        A tuple containing public orchestration metadata and pipeline errors in
        failed-target order.
    """
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

def _resolve_acronym_payload(
    payload: object,
    *,
    error_message: str,
    allow_prebuilt_blocks: bool = True,
) -> AcronymPipelineResult | list[Any]:
    """Normalize an acronym pipeline payload for response composition.

    Supported payload shapes are:

    * a prebuilt list of public response blocks
    * an ``AcronymPipelineResult``
    * a legacy ``(detector_result, extraction_result)`` tuple

    Legacy tuple payloads are converted into ``AcronymPipelineResult`` so
    downstream composition can rely on a single explicit acronym result shape.

    Args:
        payload: Raw pipeline payload stored on orchestration state.
        error_message: Error message raised when the payload shape is unsupported.
        allow_prebuilt_blocks: Whether a prebuilt block list is accepted.

    Returns:
        Either the prebuilt list of blocks, or a normalized
        ``AcronymPipelineResult``.

    Raises:
        ValueError: If the payload does not match a supported acronym pipeline
            shape.
    """
    if allow_prebuilt_blocks and isinstance(payload, list):
        return payload

    if isinstance(payload, AcronymPipelineResult):
        return payload

    if isinstance(payload, tuple) and len(payload) == 2:
        det_res, extr = payload
        if not isinstance(det_res, AcronymDetectorResult):
            raise ValueError(error_message)
        if not isinstance(extr, ExtractionResult):
            raise ValueError(error_message)
        return AcronymPipelineResult(
            detector_result=det_res,
            extraction_result=extr,
        )

    raise ValueError(error_message)

def compose_sections(
    state: OrchestrationState,
    *,
    opts: ResolveOptions,
    lang: str,
    resolution_mode: ResolutionMode,
    glossary_repo: GlossaryRepository,
) -> dict[str, Any]:
    """Compose public response sections from completed pipeline results.

    Each supported section is initialized to an empty list. For completed
    pipelines, the stored payload is normalized and then either passed through
    directly when it already contains response blocks, or mapped into public
    schema blocks using the relevant pipeline mapper.

    Args:
        state: Orchestration state containing per-pipeline results.
        opts: Resolved API options for the request.
        lang: Language hint used by acronym mapping and enrichment.
        resolution_mode: Requested resolution mode used when attaching acronym
            selection metadata.
        glossary_repo: Read-only glossary repository used for acronym
            enrichment and metadata attachment.

    Returns:
        A dictionary containing the ``acronyms``, ``defined_terms``, and
        ``structural_references`` response sections.

    Raises:
        ValueError: If a completed pipeline payload has an unsupported shape.
    """
    sections: dict[str, Any] = {
        "acronyms": [],
        "defined_terms": [],
        "structural_references": [],
    }

    if PIPELINE_ACRONYMS in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_ACRONYMS].payload
        acronym_result = _resolve_acronym_payload(
            payload,
            error_message="Unsupported acronym pipeline payload shape.",
        )

        if isinstance(acronym_result, list):
            sections["acronyms"] = acronym_result
        else:
            blocks = map_acronym_pipeline_to_blocks(
                det_res=acronym_result.detector_result,
                extr=acronym_result.extraction_result,
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

    if PIPELINE_DEFINED_TERMS in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_DEFINED_TERMS].payload
        defined_term_result = _resolve_defined_term_payload(
            payload,
            error_message="Unsupported defined-term pipeline payload shape.",
        )
        if isinstance(defined_term_result, list):
            sections["defined_terms"] = defined_term_result
        else:
            sections["defined_terms"] = map_defined_term_blocks(defined_term_result)

    if PIPELINE_STRUCTURAL_REFERENCES in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_STRUCTURAL_REFERENCES].payload
        structural_result = _resolve_structural_payload(
            payload,
            error_message="Unsupported structural-reference pipeline payload shape.",
        )
        if isinstance(structural_result, list):
            sections["structural_references"] = structural_result
        else:
            sections["structural_references"] = map_structural_blocks(structural_result)
    return sections
