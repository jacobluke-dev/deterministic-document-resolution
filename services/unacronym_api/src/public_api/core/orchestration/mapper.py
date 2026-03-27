from __future__ import annotations

from typing import Any
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult
from plainera_unacronym.nlp.extraction.structural.types import StructuralReferenceResolutionResult, \
    StructuralReferenceLink

from plainera_unacronym.orchestration import PIPELINE_ACRONYMS, PIPELINE_DEFINED_TERMS, PIPELINE_STRUCTURAL_REFERENCES
from plainera_unacronym.orchestration.state import OrchestrationState

from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import map_acronym_pipeline_to_blocks
from public_api.db.repos import GlossaryRepository
from public_api.schemas.shared import TextSpan
from public_api.schemas.extraction_types.structural import StructuralReferenceBlock
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

def _map_span(span: tuple[int, int]) -> TextSpan:
    return TextSpan(
        start=int(span[0]),
        end=int(span[1]),
    )


def map_structural_links_to_blocks(
    links: list[StructuralReferenceLink],
) -> list[StructuralReferenceBlock]:
    return [
        StructuralReferenceBlock(
            kind=link.kind,
            label=link.label,
            canonical_label=link.canonical_label,
            normalized_key=link.normalized_key,
            canonical_key=link.canonical_key,
            reference_span=_map_span(link.reference_span),
            target_span=None if link.target_span is None else _map_span(link.target_span),
            match_strategy=link.match_strategy,
            strength=float(link.strength),
            provenance=link.provenance,
            resolved=link.target_span is not None and link.match_strategy != "unresolved",
        )
        for link in links
    ]

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
            blocks = map_acronym_pipeline_to_blocks(
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
        if isinstance(payload, TermResolutionResult):
            sections["defined_terms"] = payload.term_resolutions
        else:
            raise ValueError("Unsupported defined-term pipeline payload shape.")

    if PIPELINE_STRUCTURAL_REFERENCES in state.completed_targets:
        payload = state.results_by_pipeline[PIPELINE_STRUCTURAL_REFERENCES].payload
        if isinstance(payload, StructuralReferenceResolutionResult):
            sections["structural_references"] = map_structural_links_to_blocks(payload.links)
        else:
            raise ValueError("Unsupported structural-reference pipeline payload shape.")

    return sections
