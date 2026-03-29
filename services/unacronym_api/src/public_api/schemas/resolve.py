from __future__ import annotations

import math
from enum import Enum

from plainera_unacronym.orchestration.state import PipelineErrorCode
from pydantic import Field, confloat, conint, constr, field_validator

from public_api.schemas.base import BaseSchema
from public_api.schemas.extraction_types.acronyms import ResolvedAcronymBlock
from public_api.schemas.extraction_types.defined_terms import DefinedTermBlock
from public_api.schemas.extraction_types.structural import StructuralReferenceBlock


class ResolutionMode(str, Enum):
    """Deterministic resolution policy for selecting a final acronym meaning.

    STRICT:
        Select only when the resolver can do so conservatively. Avoid permissive
        fallback behaviour and prefer leaving the acronym unresolved when
        multiple glossary candidates remain.

    DOMAIN_PRIORITY:
        Use the standard deterministic policy. Prefer the strongest available
        candidate according to the resolver's normal ordering and fallback
        rules. This is the default balanced mode.

    FALLBACK_GENERAL:
        Prefer returning a usable result whenever possible. If no stronger
        candidate is available, prefer a general-domain meaning before falling
        back to the first deterministically ordered candidate.
    """
    STRICT = "strict"
    DOMAIN_PRIORITY = "domain_priority"
    FALLBACK_GENERAL = "fallback_general"


class ResolveTarget(str, Enum):
    ACRONYMS = "acronyms"
    DEFINED_TERMS = "defined_terms"
    STRUCTURAL_REFERENCES = "structural_references"


class OrchestrationMeta(BaseSchema):
    requested: list[str]
    completed: list[str]
    failed: list[str]


class PipelineError(BaseSchema):
    pipeline: str
    code: PipelineErrorCode
    message: str


class ResolveMeta(BaseSchema):
    processing_ms: int
    model_version: str = Field(..., description="plainera-core version used for processing.")
    input_chars: int
    resolution_mode: ResolutionMode = Field(
        ...,
        description="Resolution policy applied during sense selection.",
    )


class ResolveOptions(BaseSchema):
    locale: str = Field(
        "en-GB",
        description="Locale hint for heuristics. ISO BCP-47 tag. Supported: en-GB, en-US.",
    )
    window_chars: conint(ge=0) = Field(  # type: ignore[valid-type]
        120, description="Context window size used in responses."
    )
    max_definitions_per_acronym: conint(ge=1, le=20) = Field(  # type: ignore[valid-type]
        5, description="Maximum candidate definitions/meanings to return per acronym."
    )
    include_glossary_enrichment: bool = Field(
        True,
        description="If true, attempt curated joins from repository (read-only).",
    )
    return_occurrences: bool = Field(
        True, description="If true, include all {start,end} positions."
    )
    min_confidence: confloat(ge=0.0, le=1.0) = Field(  # type: ignore[valid-type]
        0.0, description="Drop candidates below this confidence threshold."
    )

    @field_validator("min_confidence")
    @classmethod
    def _finite_min_confidence(cls, v: float) -> float:
        # Pydantic range checks do not reliably exclude NaN.
        if not math.isfinite(float(v)):
            raise ValueError("min_confidence must be a finite number.")
        return v


class ResolveRequest(BaseSchema):
    text: constr(min_length=1, max_length=100_000) = Field(  # type: ignore[valid-type]
        ..., description="Raw document content. Max length 100,000 characters."
    )

    resolution_mode: ResolutionMode = Field(
        default=ResolutionMode.DOMAIN_PRIORITY,
        description="Controls deterministic acronym sense selection policy.",
    )

    options: ResolveOptions | None = None

    targets: list[ResolveTarget] | None = Field(
        default=None,
        description=(
            "Optional explicit pipeline targets to run. "
            "If omitted, all supported targets are selected."
        ),
    )

    @field_validator("targets")
    @classmethod
    def _validate_targets_not_empty(
        cls,
        v: list[ResolveTarget] | None,
    ) -> list[ResolveTarget] | None:
        if v == []:
            raise ValueError("targets must not be empty when provided.")
        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "text": "The Metropolitan Police Service (MPS) operates in London.",
                    "resolution_mode": "domain_priority",
                    "targets": ["acronyms", "defined_terms", "structural_references"],
                    "options": {
                        "locale": "en-GB",
                        "window_chars": 120,
                        "max_definitions_per_acronym": 5,
                        "include_glossary_enrichment": True,
                        "return_occurrences": True,
                        "min_confidence": 0.0,
                    },
                },
                {"text": ""},  # invalid: shown in negative tests
            ]
        }


class ResolveResponse(BaseSchema):
    acronyms: list[ResolvedAcronymBlock] = Field(default_factory=list)
    defined_terms: list[DefinedTermBlock] = Field(default_factory=list)
    structural_references: list[StructuralReferenceBlock] = Field(default_factory=list)
    meta: ResolveMeta
    orchestration: OrchestrationMeta
    errors: list[PipelineError] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "acronyms": [
                        {
                            "acronym": "GP",
                            "first_occurrence": {"start": 33, "end": 35},
                            "definitions": [
                                {
                                    "text": "General Practitioner",
                                    "start": 4,
                                    "end": 24,
                                    "confidence": 0.99,
                                    "source": "extracted",
                                }
                            ],
                            "occurrences": [{"start": 33, "end": 35}],
                            "glossary": {
                                "matches": [
                                    {
                                        "definition": "General Practitioner",
                                        "domain": "medical",
                                        "lang": "en",
                                        "confidence": 0.98,
                                        "source": "system",
                                    },
                                    {
                                        "definition": "General Partner",
                                        "domain": "finance",
                                        "lang": "en",
                                        "confidence": 0.82,
                                        "source": "system",
                                    },
                                ]
                            },
                            "candidates": [
                                {
                                    "domain": None,
                                    "definition": "General Practitioner",
                                    "score": 1.0,
                                    "provenance": "document",
                                    "source_ref": "text_span:4-24",
                                },
                                {
                                    "domain": "finance",
                                    "definition": "General Partner",
                                    "score": 0.0,
                                    "provenance": "glossary",
                                    "source_ref": "meaning:42",
                                },
                            ],
                            "selected": {
                                "domain": None,
                                "definition": "General Practitioner",
                                "reason": "in_document_definition",
                            },
                            "conflict": True,
                            "conflict_count": 2,
                            "selection": {
                                "filtered_inactive_count": 0,
                            },
                        }
                    ],
                    "defined_terms": [
                        {
                            "occurrence_span": {"start": 120, "end": 128},
                            "term": "Services",
                            "normalized_key": "services",
                            "chosen_meaning_id": "term|services|1",
                            "chosen_definition_span": {"start": 40, "end": 96},
                            "resolution_method": "tier1",
                            "resolved": True,
                            "candidate_scores": [
                                {
                                    "meaning_id": "term|services|1",
                                    "total_score": 1.0,
                                    "tier1_score": 1.0,
                                    "tier2_score": None,
                                    "definition_span": {"start": 40, "end": 96},
                                    "components": {
                                        "section_proximity": 0.5,
                                        "recency": 0.5,
                                    },
                                }
                            ],
                            "chosen_meaning": {
                                "meaning_id": "term|services|1",
                                "surface": "Services",
                                "normalized_key": "services",
                                "ordinal": 1,
                                "intro_span": {"start": 20, "end": 28},
                                "definition_span": {"start": 40, "end": 96},
                                "definition_text": "the support and maintenance services described in Schedule 1",
                                "intro_kind": "quoted_means",
                                "section_path": ["1", "Definitions"],
                                "alias_target_span": None,
                                "alias_target_text": None,
                            },
                        }
                    ],
                    "structural_references": [
                        {
                            "kind": "Section",
                            "label": "4.2",
                            "canonical_label": "4.2",
                            "normalized_key": "section:4.2",
                            "canonical_key": "section|4.2",
                            "reference_span": {"start": 210, "end": 221},
                            "target_span": {"start": 420, "end": 438},
                            "match_strategy": "forward",
                            "strength": 1.0,
                            "provenance": "document",
                            "resolved": True,
                        }
                    ],
                    "meta": {
                        "processing_ms": 12,
                        "model_version": "plainera-core@1.0.0",
                        "input_chars": 680,
                        "resolution_mode": "domain_priority",
                    },
                    "orchestration": {
                        "requested": ["acronyms", "defined_terms", "structural_references"],
                        "completed": ["acronyms", "structural_references"],
                        "failed": ["defined_terms"],
                    },
                    "errors": [
                        {
                            "pipeline": "defined_terms",
                            "code": "PIPELINE_EXECUTION_FAILED",
                            "message": "boom",
                        }
                    ],
                }
            ]
        }
