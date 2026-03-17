from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, confloat, conint, constr, field_validator

from public_api.schemas.base import BaseSchema
from public_api.schemas.glossary import AcronymBlock


class ResolveMeta(BaseSchema):
    processing_ms: int
    model_version: str = Field(..., description="plainera-core version used for processing.")
    input_chars: int


class SelectionReason(str, Enum):
    in_document_definition = "in_document_definition"
    single_candidate = "single_candidate"
    highest_score = "highest_score"
    fallback_general = "fallback_general"
    inactive_filtered = "inactive_filtered"

    # Future-ready only; do not rely on these yet
    exact_domain_match = "exact_domain_match"
    domain_priority_policy = "domain_priority_policy"


class CandidateProvenance(str, Enum):
    document = "document"
    glossary = "glossary"
    seed = "seed"
    system = "system"


class ResolveCandidate(BaseSchema):
    domain: str | None = Field(None, description="Sense/domain tag if known, else null.")
    definition: str = Field(..., description="Candidate resolved meaning.")
    score: confloat(ge=0.0, le=1.0) = Field(..., description="Deterministic ranking score. In the MVP this is primarily used for candidate ordering, not calibrated confidence.")  # type: ignore[valid-type]
    provenance: CandidateProvenance = Field(..., description="Where the candidate came from.")
    source_ref: str | None = Field(
        None,
        description="Optional stable source reference, e.g. text span or glossary sense id.",
    )


class SelectedCandidate(BaseSchema):
    domain: str | None = Field(None, description="Selected sense/domain tag if known, else null.")
    definition: str = Field(..., description="Chosen resolved meaning.")
    reason: SelectionReason = Field(..., description="Deterministic selection reason.")


class SelectionMeta(BaseSchema):
    policy_used: str | None = Field(
        None,
        description="Optional deterministic policy identifier used during selection.",
    )
    filtered_inactive_count: conint(ge=0) = Field(  # type: ignore[valid-type]
        0,
        description="Number of inactive candidates removed before selection.",
    )


class ResolvedAcronymBlock(AcronymBlock):
    candidates: list[ResolveCandidate] = Field(
        default_factory=list,
        description="Ordered candidate senses, best to worst, bounded by request cap.",
    )
    selected: SelectedCandidate | None = Field(
        None,
        description="Chosen candidate when at least one viable sense exists.",
    )
    conflict: bool = Field(
        False,
        description="True when more than one viable candidate existed.",
    )
    conflict_count: conint(ge=0) = Field(  # type: ignore[valid-type]
        0,
        description="Number of viable candidates considered after filtering.",
    )
    selection: SelectionMeta | None = Field(
        None,
        description="Additional deterministic selection metadata.",
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
        5, description="Maximum candidate definitions/senses to return per acronym."
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
    options: ResolveOptions | None = None

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "text": "The Metropolitan Police Service (MPS) operates in London.",
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
    meta: ResolveMeta

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
                                    }
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
                                    "source_ref": "sense:42",
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
                                "policy_used": None,
                                "filtered_inactive_count": 0,
                            },
                        }
                    ],
                    "meta": {
                        "processing_ms": 12,
                        "model_version": "plainera-core@1.0.0",
                        "input_chars": 68,
                    },
                }
            ]
        }
