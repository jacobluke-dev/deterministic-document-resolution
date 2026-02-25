from __future__ import annotations

import math

from pydantic import Field, confloat, conint, constr, field_validator

from public_api.schemas.base import BaseSchema
from public_api.schemas.glossary import AcronymBlock


class ResolveMeta(BaseSchema):
    processing_ms: int
    model_version: str = Field(..., description="plainera-core version used for processing.")
    input_chars: int


class ResolveOptions(BaseSchema):
    locale: str = Field(
        "en-GB",
        description="Locale hint for heuristics. ISO BCP-47 tag. Supported: en-GB, en-US.",
    )
    window_chars: conint(ge=0) = Field(  # type: ignore[valid-type]
        120, description="Context window size used in responses."
    )
    max_definitions_per_acronym: conint(ge=1, le=20) = Field(  # type: ignore[valid-type]
        5, description="Maximum candidate definitions to return per acronym."
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
    acronyms: list[AcronymBlock] = Field(default_factory=list)
    meta: ResolveMeta

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "acronyms": [
                        {
                            "acronym": "MPS",
                            "first_occurrence": {"start": 34, "end": 37},
                            "definitions": [
                                {
                                    "text": "Metropolitan Police Service",
                                    "start": 4,
                                    "end": 31,
                                    "confidence": 0.96,
                                    "source": "extracted",
                                }
                            ],
                            "occurrences": [{"start": 34, "end": 37}],
                            "glossary": {
                                "matches": [
                                    {
                                        "definition": "Metropolitan Police Service",
                                        "domain": None,
                                        "lang": "en",
                                        "confidence": 0.99,
                                        "source": "system",
                                    }
                                ]
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
