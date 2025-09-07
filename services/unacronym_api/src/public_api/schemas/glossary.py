from typing import list, Literal, Optional

from pydantic import Field, confloat, constr

from public_api.schemas.base import BaseSchema
from public_api.schemas.definition import Definition, Span


class GlossaryMatch(BaseSchema):
    definition: str
    domain: Optional[str] = Field(None, description="Domain tag if known, else null.")
    lang: constr(min_length=2, max_length=10) = Field(..., description="ISO language tag.") # type: ignore[valid-type]
    confidence: confloat(ge=0.0, le=1.0) # type: ignore[valid-type]
    source: Literal["system"] = "system"

class GlossaryBlock(BaseSchema):
    matches: list[GlossaryMatch] = Field(default_factory=list)

class AcronymBlock(BaseSchema):
    acronym: str = Field(..., description="Surface form exactly as detected (no lowercasing).")
    first_occurrence: Span = Field(
        ..., description="First occurrence offsets in Python-slice semantics (end exclusive)."
    )
    definitions: list[Definition] = Field(
        default_factory=list,
        description="Candidate definitions for this acronym (ranked by confidence).",
    )
    occurrences: Optional[list[Span]] = Field(
        None, description="All occurrences when return_occurrences=true."
    )
    glossary: Optional[GlossaryBlock] = Field(
        None, description="Curated matches present when enrichment is enabled/available."
    )
