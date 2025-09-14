from typing import  Optional

from pydantic import Field

from public_api.schemas.base import BaseSchema
from public_api.schemas.definition import Definition, Span
from public_api.schemas.glossary import GlossaryBlock


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
