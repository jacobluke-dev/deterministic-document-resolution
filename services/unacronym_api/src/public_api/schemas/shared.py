from enum import Enum
from typing import Optional

from pydantic import Field, confloat

from public_api.schemas.base import BaseSchema


class DefinitionSource(str, Enum):
    extracted = "extracted"
    glossary = "glossary"


class Span(BaseSchema):
    start: int = Field(..., description="Start offset (inclusive).")
    end: int = Field(..., description="End offset (exclusive).")


class Definition(BaseSchema):
    text: str = Field(..., description="Expanded form of the acronym.")
    start: Optional[int] = Field(
        None, description="Start offset of definition text if found in source text."
    )
    end: Optional[int] = Field(
        None, description="End offset (exclusive)."
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(  # type: ignore[valid-type]
        ..., description="0–1 confidence score for this definition."
    )
    source: DefinitionSource = Field(
        ..., description="Origin of the definition: extracted|glossary."
    )


class TextSpan(BaseSchema):
    start: int = Field(..., description="Inclusive start character offset.")
    end: int = Field(..., description="Exclusive end character offset.")
