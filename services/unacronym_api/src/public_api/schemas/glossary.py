from typing import Literal, Optional

from pydantic import Field, confloat, constr

from public_api.schemas.base import BaseSchema
from public_api.schemas.shared import Definition, Span


class GlossaryMatch(BaseSchema):
    definition: str
    domain: Optional[str] = Field(None, description="Domain tag if known, else null.")
    lang: constr(min_length=2, max_length=10) = Field(..., description="ISO language tag.") # type: ignore[valid-type]
    confidence: confloat(ge=0.0, le=1.0) # type: ignore[valid-type]
    source: Literal["system"] = "system"

class GlossaryBlock(BaseSchema):
    matches: list[GlossaryMatch] = Field(default_factory=list)
