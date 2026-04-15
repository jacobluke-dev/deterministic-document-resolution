from enum import Enum
from typing import Optional

from public_api.schemas.base import BaseSchema
from public_api.schemas.glossary import GlossaryBlock
from public_api.schemas.shared import Definition, Span
from pydantic import Field, confloat, conint


class CandidateProvenance(str, Enum):
    document = "document"
    glossary = "glossary"
    seed = "seed"
    system = "system"


class SelectionReason(str, Enum):
    in_document_definition = "in_document_definition"
    single_candidate = "single_candidate"
    highest_score = "highest_score"
    fallback_general = "fallback_general"
    inactive_filtered = "inactive_filtered"

    # Future-ready only; do not rely on these yet
    exact_domain_match = "exact_domain_match"
    domain_priority_policy = "domain_priority_policy"


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


class ResolveCandidate(BaseSchema):
    domain: str | None = Field(None, description="Meaning/domain tag if known, else null.")
    definition: str = Field(..., description="Candidate resolved meaning.")
    score: confloat(ge=0.0, le=1.0) = Field(..., description="Deterministic ranking score. "  # type: ignore[valid-type]
                                                             "In the MVP this is primarily used for "
                                                             "candidate ordering, not calibrated confidence.")
    provenance: CandidateProvenance = Field(..., description="Where the candidate came from.")
    source_ref: str | None = Field(
        None,
        description="Optional stable source reference, e.g. text span or glossary meaning id.",
    )


class SelectedCandidate(BaseSchema):
    domain: str | None = Field(None, description="Selected meaning/domain tag if known, else null.")
    definition: str = Field(..., description="Chosen resolved meaning.")
    reason: SelectionReason = Field(..., description="Deterministic selection reason.")


class SelectionMeta(BaseSchema):
    filtered_inactive_count: conint(ge=0) = Field(  # type: ignore[valid-type]
        0,
        description="Number of inactive candidates removed before selection.",
    )


class ResolvedAcronymBlock(AcronymBlock):
    candidates: list[ResolveCandidate] = Field(
        default_factory=list,
        description="Ordered candidate meaning, best to worst, bounded by request cap.",
    )
    selected: SelectedCandidate | None = Field(
        None,
        description="Chosen candidate when at least one viable meaning exists.",
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
