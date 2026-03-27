from __future__ import annotations

from typing import Literal

from pydantic import Field

from public_api.schemas.base import BaseSchema
from public_api.schemas.shared import TextSpan


class DefinedTermMeaningRef(BaseSchema):
    meaning_id: str = Field(..., description="Stable deterministic meaning identifier.")
    surface: str = Field(..., description="Original introduced term text.")
    normalized_key: str = Field(..., description="Canonical normalized term key.")
    ordinal: int = Field(..., description="Per-key document-order ordinal.")
    intro_span: TextSpan = Field(..., description="Span of the introduced term.")
    definition_span: TextSpan | None = Field(
        None,
        description="Span of the trailing definition text when present.",
    )
    definition_text: str | None = Field(
        None,
        description="Extracted trailing definition text when present.",
    )
    intro_kind: str = Field(..., description="Introduction form used for the meaning.")
    section_path: list[str] = Field(default_factory=list, description="Structural path locating the introduction.")
    alias_target_span: TextSpan | None = Field(
        None,
        description="Antecedent span for parenthetical alias introductions when present.",
    )
    alias_target_text: str | None = Field(
        None,
        description="Antecedent text for parenthetical alias introductions when present.",
    )


class DefinedTermCandidate(BaseSchema):
    meaning_id: str = Field(..., description="Candidate meaning identifier.")
    total_score: float = Field(..., description="Final deterministic candidate score.")
    tier1_score: float = Field(..., description="Tier 1 heuristic score.")
    tier2_score: float | None = Field(None, description="Tier 2 semantic score when available.")
    definition_span: TextSpan | None = Field(
        None,
        description="Definition span associated with the candidate meaning when available.",
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Named score components contributing to the total score.",
    )


class DefinedTermBlock(BaseSchema):
    occurrence_span: TextSpan = Field(..., description="Span of the later defined-term occurrence.")
    term: str = Field(..., description="Surface text of the occurrence.")
    normalized_key: str = Field(..., description="Canonical normalized key for the occurrence.")
    chosen_meaning_id: str | None = Field(
        None,
        description="Selected meaning identifier, or null when unresolved.",
    )
    chosen_definition_span: TextSpan | None = Field(
        None,
        description="Definition span associated with the selected meaning when available.",
    )
    resolution_method: Literal["tier1", "tier2_blend", "unresolved"] = Field(
        ...,
        description="Resolution path used for the final decision.",
    )
    resolved: bool = Field(..., description="True when a meaning was selected.")
    candidate_scores: list[DefinedTermCandidate] = Field(
        default_factory=list,
        description="Ranked candidate meanings considered for this occurrence.",
    )
    chosen_meaning: DefinedTermMeaningRef | None = Field(
        None,
        description="Resolved meaning record when available.",
    )


class DefinedTermMeaningBlock(BaseSchema):
    meaning_id: str = Field(..., description="Stable deterministic meaning identifier.")
    surface: str = Field(..., description="Original introduced term text.")
    normalized_key: str = Field(..., description="Canonical normalized key used for grouping.")
    ordinal: int = Field(..., description="Per-key ordinal assigned in document order.")
    intro_span: TextSpan = Field(..., description="Span of the introduced term text.")
    definition_span: TextSpan | None = Field(
        None,
        description="Span of the trailing definition text when present.",
    )
    definition_text: str | None = Field(
        None,
        description="Extracted trailing definition text when present.",
    )
    intro_kind: str = Field(..., description="Introduction form, for example quoted_means.")
    section_path: list[str] = Field(
        default_factory=list,
        description="Structural path locating the introduction within the document.",
    )
    alias_target_span: TextSpan | None = Field(
        None,
        description="Antecedent span for parenthetical alias introductions when present.",
    )
    alias_target_text: str | None = Field(
        None,
        description="Antecedent text for parenthetical alias introductions when present.",
    )


class DefinedTermCandidateBlock(BaseSchema):
    meaning_id: str = Field(..., description="Candidate meaning identifier.")
    total_score: float = Field(..., description="Final deterministic ranking score.")
    tier1_score: float = Field(..., description="Tier 1 heuristic score.")
    tier2_score: float | None = Field(
        None,
        description="Tier 2 semantic score when available.",
    )
    definition_span: TextSpan | None = Field(
        None,
        description="Definition span associated with the candidate meaning when available.",
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Named score components contributing to the total score.",
    )
