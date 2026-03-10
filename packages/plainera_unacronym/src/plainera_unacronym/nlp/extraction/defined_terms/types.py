from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from plainera_unacronym.nlp.common.types import TextSpanTuple
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermOccurrence
from plainera_unacronym.nlp.extraction.tiers.types import Tier2Report


@dataclass(frozen=True)
class TermSense:
    sense_id: str
    surface: str
    normalized_key: str
    ordinal: int
    intro_span: TextSpanTuple
    definition_span: TextSpanTuple | None
    definition_text: str | None
    intro_kind: str
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class TermCandidateScore:
    sense_id: str
    total_score: float
    tier1_score: float
    tier2_score: float | None
    definition_span: TextSpanTuple | None
    components: dict[str, float]


@dataclass(frozen=True)
class TermResolution:
    occurrence_span: TextSpanTuple
    surface: str
    normalized_key: str
    chosen_sense_id: str | None
    chosen_definition_span: TextSpanTuple | None
    candidate_scores: tuple[TermCandidateScore, ...]
    resolution_method: str


@dataclass(frozen=True)
class TermResolutionResult:
    term_sense_index: dict[str, tuple[TermSense, ...]]
    term_resolutions: tuple[TermResolution, ...]


@dataclass(frozen=True, slots=True)
class TermResolution:
    occurrence_span: TextSpanTuple
    normalized_key: str
    chosen_sense_id: str | None
    definition_span: TextSpanTuple | None
    candidate_scores: dict[str, float]
    resolution_method: str


TermTier2SkipReason = Literal[
    "disabled",
    "pending",
    "model_unavailable",
    "single_candidate",
    "not_ambiguous",
    "tier1_decided",
    "tier1_confident",
    "no_senses",
]


@dataclass(frozen=True)
class TermTier1OccurrenceRanking:
    occ: DefinedTermOccurrence
    candidate_scores: dict[str, float]
    chosen_sense_id: str | None
    gap: float
    margin: float


@dataclass(frozen=True)
class TermTier2OccurrenceRanking:
    occ: DefinedTermOccurrence
    applied: bool
    skip_reason: TermTier2SkipReason | None
    tier2_sims: dict[str, float] | None
    blended_scores: dict[str, float] | None
