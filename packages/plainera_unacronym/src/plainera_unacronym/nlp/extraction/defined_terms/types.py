from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from plainera_unacronym.nlp.common.types import TextSpanTuple
from plainera_unacronym.nlp.detection.defined_terms.types import DefinedTermMention
from plainera_unacronym.nlp.extraction.tiers.types import Tier2Report

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


@dataclass(frozen=True, slots=True)
class TermResolution:
    occurrence_span: TextSpanTuple
    term: str
    normalized_key: str
    chosen_sense_id: str | None
    chosen_definition_span: TextSpanTuple | None
    candidate_scores: tuple[TermCandidateScore, ...]
    resolution_method: Literal["tier1", "tier2_blend", "unresolved"]


@dataclass(frozen=True, slots=True)
class TermDefinitionEntry:
    surface: str
    normalized_key: str
    intro_span: TextSpanTuple
    definition_span: TextSpanTuple | None
    definition_text: str | None
    intro_kind: str
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class TermTier1OccurrenceRanking:
    occ: DefinedTermMention
    candidate_scores: dict[str, float]
    chosen_sense_id: str | None
    gap: float
    margin: float


@dataclass(frozen=True)
class TermTier2OccurrenceRanking:
    occ: DefinedTermMention
    applied: bool
    skip_reason: TermTier2SkipReason | None
    tier2_sims: dict[str, float] | None
    blended_scores: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class TermResolutionResult:
    term_sense_index: dict[str, tuple[TermSense, ...]] = field(default_factory=dict)
    sense_index: Mapping[str, TermSense] = field(default_factory=dict)
    term_resolutions: list[TermResolution] = field(default_factory=list)
    ambiguous_keys: tuple[str, ...] = field(default_factory=tuple)
    undecided: list[TermResolution] = field(default_factory=list)
    tier2_report: Tier2Report | None = None
    tier2_ranked: tuple[TermTier2OccurrenceRanking, ...] = ()
