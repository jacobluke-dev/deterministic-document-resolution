from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from plainera_unacronym.nlp.common.types import TextSpanTuple
from plainera_unacronym.nlp.detection.defined_terms.types import DefinedTermMention
from plainera_unacronym.nlp.extraction.tiers.types import Tier2Report

type TermTier2SkipReason = Literal[
    "disabled",
    "pending",
    "model_unavailable",
    "single_candidate",
    "not_ambiguous",
    "tier1_decided",
    "tier1_confident",
    "no_meanings",
]


@dataclass(frozen=True)
class TermMeaning:
    """Resolved meaning record for a defined term introduction.

    Represents one deterministic meaning instance for a normalized term key.
    Repeated introductions of the same normalized key are assigned distinct
    ordinals and meaning IDs in document order.

    Attributes:
        meaning_id: Stable deterministic identifier, for example
            ``term|services|1``.
        surface: Original introduced term text as it appeared in the document.
        normalized_key: Canonical normalized key used for grouping and
            resolution.
        ordinal: Per-key ordinal assigned in document order.
        intro_span: Span of the introduced term text.
        definition_span: Span of the trailing ``means`` / ``shall mean``
            definition text when present.
        definition_text: Extracted trailing definition text when present.
        intro_kind: Introduction form, for example ``quoted_means`` or
            ``parenthetical_alias``.
        section_path: Structural path locating the introduction within the
            document.
        alias_target_span: Span of the antecedent phrase immediately preceding a
            parenthetical alias introduction, when extracted.
        alias_target_text: Antecedent phrase text immediately preceding a
            parenthetical alias introduction, when extracted.
    """

    meaning_id: str
    surface: str
    normalized_key: str
    ordinal: int
    intro_span: TextSpanTuple
    definition_span: TextSpanTuple | None
    definition_text: str | None
    intro_kind: str
    section_path: tuple[str, ...]
    alias_target_span: TextSpanTuple | None = None
    alias_target_text: str | None = None


@dataclass(frozen=True)
class TermCandidateScore:
    """Scored candidate meaning for a later defined-term occurrence.

    Attributes:
        meaning_id: Candidate meaning identifier being scored.
        total_score: Final score after combining all available ranking signals.
        tier1_score: Tier 1 heuristic score for the candidate.
        tier2_score: Tier 2 semantic score when available; otherwise ``None``.
        definition_span: Definition span associated with the candidate meaning,
            when available.
        components: Named score components contributing to the total score.
    """

    meaning_id: str
    total_score: float
    tier1_score: float
    tier2_score: float | None
    definition_span: TextSpanTuple | None
    components: dict[str, float]


@dataclass(frozen=True, slots=True)
class TermResolution:
    """Resolution outcome for a later occurrence of a defined term.

    Attributes:
        occurrence_span: Span of the resolved later mention.
        term: Surface text of the occurrence.
        normalized_key: Canonical normalized key for the occurrence.
        chosen_meaning_id: Selected meaning identifier, or ``None`` when the
            occurrence remains unresolved.
        chosen_definition_span: Definition span associated with the selected
            meaning, when available.
        candidate_scores: Ranked candidate meaning scores considered for this
            occurrence.
        resolution_method: Resolution path used to choose the winning meaning,
            or ``unresolved`` when no winner was selected.
    """

    occurrence_span: TextSpanTuple
    term: str
    normalized_key: str
    chosen_meaning_id: str | None
    chosen_definition_span: TextSpanTuple | None
    candidate_scores: tuple[TermCandidateScore, ...]
    resolution_method: Literal["tier1", "tier2_blend", "unresolved"]


@dataclass(frozen=True, slots=True)
class TermDefinitionEntry:
    """Intermediate extracted definition record for a detected introduction.

    Produced from detector introductions before deterministic meaning IDs are
    assigned. Carries raw extracted definition data and any alias antecedent
    information needed to later build ``TermMeaning`` records.

    Attributes:
        surface: Original introduced term text.
        normalized_key: Canonical normalized key for grouping repeated
            introductions.
        intro_span: Span of the introduced term text.
        definition_span: Span of the trailing definition text when the
            introduction form supports it.
        definition_text: Extracted trailing definition text when present.
        intro_kind: Introduction form recorded by detection.
        alias_target_span: Span of the antecedent phrase immediately preceding a
            parenthetical alias introduction, when extracted.
        alias_target_text: Antecedent phrase text immediately preceding a
            parenthetical alias introduction, when extracted.
        section_path: Structural path locating the introduction within the
            document.
    """

    surface: str
    normalized_key: str
    intro_span: TextSpanTuple
    definition_span: TextSpanTuple | None
    definition_text: str | None
    intro_kind: str
    alias_target_span: TextSpanTuple | None = None
    alias_target_text: str | None = None
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class TermTier1OccurrenceRanking:
    """Tier 1 ranking result for a later defined-term occurrence.

    Attributes:
        occ: Later mention being ranked against candidate meanings.
        candidate_scores: Tier 1 heuristic scores keyed by meaning ID.
        chosen_meaning_id: Tier 1 winner when a deterministic choice was made;
            otherwise ``None``.
        gap: Absolute score gap between the top two candidates.
        margin: Relative confidence margin for the top-ranked candidate.
    """

    occ: DefinedTermMention
    candidate_scores: dict[str, float]
    chosen_meaning_id: str | None
    gap: float
    margin: float


@dataclass(frozen=True)
class TermTier2OccurrenceRanking:
    """Tier 2 semantic reranking outcome for a later occurrence.

    Attributes:
        occ: Later mention being reranked.
        applied: Whether Tier 2 semantic reranking was actually applied.
        skip_reason: Reason Tier 2 was skipped, when not applied.
        tier2_sims: Raw semantic similarity scores by meaning ID when available.
        blended_scores: Final blended Tier 1 / Tier 2 scores when available.
    """

    occ: DefinedTermMention
    applied: bool
    skip_reason: TermTier2SkipReason | None
    tier2_sims: dict[str, float] | None
    blended_scores: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class TermResolutionResult:
    """Final resolution output for defined terms within a document.

    Aggregates meaning indexes, per-occurrence resolution decisions, unresolved
    ambiguities, and Tier 2 reporting for the defined-term pipeline.

    Attributes:
        term_meaning_index: Meanings grouped by normalized term key.
        meaning_index: Flat meaning lookup keyed by meaning ID.
        term_resolutions: Resolution outcomes for later defined-term mentions.
        ambiguous_keys: Normalized keys that had multiple plausible meanings in
            the document.
        undecided: Occurrence resolutions that remained unresolved.
        tier2_report: Aggregate Tier 2 application/skip report, when available.
        tier2_ranked: Per-occurrence Tier 2 reranking outcomes.
    """

    term_meaning_index: dict[str, tuple[TermMeaning, ...]] = field(default_factory=dict)
    meaning_index: Mapping[str, TermMeaning] = field(default_factory=dict)
    term_resolutions: list[TermResolution] = field(default_factory=list)
    ambiguous_keys: tuple[str, ...] = field(default_factory=tuple)
    undecided: list[TermResolution] = field(default_factory=list)
    tier2_report: Tier2Report | None = None
    tier2_ranked: tuple[TermTier2OccurrenceRanking, ...] = ()
