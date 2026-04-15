from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from document_resolution.nlp.common.types import AcronymMeaning, OccurrenceLite

FloatMat = NDArray[np.floating]
FloatVec = NDArray[np.floating]

type Tier2SkipReason = Literal[
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
class Tier1OccurrenceRanking:
    occ: OccurrenceLite
    candidate_scores: dict[str, float]  # preserve insertion order
    chosen_meaning_id: str | None
    gap: float
    margin: float


@dataclass(frozen=True)
class Tier2OccurrenceRanking:
    occ: OccurrenceLite
    applied: bool
    skip_reason: Tier2SkipReason | None
    tier2_sims: dict[str, float] | None
    blended_scores: dict[str, float] | None


@dataclass(frozen=True)
class Tier2Report:
    applied: int
    skipped: int
    reasons: dict[Tier2SkipReason, int]


@dataclass
class Tier1Work:
    """
    Tier-1 deterministic meaning-building and occurrence-ranking workspace.

    Holds the acronym meaning inventory derived from extracted definitions, along
    with the lightweight occurrence list and the Tier-1 ranking result for each
    occurrence.

    Attributes:
        meaning_by_acronym: Grouped meanings keyed by acronym surface/normalized key.
        meaning_index: Flat lookup of meaning_id -> AcronmMeaning.
        occurrences: Minimal occurrence records used for downstream scoring.
        ranked: Tier-1 ranking result for each occurrence, including candidate
            scores, chosen meaning (if any), and confidence separation metrics.
    """

    meaning_by_acronym: dict[str, list[AcronymMeaning]] = field(default_factory=dict)
    meaning_index: dict[str, AcronymMeaning] = field(default_factory=dict)
    occurrences: list[OccurrenceLite] = field(default_factory=list)
    ranked: list[Tier1OccurrenceRanking] = field(default_factory=list)


@dataclass
class Tier2Work:
    """
    Tier-2 semantic rerank workspace.

    Stores semantic rerank outputs aligned to Tier-1-ranked occurrences, plus a
    compact summary report describing how many occurrences were reranked versus
    skipped and why.

    Attributes:
        ranked: Tier-2 rerank results aligned to Tier-1 occurrence order.
        report: Aggregate Tier-2 application/skipping summary.
    """

    ranked: list[Tier2OccurrenceRanking] = field(default_factory=list)
    report: Tier2Report | None = None
