from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from plainera_unacronym.nlp.common.types import AcronymSense, OccurrenceLite

FloatMat = NDArray[np.floating]
FloatVec = NDArray[np.floating]

Tier2SkipReason = Literal[
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
class Tier1OccurrenceRanking:
    occ: OccurrenceLite
    candidate_scores: dict[str, float]  # preserve insertion order
    chosen_sense_id: str | None
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
    senses_by_acronym: dict[str, list[AcronymSense]] = field(default_factory=dict)
    sense_index: dict[str, AcronymSense] = field(default_factory=dict)
    occurrences: list[OccurrenceLite] = field(default_factory=list)
    ranked: list[Tier1OccurrenceRanking] = field(default_factory=list)


@dataclass
class Tier2Work:
    ranked: list[Tier2OccurrenceRanking] = field(default_factory=list)
    report: Tier2Report | None = None


@dataclass
class DisambigWork:
    tier1: Tier1Work = field(default_factory=Tier1Work)
    tier2: Tier2Work = field(default_factory=Tier2Work)
