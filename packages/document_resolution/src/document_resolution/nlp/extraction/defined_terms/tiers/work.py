from __future__ import annotations

from dataclasses import dataclass, field

from document_resolution.nlp.detection.defined_terms.types import DefinedTermMention
from document_resolution.nlp.extraction.defined_terms.types import (
    TermMeaning,
    TermTier1OccurrenceRanking,
    TermTier2OccurrenceRanking,
)
from document_resolution.nlp.extraction.tiers.types import Tier2Report


@dataclass
class TermTier1Work:
    term_meaning_index: dict[str, tuple[TermMeaning, ...]] = field(default_factory=dict)
    meaning_index: dict[str, TermMeaning] = field(default_factory=dict)
    occurrences: list[DefinedTermMention] = field(default_factory=list)
    ranked: list[TermTier1OccurrenceRanking] = field(default_factory=list)


@dataclass
class TermTier2Work:
    ranked: list[TermTier2OccurrenceRanking] = field(default_factory=list)
    report: Tier2Report | None = None
