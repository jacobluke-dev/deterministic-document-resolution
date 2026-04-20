from __future__ import annotations

from dataclasses import dataclass, field

from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.detection.defined_terms import DefinedTermDetectorResult
from document_resolution.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from document_resolution.nlp.extraction.defined_terms.structure import TermStructureIndex
from document_resolution.nlp.extraction.defined_terms.tiers.work import TermTier1Work, TermTier2Work
from document_resolution.nlp.extraction.defined_terms.types import (
    TermDefinitionEntry,
    TermMeaning,
    TermResolutionResult,
)


@dataclass
class TermFlowState:
    """Mutable state container for the defined-term resolution pipeline.
    """

    text: str
    det_cfg: DefinedTermDetectorConfig
    ext_cfg: DefinedTermExtractionConfig
    last_info: str = ""

    det_res: DefinedTermDetectorResult | None = None

    term_meaning_index: dict[str, tuple[TermMeaning, ...]] = field(default_factory=dict)
    structure_index: TermStructureIndex | None = None
    definition_entries: list[TermDefinitionEntry] = field(default_factory=list)

    tier_1: TermTier1Work = field(default_factory=TermTier1Work, repr=False)
    tier_2: TermTier2Work = field(default_factory=TermTier2Work, repr=False)

    extr: TermResolutionResult | None = None
