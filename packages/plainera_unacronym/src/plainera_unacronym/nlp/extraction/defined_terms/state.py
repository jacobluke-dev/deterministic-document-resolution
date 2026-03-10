from __future__ import annotations

from dataclasses import dataclass, field

from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetectorResult
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult, TermSense
from plainera_unacronym.nlp.extraction.defined_terms.work import TermTier1Work, TermTier2Work


@dataclass
class TermFlowState:
    text: str
    det_cfg: DefinedTermDetectorConfig
    ext_cfg: DefinedTermExtractionConfig
    last_info: str = ""

    det_res: DefinedTermDetectorResult | None = None

    term_sense_index: dict[str, tuple[TermSense, ...]] = field(default_factory=dict)

    tier_1: TermTier1Work = field(default_factory=TermTier1Work, repr=False)
    tier_2: TermTier2Work = field(default_factory=TermTier2Work, repr=False)

    extr: TermResolutionResult | None = None
