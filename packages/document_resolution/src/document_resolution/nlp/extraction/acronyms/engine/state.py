from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from document_resolution.nlp.common.types import (
    AcronymDetectorConfig,
    AcronymDetectorResult,
    ExtractedDefinition,
    ExtractionResult,
    InTextPick,
)
from document_resolution.nlp.detection.cleanup import DroppedOccurrence
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig
from document_resolution.nlp.extraction.tiers.types import Tier1Work, Tier2Work


@dataclass
class FlowState:
    text: str
    det_cfg: AcronymDetectorConfig
    ext_cfg: ExtractionConfig
    last_info: str = ""

    det_res: AcronymDetectorResult | None = None
    cleanup_dropped: list[DroppedOccurrence] = field(default_factory=list)
    picks: dict[str, InTextPick | None] = field(default_factory=dict)

    anchored_defs: list[ExtractedDefinition] = field(default_factory=list)
    harvested_defs: list[ExtractedDefinition] = field(default_factory=list)
    global_defs: list[ExtractedDefinition] = field(default_factory=list)
    backref_defs: list[ExtractedDefinition] = field(default_factory=list)
    all_defs: list[ExtractedDefinition] = field(default_factory=list)

    tier2_model: Any | None = None
    coverage: float = 0.0
    missing_keys: tuple[str, ...] = ()

    tier_1: Tier1Work = field(default_factory=Tier1Work, repr=False)
    tier_2: Tier2Work = field(default_factory=Tier2Work, repr=False)

    extr: ExtractionResult | None = None
