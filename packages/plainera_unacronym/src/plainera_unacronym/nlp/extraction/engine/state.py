from dataclasses import dataclass, field
from typing import Optional

from plainera_unacronym.nlp.common.types import (
    DetectorConfig,
    DetectorResult,
    ExtractedDefinition,
    ExtractionResult,
    InTextPick,
)
from plainera_unacronym.nlp.detection.cleanup import DroppedOccurrence
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.tiers.types import DisambigWork


@dataclass
class FlowState:
    text: str
    det_cfg: DetectorConfig
    ext_cfg: ExtractionConfig
    last_info: str = ""

    det_res: Optional[DetectorResult] = None
    cleanup_dropped: list[DroppedOccurrence] = field(default_factory=list)
    picks: dict[str, Optional[InTextPick]] = field(default_factory=dict)

    anchored_defs: list[ExtractedDefinition] = field(default_factory=list)
    harvested_defs: list[ExtractedDefinition] = field(default_factory=list)
    global_defs: list[ExtractedDefinition] = field(default_factory=list)
    backref_defs: list[ExtractedDefinition] = field(default_factory=list)
    all_defs: list[ExtractedDefinition] = field(default_factory=list)

    coverage: float = 0.0
    missing_keys: tuple[str, ...] = ()

    disambig: DisambigWork = field(default_factory=DisambigWork, repr=False)

    extr: Optional[ExtractionResult] = None
