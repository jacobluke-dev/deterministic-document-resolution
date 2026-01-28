from dataclasses import dataclass, field
from typing import Optional

from plainera_unacronym.nlp.detection.cleanup.post_detect_cleanup import DroppedOccurrence
from plainera_unacronym.nlp.common.types import (
    DetectorConfig, InTextPick, ExtractedDefinition, ExtractionResult, DetectorResult
)
from plainera_unacronym.nlp.extraction.config import ExtractionConfig

@dataclass
class FlowState:
    text: str
    det_cfg: DetectorConfig
    ext_cfg: ExtractionConfig
    _last_info: str = ""

    det_res: Optional[DetectorResult] = None
    cleanup_dropped: list[DroppedOccurrence] = field(default_factory=list)
    picks: dict[str, Optional[InTextPick]] = field(default_factory=dict)

    anchored_defs: list[ExtractedDefinition] = field(default_factory=list)
    harvested_defs: list[ExtractedDefinition] = field(default_factory=list)
    global_defs: list[ExtractedDefinition] = field(default_factory=list)
    backref_defs: list[ExtractedDefinition] = field(default_factory=list)
    all_defs: list[ExtractedDefinition] = field(default_factory=list)

    strategy: str = "anchored+harvest"
    coverage: float = 0.0
    missing_keys: tuple[str, ...] = ()

    extr: Optional[ExtractionResult] = None
