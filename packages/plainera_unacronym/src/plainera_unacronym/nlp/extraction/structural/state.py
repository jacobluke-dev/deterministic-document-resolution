from __future__ import annotations

from dataclasses import dataclass, field

from plainera_unacronym.nlp.detection.structural.types import StructuralReferenceDetectorResult
from plainera_unacronym.nlp.extraction.structural.config import StructuralReferenceExtractionConfig
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolution,
    StructuralReferenceResolutionResult,
)


@dataclass
class StructuralFlowState:
    text: str
    det_cfg: object
    ext_cfg: StructuralReferenceExtractionConfig

    det_res: StructuralReferenceDetectorResult | None = None
    resolution_entries: list[StructuralReferenceResolution] = field(default_factory=list)
    extr: StructuralReferenceResolutionResult | None = None

    last_info: str = ""
