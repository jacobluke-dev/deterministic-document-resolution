from __future__ import annotations

from dataclasses import dataclass, field

from document_resolution.nlp.detection.structural.types import StructuralReferenceDetectorResult
from document_resolution.nlp.extraction.structural.config import StructuralReferenceExtractionConfig
from document_resolution.nlp.extraction.structural.types import (
    StructuralAnchor,
    StructuralReferenceEntry,
    StructuralReferenceLink,
    StructuralReferenceResolutionResult,
)


@dataclass
class StructuralFlowState:
    text: str
    det_cfg: object
    ext_cfg: StructuralReferenceExtractionConfig

    det_res: StructuralReferenceDetectorResult | None = None
    reference_entries: list[StructuralReferenceEntry] = field(default_factory=list)
    extr: StructuralReferenceResolutionResult | None = None
    anchors: list[StructuralAnchor] = field(default_factory=list)
    anchor_index: dict[str, list[StructuralAnchor]] = field(default_factory=dict)
    link_entries: list[StructuralReferenceLink] = field(default_factory=list)

    last_info: str = ""
