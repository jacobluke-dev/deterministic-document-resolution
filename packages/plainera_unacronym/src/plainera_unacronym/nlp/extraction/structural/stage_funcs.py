from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.detector import StructuralReferenceDetector
from plainera_unacronym.nlp.extraction.base.stages import StageResult
from plainera_unacronym.nlp.extraction.structural.anchor import extract_structural_anchors, \
    build_structural_anchor_index
from plainera_unacronym.nlp.extraction.structural.assemble import (
    assemble_structural_reference_resolution_result,
)
from plainera_unacronym.nlp.extraction.structural.link import build_structural_reference_links
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.transform import (
    build_structural_reference_resolutions,
)


def st_detect_structural_references(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Run structural-reference detection and store the detector result."""
    det = StructuralReferenceDetector(config=s.det_cfg).detect(s.text)
    s.det_res = det
    s.last_info = f"references={len(det.references)}"
    return StageResult(s, s.last_info)


def st_build_structural_reference_entries(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Build structural-reference resolution entries from detector output."""
    assert s.det_res is not None

    s.resolution_entries = build_structural_reference_resolutions(
        references=s.det_res.references,
        cfg=s.ext_cfg,
    )

    s.last_info = f"references={len(s.det_res.references)} " f"resolution_entries={len(s.resolution_entries)}"
    return StageResult(s, s.last_info)


def st_extract_structural_anchors(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Extract heading-like structural anchors from the source text."""
    s.anchors = extract_structural_anchors(
        text=s.text,
        cfg=s.ext_cfg,
    )
    s.last_info = f"anchors={len(s.anchors)}"
    return StageResult(s, s.last_info)


def st_build_structural_anchor_index(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Build lookup index for extracted structural anchors."""
    s.anchor_index = build_structural_anchor_index(s.anchors)
    s.last_info = (
        f"anchors={len(s.anchors)} "
        f"anchor_keys={len(s.anchor_index)}"
    )
    return StageResult(s, s.last_info)


def st_link_structural_references(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Resolve structural reference entries to indexed structural anchors."""
    s.link_entries = build_structural_reference_links(
        references=s.resolution_entries,
        anchor_index=s.anchor_index,
    )

    resolved = sum(1 for link in s.link_entries if link.target_span is not None)
    unresolved = len(s.link_entries) - resolved

    s.last_info = (
        f"references={len(s.resolution_entries)} "
        f"links={len(s.link_entries)} "
        f"resolved={resolved} "
        f"unresolved={unresolved}"
    )
    return StageResult(s, s.last_info)


def st_assemble_structural_reference_resolutions(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Assemble final structural-reference resolution output."""
    s.extr = assemble_structural_reference_resolution_result(s)
    s.last_info = (
        f"references={len(s.extr.references)} "
        f"links={len(s.extr.links)} "
        f"unique_keys={len(s.extr.unique_keys)}"
    )
    return StageResult(s, s.last_info)
