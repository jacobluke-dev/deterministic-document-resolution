from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.detector import StructuralReferenceDetector
from plainera_unacronym.nlp.extraction.engine.stages import StageResult
from plainera_unacronym.nlp.extraction.structural.assemble import (
    assemble_structural_reference_resolution_result,
)
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


def st_build_structural_reference_resolutions(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Build structural-reference resolution entries from detector output."""
    assert s.det_res is not None

    s.resolution_entries = build_structural_reference_resolutions(
        references=s.det_res.references,
        cfg=s.ext_cfg,
    )

    s.last_info = (
        f"references={len(s.det_res.references)} "
        f"resolution_entries={len(s.resolution_entries)}"
    )
    return StageResult(s, s.last_info)


def st_assemble_structural_reference_resolutions(s: StructuralFlowState) -> StageResult[StructuralFlowState]:
    """Assemble final structural-reference resolution output."""
    s.extr = assemble_structural_reference_resolution_result(s)
    s.last_info = (
        f"references={len(s.extr.references)} "
        f"unique_keys={len(s.extr.unique_keys)}"
    )
    return StageResult(s, s.last_info)
