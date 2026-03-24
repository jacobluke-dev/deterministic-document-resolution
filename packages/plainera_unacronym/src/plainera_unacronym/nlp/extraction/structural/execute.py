from __future__ import annotations

from plainera_unacronym.nlp.extraction import run_flow_with_options
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceDetectorConfig,
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.extract_flow import StructuralReferenceResolutionFlow
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState


def detect_and_resolve_structural_references(
    text: str,
    *,
    det_cfg: StructuralReferenceDetectorConfig | None = None,
    ext_cfg: StructuralReferenceExtractionConfig | None = None,
    return_reports: bool = False,
    return_state: bool = False,
):
    """Run the full structural-reference detection + resolution pipeline.

    Args:
        text: Source document text to process.
        det_cfg: Optional structural detector config override. If ``None``, the
            flow default is used.
        ext_cfg: Optional structural extraction config override. If ``None``, the
            flow default is used.
        return_reports: If True, include per-stage ``StageReport`` objects in the
            return tuple.
        return_state: If True, include the final ``StructuralFlowState`` in the
            return tuple.

    Returns:
        The return shape depends on ``return_reports`` and ``return_state``:

        - Default:
            ``(det_res, extr)``

        - If ``return_reports``:
            ``(det_res, extr, reports)``

        - If ``return_state``:
            ``(det_res, extr, state)``

        - If ``return_state`` and ``return_reports``:
            ``(det_res, extr, reports, state)``
    """
    flow = StructuralReferenceResolutionFlow(
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
    )
    state = StructuralFlowState(
        text=text,
        det_cfg=flow.det_cfg,
        ext_cfg=flow.ext_cfg,
    )
    return run_flow_with_options(
        flow=flow,
        state=state,
        return_reports=return_reports,
        return_state=return_state,
        trace=False,
    )
