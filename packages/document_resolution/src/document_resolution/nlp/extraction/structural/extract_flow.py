from __future__ import annotations

from document_resolution.nlp.detection.structural.types import (
    StructuralReferenceDetectorResult,
)
from document_resolution.nlp.extraction import (
    BaseResolutionFlow,
    Chain,
    Stage,
    StageReport,
)
from document_resolution.nlp.extraction.structural import stage_funcs as f
from document_resolution.nlp.extraction.structural.config import (
    StructuralReferenceDetectorConfig,
    StructuralReferenceExtractionConfig,
)
from document_resolution.nlp.extraction.structural.state import StructuralFlowState
from document_resolution.nlp.extraction.structural.types import (
    StructuralReferenceResolutionResult,
)


def _make_term_flow_state(
    text: str,
    det_cfg: StructuralReferenceDetectorConfig,
    ext_cfg: StructuralReferenceExtractionConfig,
) -> StructuralFlowState:
    return StructuralFlowState(
        text=text,
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
    )


class StructuralReferenceResolutionFlow(
    BaseResolutionFlow[
        StructuralFlowState,
        StructuralReferenceDetectorResult,
        StructuralReferenceResolutionResult,
        StructuralReferenceDetectorConfig,
        StructuralReferenceExtractionConfig,
    ]
):
    """Run the end-to-end structural-reference extraction pipeline."""

    def __init__(
        self,
        det_cfg: StructuralReferenceDetectorConfig | None = None,
        ext_cfg: StructuralReferenceExtractionConfig | None = None,
        trace: bool = False,
        trace_filter: str | None = None,
    ):
        """Initialise the structural-reference extraction flow.

        Args:
            det_cfg: Detector configuration for structural-reference detection.
                If ``None``, defaults to ``StructuralReferenceDetectorConfig()``.
            ext_cfg: Extraction configuration for structural-reference
                canonicalisation and assembly. If ``None``, defaults to
                ``StructuralReferenceExtractionConfig()``.
        """
        super().__init__(
            state_factory=_make_term_flow_state,
            det_cfg=det_cfg or StructuralReferenceDetectorConfig(),
            ext_cfg=ext_cfg or StructuralReferenceExtractionConfig(),
            trace_filter=trace_filter,
            trace=trace,
        )

    def build_chain(self) -> Chain[StructuralFlowState]:
        """Build the staged execution chain for structural-reference extraction."""
        return Chain(
            [
                Stage(
                    "detect_structural_references",
                    f.st_detect_structural_references,
                    lambda s: s.last_info,
                ),
                Stage(
                    "build_structural_reference_entries",
                    f.st_build_structural_reference_entries,
                    lambda s: s.last_info,
                    trace_fields=("reference_entries",),
                ),
                Stage(
                    "extract_structural_anchors",
                    f.st_extract_structural_anchors,
                    lambda s: s.last_info,
                    trace_fields=("anchors",),
                ),
                Stage(
                    "build_structural_anchor_index",
                    f.st_build_structural_anchor_index,
                    lambda s: s.last_info,
                    trace_fields=("anchor_index",),
                ),
                Stage(
                    "link_structural_references",
                    f.st_link_structural_references,
                    lambda s: s.last_info,
                    trace_fields=("link_entries",),
                ),
                Stage(
                    "assemble_structural_reference_resolutions",
                    f.st_assemble_structural_reference_resolutions,
                    lambda s: s.last_info,
                ),
            ]
        )

    def _finalize(
        self,
        state: StructuralFlowState,
        reports: list[StageReport],
    ) -> tuple[
        StructuralReferenceDetectorResult,
        StructuralReferenceResolutionResult,
        list[StageReport],
    ]:
        """Validate final structural flow state and return typed outputs."""
        assert state.det_res is not None and state.extr is not None
        return state.det_res, state.extr, reports
