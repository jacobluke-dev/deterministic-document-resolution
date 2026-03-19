from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.types import (
    StructuralReferenceDetectorResult,
)
from plainera_unacronym.nlp.extraction.engine.base_flow import BaseResolutionFlow
from plainera_unacronym.nlp.extraction.engine.stages import (
    Chain,
    Stage,
    StageReport,
)
from plainera_unacronym.nlp.extraction.structural import stage_funcs as f
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig, StructuralReferenceDetectorConfig,
)
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolutionResult,
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
            state_cls=StructuralFlowState,
            det_cfg=det_cfg or StructuralReferenceDetectorConfig(),
            ext_cfg=ext_cfg or StructuralReferenceExtractionConfig(),
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
                    "build_structural_reference_resolutions",
                    f.st_build_structural_reference_resolutions,
                    lambda s: s.last_info,
                    trace_fields=("resolution_entries",),
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
