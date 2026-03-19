from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.types import (
    StructuralReferenceDetectorResult,
)
from plainera_unacronym.nlp.extraction.engine.stages import (
    Chain,
    Stage,
    StageReport,
)
from plainera_unacronym.nlp.extraction.structural import stage_funcs as f
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolutionResult,
)


class StructuralReferenceResolutionFlow:
    """Run the end-to-end structural-reference extraction pipeline."""

    def __init__(
        self,
        det_cfg: object | None = None,
        ext_cfg: StructuralReferenceExtractionConfig | None = None,
    ):
        """Initialise the structural-reference extraction flow.

        Args:
            det_cfg: Detector configuration for structural-reference detection.
            ext_cfg: Extraction configuration for structural-reference
                canonicalisation and assembly. If ``None``, defaults to
                ``StructuralReferenceExtractionConfig()``.
        """
        self.det_cfg = det_cfg or object()
        self.ext_cfg = ext_cfg or StructuralReferenceExtractionConfig()

    @staticmethod
    def build_chain() -> Chain[StructuralFlowState]:
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

    def run(
        self,
        text: str,
    ) -> tuple[
        StructuralReferenceDetectorResult,
        StructuralReferenceResolutionResult,
        list[StageReport],
    ]:
        """Run structural-reference extraction over the provided text.

        Args:
            text: Source text to process.

        Returns:
            A tuple of detector result, assembled extraction result, and stage
            reports.
        """
        state = StructuralFlowState(
            text=text,
            det_cfg=self.det_cfg,
            ext_cfg=self.ext_cfg,
        )
        state, reports = self.build_chain().run(state)
        assert state.det_res is not None and state.extr is not None
        return state.det_res, state.extr, reports
