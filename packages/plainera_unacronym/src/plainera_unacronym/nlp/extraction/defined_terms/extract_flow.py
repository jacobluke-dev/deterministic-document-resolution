from __future__ import annotations

from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetectorResult
from plainera_unacronym.nlp.extraction.base.base_flow import BaseResolutionFlow
from plainera_unacronym.nlp.extraction.base.stages import Chain, Stage, StageReport
from plainera_unacronym.nlp.extraction.defined_terms import stage_funcs as f
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult


def _make_term_flow_state(
    text: str,
    det_cfg: DefinedTermDetectorConfig,
    ext_cfg: DefinedTermExtractionConfig,
) -> TermFlowState:
    return TermFlowState(
        text=text,
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
    )


class DefinedTermResolutionFlow(
    BaseResolutionFlow[
        TermFlowState,
        DefinedTermDetectorResult,
        TermResolutionResult,
        DefinedTermDetectorConfig,
        DefinedTermExtractionConfig,
    ]
):
    """Run the end-to-end defined-term resolution pipeline."""

    def __init__(
        self,
        det_cfg: DefinedTermDetectorConfig | None = None,
        ext_cfg: DefinedTermExtractionConfig | None = None,
        *,
        disambig_margin_threshold: float | None = None,
        trace: bool = False,
        trace_filter: str | None = None,
    ):
        """Initialise the defined-term resolution flow.

        Args:
            det_cfg: Detector configuration. If omitted, a default
                `DefinedTermDetectorConfig()` is used.
            ext_cfg: Extraction configuration. If omitted, a default
                `DefinedTermExtractionConfig()` is used.
            disambig_margin_threshold: Optional override for the
                disambiguation margin threshold.
        """
        if det_cfg is None:
            det_cfg = DefinedTermDetectorConfig()
        if ext_cfg is None:
            ext_cfg = DefinedTermExtractionConfig()

        super().__init__(
            state_factory=_make_term_flow_state,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
            trace_filter=trace_filter,
            trace=trace,
        )
        self._ovr_margin = disambig_margin_threshold

    @staticmethod
    def _n_term_keys(s: TermFlowState) -> int:
        return len(s.tier_1.term_meaning_index)

    @staticmethod
    def _n_occurrences(s: TermFlowState) -> int:
        if s.det_res is None:
            return 0
        return len(s.det_res.mentions)

    def build_chain(self) -> Chain[TermFlowState]:
        """Build the staged execution chain for the extraction pipeline.

        Returns:
            Chain[TermFlowState]: A chain of `Stage`s that transform a `TermFlowState`
            through detection, extraction, merge, gap-fill, and disambiguation.

        """

        return Chain(
            [
                Stage(
                    "detect_terms",
                    f.st_detect_terms,
                    lambda s: s.last_info,
                ),
                Stage(
                    "build_structure_index",
                    f.st_build_structure_index,
                    lambda s: s.last_info,
                    trace_fields=("structure_index",),
                ),
                Stage(
                    "extract_term_definitions",
                    f.st_extract_term_definitions,
                    lambda s: s.last_info,
                    trace_fields=("definition_entries",),
                ),
                Stage(
                    "build_term_meaning_index",
                    f.st_build_term_meaning_index,
                    lambda s: s.last_info,
                    trace_fields=("tier_1.term_meaning_index", "tier_1.meaning_index"),
                ),
                Stage(
                    "tier1_score_term_occurrences",
                    f.st_tier1_score_term_occurrences,
                    lambda s: s.last_info,
                    trace_fields=("tier_1.ranked",),
                ),
                Stage(
                    "tier2_term_semantic_rerank",
                    f.st_tier2_term_semantic_rerank,
                    lambda s: s.last_info,
                    trace_fields=("tier_2.report", "tier_2.ranked"),
                ),
                Stage(
                    "assemble_term_resolutions",
                    f.st_assemble_term_resolutions,
                    lambda s: s.last_info,
                ),
            ]
        )

    def _finalize(
        self,
        state: TermFlowState,
        reports: list[StageReport],
    ) -> tuple[DefinedTermDetectorResult, TermResolutionResult, list[StageReport]]:
        """Validate final structural flow state and return typed outputs."""
        assert state.det_res is not None and state.extr is not None
        self.trace_events = self._tracer.events if self._tracer else None
        return state.det_res, state.extr, reports
