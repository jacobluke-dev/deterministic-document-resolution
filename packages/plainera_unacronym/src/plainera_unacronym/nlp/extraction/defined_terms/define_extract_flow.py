from __future__ import annotations

from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetectorResult
from plainera_unacronym.nlp.extraction.defined_terms import stage_funcs as f
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult
from plainera_unacronym.nlp.extraction.engine.stages import Chain, Stage, StageReport, TraceEvent, Tracer


class DefinedTermResolutionFlow:
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
        """Initialize an ExtractionFlow.

        Args:
            det_cfg (DefinedTermDetectorConfig | None): Detector config. If None, then `DetectorConfig()`.
            ext_cfg (DefinedTermExtractionConfig | None): Extraction config.
            If None, then `DefinedTermExtractionConfig()`.
                when performing anchored extraction.
                when performing anchored extraction.
            trace (bool): If True, capture structured trace events for selected stage fields.
            trace_filter (str | None): Optional regex filter applied to acronym keys when tracing.
        """
        self.trace_events: list[TraceEvent] | None = None
        self.det_cfg = det_cfg or DefinedTermDetectorConfig()
        self.ext_cfg = ext_cfg or DefinedTermExtractionConfig()
        self._ovr_margin = disambig_margin_threshold
        self._tracer = Tracer(trace_filter) if trace else None

    @staticmethod
    def _n_term_keys(s: TermFlowState) -> int:
        return len(s.tier_1.term_sense_index)

    @staticmethod
    def _n_occurrences(s: TermFlowState) -> int:
        if s.det_res is None:
            return 0
        return len(s.det_res.mentions)

    @staticmethod
    def build_chain() -> Chain[TermFlowState]:
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
                    "build_term_sense_index",
                    f.st_build_term_sense_index,
                    lambda s: s.last_info,
                    trace_fields=("tier_1.term_sense_index", "tier_1.sense_index"),
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

    def run(self, text: str) -> tuple[DefinedTermDetectorResult, TermResolutionResult, list[StageReport]]:
        state = TermFlowState(
            text=text,
            det_cfg=self.det_cfg,
            ext_cfg=self.ext_cfg,
        )
        state, reports = self.build_chain().run(state, tracer=self._tracer)
        assert state.det_res and state.extr
        self.trace_events = self._tracer.events if self._tracer else None
        return state.det_res, state.extr, reports
