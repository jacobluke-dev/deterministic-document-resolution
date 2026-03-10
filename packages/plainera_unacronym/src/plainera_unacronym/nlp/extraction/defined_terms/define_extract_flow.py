from __future__ import annotations


from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetectorResult
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms import stage_funcs as f
from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult
from plainera_unacronym.nlp.extraction.engine.stages import Chain, Stage, StageReport, TraceEvent, Tracer



class DefinedTermsExtractionFlow:
    """Run the end-to-end defined terms detection and extraction pipeline.

    This orchestrates a staged workflow over a single input text

    The pipeline is executed via a `Chain` of `Stage`s, producing stage reports and
    optionally trace events for debugging.

    Attributes:
        det_cfg (DefinedTermDetectorConfig): Configuration used by the acronym detector.
        ext_cfg (DefinedTermExtractionConfig): Configuration used by extraction strategies.
        window_left (int): Characters to include to the left of a first occurrence
            when building the local anchored extraction window.
        window_right (int): Characters to include to the right of a first occurrence
            when building the local anchored extraction window.
        trace_events (list[TraceEvent] | None): Trace events captured during the last run
            when tracing is enabled; otherwise None.

    """

    def __init__(
        self,
        det_cfg: DefinedTermDetectorConfig | None = None,
        ext_cfg: DefinedTermExtractionConfig | None = None,
        *,
        window_left: int = 320,
        window_right: int = 280,
        # optional runtime overrides so we don't mutate frozen configs
        disambig_margin_threshold: float | None = None,
        trace: bool = False,
        trace_filter: str | None = None,
    ):
        """Initialize an ExtractionFlow.

        Args:
            det_cfg (DefinedTermDetectorConfig | None): Detector config. If None, then `DetectorConfig()`.
            ext_cfg (DefinedTermExtractionConfig | None): Extraction config. If None, then `DefinedTermExtractionConfig()`.
            window_left (int): Chars to include to the left of the first occurrence
                when performing anchored extraction.
            window_right (int): Chars to include to the right of the first occurrence
                when performing anchored extraction.
            trace (bool): If True, capture structured trace events for selected stage fields.
            trace_filter (str | None): Optional regex filter applied to acronym keys when tracing.
        """
        self.trace_events: list[TraceEvent] | None = None
        self.det_cfg = det_cfg or DefinedTermDetectorConfig()
        self.ext_cfg = ext_cfg or DefinedTermExtractionConfig()
        self.window_left = window_left
        self.window_right = window_right
        self._ovr_margin = disambig_margin_threshold
        self._tracer = Tracer(trace_filter) if trace else None



    def build_chain(self) -> Chain[TermFlowState]:
        """Build the staged execution chain for the extraction pipeline.

        Returns:
            Chain[TermFlowState]: A chain of `Stage`s that transform a `TermFlowState`
            through detection, extraction, merge, gap-fill, and disambiguation.

        """
        wl, wr = self.window_left, self.window_right

        return Chain(
            [
                Stage("detect_terms",
                      f.st_detect_terms,
                      lambda s: f"firsts={self._n_firsts(s)} dropped={len(s.cleanup_dropped)}"
                      ),
                Stage(
                    "build_term_sense_index",
                    f.st_build_term_sense_index,
                    lambda s: f"firsts={self._n_firsts(s)} dropped={len(s.cleanup_dropped)}",
                    trace_fields=("cleanup_dropped",),
                ),
                Stage(
                    "tier1_score_term_occurrences",
                    lambda s: f.st_tier1_score_term_occurrences(s, window_left=wl, window_right=wr),
                    lambda s: f"{sum(1 for v in s.picks.values() if v)}/{len(s.picks)}",
                    trace_fields=("picks",),
                ),
                Stage(
                    "tier2_term_semantic_rerank",
                    f.st_tier2_term_semantic_rerank,
                    lambda s: f"{len(s.harvested_defs)}",
                    trace_fields=("harvested_defs",),
                ),
                Stage("assemble_term_resolutions",
                      f.st_assemble_term_resolutions,
                      lambda s: f"{len(s.all_defs)}",
                      trace_fields=("all_defs",)),
            ]
        )

    def run(self, text: str) -> tuple[DefinedTermDetectorResult, TermResolutionResult, list[StageReport]]:
        """Run the pipeline over `text`.

        Args:
            text (str): Source document text.

        Returns:
            tuple[DefinedTermDetectorResult, TermResolutionResult, list[StageReport]]:
                - DetectorResult: Raw detector output after cleanup.
                - TermResolutionResult: Final extraction output (picks, defs, senses, resolutions).
                - list[StageReport]: Per-stage execution reports.

        Raises:
            AssertionError: If the pipeline completes without producing detector or extraction results.

        """
        state = TermFlowState(text=text, det_cfg=self.det_cfg, ext_cfg=self.ext_cfg)
        state, reports = self.build_chain().run(state, tracer=self._tracer)
        assert state.det_res and state.extr
        self.trace_events = self._tracer.events if self._tracer else None
        return state.det_res, state.extr, reports
