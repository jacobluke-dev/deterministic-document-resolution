from __future__ import annotations

from collections import Counter
from typing import Optional

from plainera_unacronym.nlp.common.types import DetectorConfig, DetectorResult, ExtractionResult
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.engine import stage_funcs as f
from plainera_unacronym.nlp.extraction.engine.stages import Chain, Stage, StageReport, TraceEvent, Tracer
from plainera_unacronym.nlp.extraction.engine.state import FlowState


class ExtractionFlow:
    """Run the end-to-end acronym detection and extraction pipeline.

    This orchestrates a staged workflow over a single input text:
      1) Detect acronym occurrences and first occurrences.
      2) Apply post-detection cleanup to remove/adjust problematic occurrences.
      3) Extract local (anchored) definitions near first occurrences.
      4) Harvest additional definition candidates across the document.
      5) Extract sentence back-references (definition appears earlier, acronym appears later).
      6) Merge and de-duplicate extracted definitions.
      7) Gap-fill missing picks using extracted definitions.
      8) Build senses and disambiguate occurrences.

    The pipeline is executed via a `Chain` of `Stage`s, producing stage reports and
    optionally trace events for debugging.

    Attributes:
        det_cfg (DetectorConfig): Configuration used by the acronym detector.
        ext_cfg (ExtractionConfig): Configuration used by extraction strategies.
        window_left (int): Characters to include to the left of a first occurrence
            when building the local anchored extraction window.
        window_right (int): Characters to include to the right of a first occurrence
            when building the local anchored extraction window.
        trace_events (list[TraceEvent] | None): Trace events captured during the last run
            when tracing is enabled; otherwise None.

    """

    def __init__(
        self,
        det_cfg: DetectorConfig | None = None,
        ext_cfg: ExtractionConfig | None = None,
        *,
        window_left: int = 320,
        window_right: int = 280,
        # optional runtime overrides so we don't mutate frozen configs
        disambig_window_chars: int | None = None,
        disambig_margin_threshold: float | None = None,
        trace: bool = False,
        trace_filter: str | None = None,
    ):
        """Initialize an ExtractionFlow.

        Args:
            det_cfg (DetectorConfig | None): Detector config. If None, then `DetectorConfig()`.
            ext_cfg (ExtractionConfig | None): Extraction config. If None, then `ExtractionConfig()`.
            window_left (int): Chars to include to the left of the first occurrence
                when performing anchored extraction.
            window_right (int): Chars to include to the right of the first occurrence
                when performing anchored extraction.
            disambig_window_chars (int | None): Optional runtime override for the
                disambiguation context window size (in chars). If None, uses config defaults.
            disambig_margin_threshold (float | None): Optional runtime override for the
                disambiguation margin threshold. If None, uses config defaults.
            trace (bool): If True, capture structured trace events for selected stage fields.
            trace_filter (str | None): Optional regex filter applied to acronym keys when tracing.
        """
        self.trace_events: Optional[list[TraceEvent]] = None
        self.det_cfg = det_cfg or DetectorConfig()
        self.ext_cfg = ext_cfg or ExtractionConfig()
        self.window_left = window_left
        self.window_right = window_right
        self._ovr_win = disambig_window_chars
        self._ovr_margin = disambig_margin_threshold
        self._tracer = Tracer(trace_filter) if trace else None

    @staticmethod
    def _n_firsts(s: FlowState) -> int:
        return len(s.det_res.unique_acronyms) if s.det_res is not None else 0

    def build_chain(self) -> Chain[FlowState]:
        """Build the staged execution chain for the extraction pipeline.

        Returns:
            Chain[FlowState]: A chain of `Stage`s that transform a `FlowState`
            through detection, extraction, merge, gap-fill, and disambiguation.

        """
        wl, wr = self.window_left, self.window_right

        # compute disambig knobs once here (engine concern)
        def _win(s: FlowState) -> int:
            if self._ovr_win is not None:
                return self._ovr_win
            dis = getattr(s.ext_cfg, "disambig", None)
            return int(getattr(dis, "window_chars", 320))

        def _t2_win(s: FlowState) -> int:
            t2 = getattr(s.ext_cfg, "tier2", None)
            v = getattr(t2, "context_window_chars", None)
            return int(v) if v is not None else _win(s)

        def _margin(s: FlowState) -> float:
            if self._ovr_margin is not None:
                return self._ovr_margin
            dis = getattr(s.ext_cfg, "disambig", None)
            return float(getattr(dis, "margin_threshold", 0.20))

        def _t1_margin(s: FlowState) -> float:
            dis = getattr(s.ext_cfg, "disambig", None)
            return float(getattr(dis, "margin_threshold", 0.20))

        def _t2_ceiling(s: FlowState) -> float:
            t2 = getattr(s.ext_cfg, "tier2", None)
            return float(getattr(t2, "auto_margin_ceiling", 0.75))

        def _t2_select_margin(s: FlowState) -> float:
            t2 = getattr(s.ext_cfg, "tier2", None)
            return float(getattr(t2, "select_margin_threshold", _margin(s)))

        return Chain(
            [
                Stage("detect", f.st_detect, lambda s: f"firsts={self._n_firsts(s)} dropped={len(s.cleanup_dropped)}"),
                Stage(
                    "post_detect_cleanup",
                    f.st_post_detect_cleanup,
                    lambda s: f"firsts={self._n_firsts(s)} dropped={len(s.cleanup_dropped)}",
                    trace_fields=("cleanup_dropped",),
                ),
                Stage(
                    "picks_first_occurrence_anchored",
                    lambda s: f.st_picks_first_occurrence_anchored(s, window_left=wl, window_right=wr),
                    lambda s: f"{sum(1 for v in s.picks.values() if v)}/{len(s.picks)}",
                    trace_fields=("picks",),
                ),
                Stage(
                    "defs_from_first_occurrence_picks",
                    f.st_defs_from_first_occurrence_picks,
                    lambda s: f"{len(s.anchored_defs)}",
                    trace_fields=("anchored_defs",),
                ),
                # IMPORTANT: required for multi-sense acronyms when later occurrences introduce new definitions.
                Stage(
                    "defs_scan_all_occurrences",
                    f.st_defs_scan_all_occurrences,
                    lambda s: f"{len(s.harvested_defs)}",
                    trace_fields=("harvested_defs",),
                ),
                Stage(
                    "sentence_backref",
                    f.st_sentence_backref,
                    lambda s: f"{len(s.backref_defs)}",
                    trace_fields=("sentence_backref",),
                ),
                Stage("merge_dedupe", f.st_merge, lambda s: f"{len(s.all_defs)}", trace_fields=("all_defs",)),
                Stage(
                    "finalise_picks",
                    f.st_finalise_picks,
                    lambda s: (
                        f"cov={s.coverage:.0%} miss={len(s.missing_keys)} "
                        f"by_route={dict(sorted(Counter(p.route for p in s.picks.values() if p).items()))}"
                    ),
                    trace_fields=("picks",),
                ),
                Stage(
                    "tier1_build_senses",
                    f.st_tier1_build_senses,
                    lambda s: s.last_info,
                ),
                Stage(
                    "tier1_score_occurrences",
                    lambda s: f.st_tier1_score_occurrences(
                        s,
                        window_chars=_win(s),
                        margin_threshold=_t1_margin(s)
                    ),
                    lambda s: s.last_info,
                    trace_fields=("disambig.tier1.ranked",)
                ),
                Stage(
                    "tier2_semantic_rerank",
                    lambda s: f.st_tier2_semantic_rerank(
                        s,
                        window_chars=_t2_win(s),
                        auto_margin_ceiling=_t2_ceiling(s)
                    ),
                    lambda s: s.last_info,
                    trace_fields=("disambig.tier2.report", "disambig.tier2.ranked")
                ),
                Stage(
                    "tiers_select_and_assemble",
                    lambda s: f.st_tiers_select_and_assemble(
                        s,
                        margin_threshold=_t2_select_margin(s)
                    ),
                    lambda s: "ready",
                ),
            ]
        )

    def run(self, text: str) -> tuple[DetectorResult, ExtractionResult, list[StageReport]]:
        """Run the pipeline over `text`.

        Args:
            text (str): Source document text.

        Returns:
            tuple[DetectorResult, ExtractionResult, list[StageReport]]:
                - DetectorResult: Raw detector output after cleanup.
                - ExtractionResult: Final extraction output (picks, defs, senses, resolutions).
                - list[StageReport]: Per-stage execution reports.

        Raises:
            AssertionError: If the pipeline completes without producing detector or extraction results.

        """
        state = FlowState(text=text, det_cfg=self.det_cfg, ext_cfg=self.ext_cfg)
        state, reports = self.build_chain().run(state, tracer=self._tracer)
        assert state.det_res and state.extr
        self.trace_events = self._tracer.events if self._tracer else None
        return state.det_res, state.extr, reports
