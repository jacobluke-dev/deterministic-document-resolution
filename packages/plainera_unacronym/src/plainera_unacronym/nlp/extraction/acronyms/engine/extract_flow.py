from __future__ import annotations

from collections import Counter

from plainera_unacronym.nlp.common.types import AcronymDetectorConfig, AcronymDetectorResult, ExtractionResult
from plainera_unacronym.nlp.extraction.acronyms.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.engine import stage_funcs as f
from plainera_unacronym.nlp.extraction.acronyms.engine.state import FlowState

from plainera_unacronym.nlp.extraction import BaseResolutionFlow

from plainera_unacronym.nlp.extraction import (
    Chain,
    Stage,
    StageReport,
)


def _make_term_flow_state(
    text: str,
    det_cfg: AcronymDetectorConfig,
    ext_cfg: ExtractionConfig,
) -> FlowState:
    return FlowState(
        text=text,
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
    )


class ExtractionFlow(
    BaseResolutionFlow[
        FlowState,
        AcronymDetectorResult,
        ExtractionResult,
        AcronymDetectorConfig,
        ExtractionConfig,
    ]
):
    """Run the end-to-end acronym detection and extraction pipeline.

    This orchestrates a staged workflow over a single input text:
      1) Detect acronym occurrences and first occurrences.
      2) Apply post-detection cleanup to remove/adjust problematic occurrences.
      3) Extract local (anchored) definitions near first occurrences.
      4) Harvest additional definition candidates across the document.
      5) Extract sentence back-references (definition appears earlier, acronym appears later).
      6) Merge and de-duplicate extracted definitions.
      7) Gap-fill missing picks using extracted definitions.
      8) Build meanings and disambiguate occurrences.

    The pipeline is executed via a `Chain` of `Stage`s, producing stage reports and
    optionally trace events for debugging.

    Attributes:
        det_cfg (AcronymDetectorConfig): Configuration used by the acronym detector.
        ext_cfg (ExtractionConfig): Configuration used by extraction strategies.
        window_left (int): Characters to include to the left of a first occurrence
            when building the local anchored extraction window.
        window_right (int): Characters to include to the right of a first occurrence
            when building the local anchored extraction window.
    """

    def __init__(
        self,
        det_cfg: AcronymDetectorConfig | None = None,
        ext_cfg: ExtractionConfig | None = None,
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
            det_cfg (AcronymDetectorConfig | None): Detector config. If None, then `DetectorConfig()`.
            ext_cfg (ExtractionConfig | None): Extraction config. If None, then `ExtractionConfig()`.
            window_left (int): Chars to include to the left of the first occurrence
                when performing anchored extraction.
            window_right (int): Chars to include to the right of the first occurrence
                when performing anchored extraction.
        """
        super().__init__(
            state_factory=_make_term_flow_state,
            det_cfg=det_cfg or AcronymDetectorConfig(),
            ext_cfg=ext_cfg or ExtractionConfig(),
            trace_filter=trace_filter,
            trace=trace,
        )
        self.window_left = window_left
        self.window_right = window_right
        self._ovr_margin = disambig_margin_threshold

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

        def _t1_margin(s: FlowState) -> float:
            return float(getattr(s.ext_cfg, "tier_1_margin_threshold", 0.20))

        def _t1_window_chars(s: FlowState) -> int:
            return int(getattr(s.ext_cfg, "tier_1_window_chars", 140))

        def _t2_ceiling(s: FlowState) -> float:
            t2 = getattr(s.ext_cfg, "tier2", None)
            return float(getattr(t2, "auto_margin_ceiling", 0.75))

        def _multi_select_margin(s: FlowState) -> float:
            multi = getattr(s.ext_cfg, "multi_tier", None)
            return float(getattr(multi, "select_margin_threshold", 0.10))

        return Chain(
            [
                # DETECTION STAGES
                Stage("detect", f.st_detect, lambda s: f"firsts={self._n_firsts(s)} dropped={len(s.cleanup_dropped)}"),
                Stage(
                    "post_detect_cleanup",
                    f.st_post_detect_cleanup,
                    lambda s: f"firsts={self._n_firsts(s)} dropped={len(s.cleanup_dropped)}",
                    trace_fields=("cleanup_dropped",),
                ),
                # TIER 1 STAGES
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
                # IMPORTANT: required for multi-meaning acronyms when later occurrences introduce new definitions.
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
                    "tier1_build_meanings",
                    f.st_tier1_build_meanings,
                    lambda s: s.last_info,
                ),
                Stage(
                    "tier1_score_occurrences",
                    lambda s: f.st_tier1_score_occurrences(
                        s,
                        window_chars=_t1_window_chars(s),
                        margin_threshold=_t1_margin(s),
                    ),
                    lambda s: s.last_info,
                    trace_fields=("tier_1.ranked",),
                ),
                # TIER 2 STAGES
                Stage(
                    "tier2_semantic_rerank",
                    lambda s: f.st_tier2_semantic_rerank(
                        s,
                        auto_margin_ceiling=_t2_ceiling(s),
                    ),
                    lambda s: s.last_info,
                    trace_fields=("tier_2.report", "tier_2.ranked"),
                ),
                # 'MERGING' OF THE TIERS
                Stage(
                    "tiers_select_and_assemble",
                    lambda s: f.st_tiers_select_and_assemble(s, margin_threshold=_multi_select_margin(s)),
                    lambda s: "ready",
                ),
            ]
        )

    def _finalize(
        self,
        state: FlowState,
        reports: list[StageReport],
    ) -> tuple[AcronymDetectorResult, ExtractionResult, list[StageReport]]:
        """Validate final acronym flow state and return typed outputs."""
        assert state.det_res is not None and state.extr is not None
        self.trace_events = self._tracer.events if self._tracer else None
        return state.det_res, state.extr, reports
