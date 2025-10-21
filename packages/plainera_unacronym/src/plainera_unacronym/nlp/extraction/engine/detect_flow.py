from dataclasses import dataclass, field
from typing import Optional

from .stages import Stage, StageResult, Chain, StageReport, Tracer
from .. import ExtractionConfig, extract_iter
from ..defs_utils import defs_from_picks, dedupe_defs
from ..extract_first_occ import extract_near_firsts
from ..harvest import harvest_defs_all
from ... import DetectorConfig, Detector
from ...common.types import OccurrenceLite, ExtractionResult, DetectorResult
from ...senses.disambiguate import disambiguate_occurrences
from ...senses.sense_build import build_senses

from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import InTextPick, ExtractedDefinition, FirstOccurrence


def _fill_missing_from_defs(
    text: str,
    *,
    firsts: dict[str, FirstOccurrence],
    det_cfg: DetectorConfig,
    defs: list[ExtractedDefinition],
) -> dict[str, Optional[InTextPick]]:
    dotted_mode = getattr(det_cfg, "dotted_display", "strip")
    index: dict[str, list[ExtractedDefinition]] = {}
    for d in defs:
        k = normalize_acronym_key(d.acronym, det_cfg.allow_chars, dotted_mode)
        if k:
            index.setdefault(k, []).append(d)

    fills: dict[str, Optional[InTextPick]] = {}
    for key, fo in firsts.items():
        cands = index.get(key, [])
        if not cands:
            fills[key] = None
            continue
        best = min(
            cands,
            key=lambda c: (abs(c.acr_start - fo.start_offset), -c.confidence, c.acr_start),
        )
        fills[key] = InTextPick(
            definition=best.definition,
            acr_span=(best.acr_start, best.acr_end),
            def_span=(best.def_start, best.def_end),
            confidence=best.confidence,
            original_definition=best.original_definition,
        )
    return fills


def extract_pipeline_iter(
    text: str,
    det_cfg: Optional[DetectorConfig],  # kept for signature parity / future use
    ext_cfg: ExtractionConfig,
    plan: object | None = None,  # reserved for future policy hooks
) -> list[ExtractedDefinition]:
    """
    Global free-scan extractor used by Stage E.

    Currently a thin adapter over existing `extract_iter`, returning all
    in-text definitions found anywhere in the document. Deduping is handled
    later by the merge stage.
    """
    return list(extract_iter(text, ext_cfg))


@dataclass
class FlowState:
    text: str
    det_cfg: DetectorConfig
    ext_cfg: ExtractionConfig

    det_res: Optional[DetectorResult] = None
    picks: dict[str, Optional[InTextPick]] = field(default_factory=dict)

    anchored_defs: list[ExtractedDefinition] = field(default_factory=list)
    harvested_defs: list[ExtractedDefinition] = field(default_factory=list)
    global_defs: list[ExtractedDefinition] = field(default_factory=list)
    all_defs: list[ExtractedDefinition] = field(default_factory=list)

    strategy: str = "anchored+harvest"
    coverage: float = 0.0
    missing_keys: tuple[str, ...] = ()

    extr: Optional[ExtractionResult] = None


class ExtractionFlow:
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
        self.trace_events = None
        self.det_cfg = det_cfg or DetectorConfig()
        self.ext_cfg = ext_cfg or ExtractionConfig()
        self.window_left = window_left
        self.window_right = window_right
        self._ovr_win = disambig_window_chars
        self._ovr_margin = disambig_margin_threshold
        self._tracer = Tracer(trace_filter) if trace else None

    # ---- stage methods: (FlowState) -> StageResult[FlowState] ----

    def _st_detect(self, s: FlowState) -> StageResult[FlowState]:
        det = Detector(config=s.det_cfg).detect(s.text)
        s.det_res = det
        s._last_info = f"firsts={len(det.unique_acronyms)} occs={len(det.occurrences)}"
        return StageResult(s, s._last_info)

    def _st_anchored(self, s: FlowState) -> StageResult[FlowState]:
        s.picks = extract_near_firsts(
            s.text, firsts=s.det_res.unique_acronyms, cfg=s.ext_cfg,
            window_left=self.window_left, window_right=self.window_right
        )
        got = sum(1 for v in s.picks.values() if v)
        s._last_info = f"anchored picks {got}/{len(s.picks)}"
        return StageResult(s, s._last_info)

    def _st_defs_from_picks(self, s: FlowState) -> StageResult[FlowState]:
        s.anchored_defs = defs_from_picks(s.text, s.picks)
        s._last_info = f"anchored defs={len(s.anchored_defs)}"
        return StageResult(s, s._last_info)

    def _st_harvest(self, s: FlowState) -> StageResult[FlowState]:
        s.harvested_defs = harvest_defs_all(s.text, s.det_res.occurrences, s.ext_cfg)
        s._last_info = f"harvested={len(s.harvested_defs)}"
        return StageResult(s, s._last_info)

    def _st_global(self, s: FlowState) -> StageResult[FlowState]:
        s.global_defs = extract_pipeline_iter(s.text, s.det_cfg, s.ext_cfg, plan=None)
        s._last_info = f"global={len(s.global_defs)}"
        return StageResult(s, s._last_info)

    def _st_merge(self, s: FlowState) -> StageResult[FlowState]:
        s.all_defs = dedupe_defs(s.anchored_defs + s.harvested_defs + s.global_defs)
        s._last_info = f"merged unique={len(s.all_defs)}"
        return StageResult(s, s._last_info)

    def _st_gapfill(self, s: FlowState) -> StageResult[FlowState]:
        missing = [k for k, v in s.picks.items() if v is None]
        if missing:
            fills = _fill_missing_from_defs(s.text, firsts=s.det_res.unique_acronyms, det_cfg=s.det_cfg,
                                            defs=s.all_defs)
            for k in missing:
                s.picks[k] = s.picks[k] or fills.get(k)
        s.strategy = "anchored+harvest" if not missing else "anchored+harvest+global-pipeline"
        s.coverage = (len(s.picks) - sum(1 for v in s.picks.values() if v is None)) / max(1, len(s.picks))
        s.missing_keys = tuple(sorted(k for k, v in s.picks.items() if v is None))
        s._last_info = f"{s.strategy} coverage={s.coverage:.2%} missing={len(s.missing_keys)}"
        return StageResult(s, s._last_info)

    def _st_senses_and_assemble(self, s: FlowState) -> StageResult[FlowState]:
        senses_by_acr = build_senses(s.all_defs)
        sense_index = {x.sense_id: x for xs in senses_by_acr.values() for x in xs}
        occs = [OccurrenceLite(o.acronym, o.start_offset, o.end_offset) for o in s.det_res.occurrences]

        # pull knobs from ext_cfg if we add a nested disambig config, else use overrides/defaults
        win_chars = self._ovr_win if self._ovr_win is not None else getattr(getattr(s.ext_cfg, "disambig", s.det_cfg),
                                                                            "window_chars", 320)
        margin = self._ovr_margin if self._ovr_margin is not None else getattr(getattr(s.ext_cfg, "disambig", None),
                                                                               "margin_threshold", 0.20)

        resolutions = disambiguate_occurrences(
            text=s.text, occurrences=occs, senses=senses_by_acr,
            window_chars=win_chars, margin_threshold=margin,
        )
        undecided = [r for r in resolutions if r.chosen_sense_id is None]
        ambiguous = tuple(sorted(k for k, v in senses_by_acr.items() if len(v) > 1))

        s.extr = ExtractionResult(
            picks=s.picks, definitions=s.all_defs, strategy=s.strategy, coverage=s.coverage,
            missing_keys=s.missing_keys, senses_by_acronym=senses_by_acr, sense_index=sense_index,
            resolutions=resolutions, ambiguous_keys=ambiguous, undecided=undecided,
        )
        s._last_info = f"senses={sum(len(v) for v in senses_by_acr.values())}, undecided={len(undecided)}"
        return StageResult(s, s._last_info)

    # Build a Chain using bound methods
    def build_chain(self) -> Chain[FlowState, FlowState]:
        return Chain([
            Stage("detect",
                  self._st_detect,
                  lambda s: f"firsts={len(s.det_res.unique_acronyms)}"),
            Stage("anchored_picks",
                  self._st_anchored,
                  lambda s: f"{sum(1 for v in s.picks.values() if v)}/{len(s.picks)}",
                  trace_fields=("picks",)),
            Stage("defs_from_picks",
                  self._st_defs_from_picks,
                  lambda s: f"{len(s.anchored_defs)}",
                  trace_fields=("anchored_defs",)),
            Stage("harvest",
                  self._st_harvest,
                  lambda s: f"{len(s.harvested_defs)}",
                  trace_fields=("harvested_defs",)),
            Stage("global_pipeline",
                  self._st_global,
                  lambda s: f"{len(s.global_defs)}",
                  trace_fields=("global_defs",)),
            Stage("merge_dedupe",
                  self._st_merge,
                  lambda s: f"{len(s.all_defs)}",
                  trace_fields=("all_defs",)),
            Stage("gap_fill_picks",
                  self._st_gapfill,
                  lambda s: f"cov={s.coverage:.0%} miss={len(s.missing_keys)}",
                  trace_fields=("picks",)),
            Stage("senses_disambiguate",
                  self._st_senses_and_assemble,
                  lambda s: "ready"),
        ])

    def run(self, text: str) -> tuple[DetectorResult, ExtractionResult, list[StageReport]]:
        state = FlowState(text=text, det_cfg=self.det_cfg, ext_cfg=self.ext_cfg)
        chain = self.build_chain()
        state, reports = chain.run(state, tracer=self._tracer)   # <<< tracer passed once
        assert state.det_res and state.extr
        self.trace_events = self._tracer.events if self._tracer else []  # expose for tests
        return state.det_res, state.extr, reports
