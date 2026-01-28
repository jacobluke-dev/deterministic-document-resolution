from plainera_unacronym.nlp import Detector
from plainera_unacronym.nlp.detection.cleanup.post_detect_cleanup import post_detect_cleanup
from plainera_unacronym.nlp.extraction.anchored.extract import extract_near_firsts
from plainera_unacronym.nlp.extraction.backref.extract import extract_sentence_backrefs
from plainera_unacronym.nlp.extraction.core.defs import defs_from_picks, dedupe_defs
from plainera_unacronym.nlp.extraction.strategies.harvest import harvest_defs_all
from plainera_unacronym.nlp.extraction.strategies.gapfill import fill_missing_from_defs
from plainera_unacronym.nlp.senses.disambiguate import disambiguate_occurrences
from plainera_unacronym.nlp.senses.sense_build import build_senses
from plainera_unacronym.nlp.common.types import OccurrenceLite, ExtractionResult
from .stages import StageResult
from .state import FlowState

def st_detect(s: FlowState) -> StageResult[FlowState]:
    det = Detector(config=s.det_cfg).detect(s.text)
    s.det_res = det
    s._last_info = f"firsts={len(det.unique_acronyms)} occs={len(det.occurrences)}"
    return StageResult(s, s._last_info)

def st_post_detect_cleanup(s: FlowState) -> StageResult[FlowState]:
    assert s.det_res is not None
    cleaned, summary, dropped = post_detect_cleanup(s.text, s.det_res, s.det_cfg)
    s.det_res = cleaned
    s.cleanup_dropped = dropped
    s._last_info = summary
    return StageResult(s, s._last_info)

def st_anchored(s: FlowState, *, window_left: int, window_right: int) -> StageResult[FlowState]:
    assert s.det_res is not None
    s.picks = extract_near_firsts(
        s.text, firsts=s.det_res.unique_acronyms, cfg=s.ext_cfg,
        window_left=window_left, window_right=window_right,
    )
    got = sum(1 for v in s.picks.values() if v)
    s._last_info = f"anchored picks {got}/{len(s.picks)}"
    return StageResult(s, s._last_info)

def st_defs_from_picks(s: FlowState) -> StageResult[FlowState]:
    s.anchored_defs = defs_from_picks(s.text, s.picks)
    s._last_info = f"anchored defs={len(s.anchored_defs)}"
    return StageResult(s, s._last_info)

def st_harvest(s: FlowState) -> StageResult[FlowState]:
    assert s.det_res is not None
    s.harvested_defs = harvest_defs_all(s.text, s.det_res.occurrences, s.ext_cfg)
    s._last_info = f"harvested={len(s.harvested_defs)}"
    return StageResult(s, s._last_info)

def st_sentence_backref(s: FlowState) -> StageResult[FlowState]:
    assert s.det_res is not None
    s.backref_defs = extract_sentence_backrefs(text=s.text, firsts=s.det_res.unique_acronyms, cfg=s.ext_cfg)
    s._last_info = f"backref={len(s.backref_defs)}"
    return StageResult(s, s._last_info)

def st_merge(s: FlowState) -> StageResult[FlowState]:
    s.all_defs = dedupe_defs(s.anchored_defs + s.harvested_defs + s.global_defs + s.backref_defs)
    s._last_info = f"merged unique={len(s.all_defs)}"
    return StageResult(s, s._last_info)

def st_gapfill(s: FlowState) -> StageResult[FlowState]:
    assert s.det_res is not None
    missing = [k for k, v in s.picks.items() if v is None]
    if missing:
        fills = fill_missing_from_defs(
            s.text, firsts=s.det_res.unique_acronyms, det_cfg=s.det_cfg, defs=s.all_defs
        )
        for k in missing:
            s.picks[k] = s.picks[k] or fills.get(k)

    s.strategy = "anchored+harvest"
    s.coverage = (len(s.picks) - sum(1 for v in s.picks.values() if v is None)) / max(1, len(s.picks))
    s.missing_keys = tuple(sorted(k for k, v in s.picks.items() if v is None))
    s._last_info = f"{s.strategy} coverage={s.coverage:.2%} missing={len(s.missing_keys)}"
    return StageResult(s, s._last_info)

def st_senses_and_assemble(
    s: FlowState, *, disambig_window_chars: int, disambig_margin_threshold: float
) -> StageResult[FlowState]:
    assert s.det_res is not None

    senses_by_acr = build_senses(s.all_defs)
    sense_index = {x.sense_id: x for xs in senses_by_acr.values() for x in xs}
    occs = [OccurrenceLite(o.acronym, o.start_offset, o.end_offset) for o in s.det_res.occurrences]

    resolutions = disambiguate_occurrences(
        text=s.text, occurrences=occs, senses=senses_by_acr,
        window_chars=disambig_window_chars, margin_threshold=disambig_margin_threshold,
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
