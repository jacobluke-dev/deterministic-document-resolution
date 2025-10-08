from typing import Optional

from .stages import Stage, StageResult, Chain
from .. import ExtractionConfig
from ..defs_utils import defs_from_picks, dedupe_defs
from ..extract_first_occ import extract_near_firsts
from ..harvest import harvest_defs_all
from ..util import picks_from_global
from ... import DetectorConfig, Detector
from ...common.types import OccurrenceLite, ExtractionResult
from ...senses.disambiguate import disambiguate_occurrences
from ...senses.sense_build import build_senses
from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import InTextPick, ExtractedDefinition, FirstOccurrence


# def build_extraction_pipeline(ext_cfg: ExtractionConfig) -> Pipeline:
#
#     parenth_pat = compile_parenthetical_patterns(ext_cfg)
#     inline_pats = _compile_inline(ext_cfg)
#
#     return Pipeline([
#         # choose one capture at a time; you can run twice and concat
#         CaptureParenthetical(parenth_pat),
#         Guards(),
#         TightenTail(),
#         NormalizeDisplay(),
#         PolicyGates(),
#         TightenByAcronym(),
#         ParentheticalNumericReattach(),
#         ScoreAndPack(),
#         Dedupe(),
#     ]), Pipeline([
#         CaptureInline(inline_pats),
#         Guards(),
#         TightenTail(),
#         NormalizeDisplay(),
#         PolicyGates(),
#         TightenByAcronym(),
#         ScoreAndPack(),
#         Dedupe(),
#     ])
#
# def extract_pipeline_iter(text: str, det_cfg: DetectorConfig, ext_cfg: ExtractionConfig, plan) -> list[FinalPick]:
#     ctx = Ctx(text=text, det_cfg=det_cfg, ext_cfg=ext_cfg, plan=plan, trace=False)
#     parenth_pl, inline_pl = build_extraction_pipeline(ext_cfg)
#     out = parenth_pl.run([None], ctx) + inline_pl.run([None], ctx)
#     # If you prefer to keep your original return type:
#     from plainera_unacronym.nlp.common.types import ExtractedDefinition
#     return [
#         ExtractedDefinition(
#             acronym=f.acronym, definition=f.definition, original_definition=f.original_definition,
#             acr_start=f.acr_span[0], acr_end=f.acr_span[1], def_start=f.def_span[0], def_end=f.def_span[1],
#             confidence=f.confidence, source="in_text"
#         )
#         for f in out
#     ]





def detect_and_extract(text: str, *, det_cfg=None, ext_cfg=None, window_left=320, window_right=280):
    det_cfg = det_cfg or DetectorConfig()
    ext_cfg = ext_cfg or ExtractionConfig()

    # Stage A: detection
    def st_detect(_):
        det = Detector(config=det_cfg).detect(text)
        return StageResult(det, f"firsts={len(det.unique_acronyms)} occs={len(det.occurrences)}")

    # Stage B: anchored picks
    def st_anchored(det_res):
        picks = extract_near_firsts(text,
                                    firsts=det_res.unique_acronyms,
                                    cfg=ext_cfg,
                                    window_left=window_left,
                                    window_right=window_right)
        got = sum(1 for v in picks.values() if v)
        return StageResult((det_res, picks), f"anchored picks {got}/{len(picks)}")

    # Stage C: defs_from_picks
    def st_defs(inp):
        det_res, picks = inp
        defs = defs_from_picks(text, picks)
        return StageResult((det_res, picks, defs), f"anchored defs={len(defs)}")

    # Stage D: harvest
    def st_harvest(inp):
        det_res, picks, anch_defs = inp
        extra = harvest_defs_all(text, det_res.occurrences, ext_cfg)
        return StageResult((det_res, picks, anch_defs, extra), f"harvested={len(extra)}")

    # # Stage E: global (optional – or call your _nearest_from_global)
    # def st_global(inp):
    #     det_res, picks, anch_defs, extra = inp
    #     global_defs = extract_pipeline_iter(text, det_res.cfg, ext_cfg, plan=None)
    #     return StageResult((det_res, picks, anch_defs, extra, global_defs), f"global={len(global_defs)}")

    def st_merge(inp):
        # Accept either (det_res, picks, anch_defs, extra) or (+ global_defs)
        if len(inp) == 4:
            det_res, picks, anch_defs, extra = inp
            global_defs = []
        elif len(inp) == 5:
            det_res, picks, anch_defs, extra, global_defs = inp
        else:
            raise ValueError(f"merge expects 4 or 5 items, got {len(inp)}: {inp}")

        all_defs = dedupe_defs(anch_defs + extra + global_defs)
        return StageResult((det_res, picks, all_defs), f"merged unique={len(all_defs)}")

    # Stage G: gap-fill picks (nearest-to-FO using defs)
    def st_gap(inp):
        det_res, picks, all_defs = inp
        missing = [k for k, v in picks.items() if v is None]
        if missing:
            # det_cfg is available in the outer scope of detect_and_extract
            fills = picks_from_global(
                text,
                firsts=det_res.unique_acronyms,
                det_cfg=det_cfg,  # <- use the closure var, not det_res.cfg
                ext_cfg=all_defs  # <- pass the definitions, not ext_cfg
            )
            # only fill missing keys; keep existing picks as-is
            picks = {**picks, **{k: picks[k] or fills.get(k) for k in missing}}

        strategy = "anchored+harvest" if not missing else "anchored+harvest+global-pipeline"
        cov = (len(picks) - sum(1 for v in picks.values() if v is None)) / max(1, len(picks))
        miss = tuple(sorted(k for k, v in picks.items() if v is None))
        return StageResult(
            (det_res, picks, all_defs, strategy, cov, miss),
            f"{strategy} coverage={cov:.2%} missing={len(miss)}"
        )

    # Stage H: senses + disambiguation
    def st_senses(inp):
        det_res, picks, all_defs, strategy, cov, miss = inp
        senses_by_acr = build_senses(all_defs)
        sense_index = {s.sense_id: s for senses in senses_by_acr.values() for s in senses}

        occs = [OccurrenceLite(o.acronym, o.start_offset, o.end_offset) for o in det_res.occurrences]
        window_chars = getattr(det_cfg, "window_chars", 320)  # <-- use outer det_cfg
        resolutions = disambiguate_occurrences(
            text=text,
            occurrences=occs,
            senses=senses_by_acr,
            window_chars=window_chars,
            margin_threshold=0.20,
        )

        undecided = [r for r in resolutions if r.chosen_sense_id is None]
        ambiguous = tuple(sorted(k for k, v in senses_by_acr.items() if len(v) > 1))
        extr = ExtractionResult(
            picks=picks,
            definitions=all_defs,
            strategy=strategy,
            coverage=cov,
            missing_keys=miss,
            senses_by_acronym=senses_by_acr,
            sense_index=sense_index,
            resolutions=resolutions,
            ambiguous_keys=ambiguous,
            undecided=undecided,
        )
        return StageResult((det_res, extr),
                           f"senses={sum(len(v) for v in senses_by_acr.values())}, undecided={len(undecided)}")

    chain = Chain([
        Stage("detect",            st_detect, lambda d: f"firsts={len(d.unique_acronyms)}"),
        Stage("anchored_picks",    st_anchored, lambda t: "…"),
        Stage("defs_from_picks",   st_defs,     lambda t: "…"),
        Stage("harvest",           st_harvest,  lambda t: "…"),
        # Stage("global_pipeline",   st_global,   lambda t: "…"),
        Stage("merge_dedupe",      st_merge,    lambda t: "…"),
        Stage("gap_fill_picks",    st_gap,      lambda t: "…"),
        Stage("senses_disambiguate", st_senses, lambda t: "ready"),
    ])
    (det_res, extr), reports = chain.run(object())
    return det_res, extr, reports
