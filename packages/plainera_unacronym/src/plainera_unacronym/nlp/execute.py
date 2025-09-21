import json
from dataclasses import asdict
from typing import Optional, Dict, Tuple, List

from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.detection.detector import Detector
from plainera_unacronym.nlp.common.types import (
    SCHEMA_VERSION,
    DetectorConfig, DetectorResult,
    ExtractionResult, ExtractedDefinition, InTextPick, FirstOccurrence, OccurrenceLite
)
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.defs_utils import defs_from_picks
from plainera_unacronym.nlp.extraction.extract_first_occ import extract_near_firsts
from plainera_unacronym.nlp.extraction.extract import extract_iter
from plainera_unacronym.nlp.extraction.harvest import harvest_defs_all
from plainera_unacronym.nlp.extraction.helper_patterns import dedupe_defs
from plainera_unacronym.nlp.senses.disambiguate import disambiguate_occurrences
from plainera_unacronym.nlp.senses.sense_build import build_senses


def serialize_detection_and_extraction(det: DetectorResult, extr: ExtractionResult, *, pretty: bool = False) -> str:
    """Serialize detection and in-text extraction results to a JSON string.

        Produces a Unicode JSON payload (``ensure_ascii=False``) that combines the
        detector output and the extraction summary under a stable schema version.

        Payload structure:
          - ``schema_version`` (str): Version tag (from ``SCHEMA_VERSION``).
          - ``detection`` (object):
              - ``unique_acronyms`` (dict[str, FirstOccurrence]): Map of normalized
                acronym key → first occurrence fields (``acronym``, offsets,
                ``confidence``, ``normalized_key``).
              - ``occurrences`` (list[Occurrence]): All accepted acronym occurrences,
                including context windows and optional ``reasons`` when enabled.
          - ``extraction`` (object):
              - ``strategy`` (str): One of ``"anchored"``, ``"hybrid-filled"``, or ``"global"``.
              - ``coverage`` (float): Fraction of acronyms that received an in-text definition.
              - ``missing_keys`` (list[str]): Normalized keys with no in-text definition.
              - ``picks`` (dict[str, InTextPick | null]): Best in-text definition per key
                (or ``null`` if none). ``InTextPick`` includes ``definition``,
                ``acr_span``, ``def_span``, ``confidence``, and ``original_definition``.
              - ``definitions`` (list[ExtractedDefinition]): All definition locations considered/returned.
                ``ExtractedDefinition`` includes ``acronym``, normalized ``definition``,
                ``source`` (always ``"in_text"``), ``confidence``, acronym/definition spans,
                and ``original_definition``.

        Args:
            det (DetectorResult): The detector output containing first occurrences and all occurrences.
            extr (ExtractionResult): The extraction output with per-key picks and definition locations.
            pretty (bool, optional): If ``True``, pretty-prints JSON with indentation.
                Defaults to ``False``.

        Returns:
            str: A JSON string with the fields described above. Unicode characters are preserved
            (``ensure_ascii=False``).

        Example:
            >>> json_str = serialize_detection_and_extraction(det_res, extr, pretty=True)
            >>> data = json.loads(json_str)
            >>> data["extraction"]["strategy"]
            'anchored'
        """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "detection": {
            "unique_acronyms": {k: asdict(v) for k, v in det.unique_acronyms.items()},
            "occurrences": [asdict(o) for o in det.occurrences],
        },
        "extraction": {
            "strategy": extr.strategy,
            "coverage": extr.coverage,
            "missing_keys": list(extr.missing_keys),
            "picks": {k: (asdict(v) if v else None) for k, v in extr.picks.items()},
            "definitions": [asdict(d) for d in extr.definitions],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _nearest_from_global(
    text: str,
    firsts: Dict[str, FirstOccurrence],
    det_cfg: DetectorConfig,
    ext_cfg: ExtractionConfig,
) -> Tuple[Dict[str, Optional[InTextPick]], List[ExtractedDefinition]]:
    """
    Single global scan; index by detector's key and pick nearest per FO.
    Returns (picks_by_key, all_defs).
    """
    defs = list(extract_iter(text, ext_cfg))
    # index by normalized key
    dotted_mode = getattr(det_cfg, "dotted_display", "strip")
    index: Dict[str, List[ExtractedDefinition]] = {}
    for d in defs:
        k = normalize_acronym_key(d.acronym, det_cfg.allow_chars, dotted_mode)
        if not k:
            continue
        index.setdefault(k, []).append(d)

    picks: Dict[str, Optional[InTextPick]] = {}
    for key, fo in firsts.items():
        cands = index.get(key, [])
        if not cands:
            picks[key] = None
            continue
        # nearest by acronym start; prefer higher confidence on ties
        best = min(cands, key=lambda c: (abs(c.acr_start - fo.start_offset), -c.confidence, c.acr_start))
        picks[key] = InTextPick(
            definition=best.definition,
            acr_span=(best.acr_start, best.acr_end),
            def_span=(best.def_start, best.def_end),
            confidence=best.confidence,
            original_definition=best.original_definition,
        )
    return picks, defs


def detect_and_extract(
    text: str,
    *,
    det_cfg: Optional[DetectorConfig] = None,
    ext_cfg: Optional[ExtractionConfig] = None,
    window_left: int = 320,
    window_right: int = 280,
) -> Tuple[DetectorResult, ExtractionResult]:
    det = Detector(config=det_cfg or DetectorConfig())
    det_res = det.detect(text)

    ext_cfg = ext_cfg or ExtractionConfig()

    anchored_picks = extract_near_firsts(
        text, firsts=det_res.unique_acronyms, cfg=ext_cfg,
        window_left=window_left, window_right=window_right,
    )

    # 1) defs from anchored + 2) harvest extra
    anchored_defs = defs_from_picks(text, anchored_picks)
    extra_defs = harvest_defs_all(text, det_res.occurrences, ext_cfg)

    # 3) dedupe + choose strategy fields
    all_defs = dedupe_defs(anchored_defs + extra_defs)

    # Optionally still run your global gap-fill if you want picks for missing keys
    missing = tuple(sorted(k for k, v in anchored_picks.items() if v is None))
    if missing:
        global_picks, global_defs = _nearest_from_global(text, det_res.unique_acronyms, det.cfg, ext_cfg)
        merged = dict(anchored_picks)
        for k in missing:
            merged[k] = global_picks.get(k)
        picks = merged
        strategy = "anchored+harvest+global"
        coverage = (len(merged) - sum(1 for v in merged.values() if v is None)) / max(1, len(merged))
        missing_keys = tuple(sorted(k for k, v in merged.items() if v is None))
        # Optionally merge global_defs into the pool as well:
        all_defs = dedupe_defs(all_defs + list(global_defs))
    else:
        picks = anchored_picks
        strategy = "anchored+harvest"
        coverage = (len(picks) - 0) / max(1, len(picks))
        missing_keys = ()

    # 4) senses + disambiguation
    senses_by_acr = build_senses(all_defs)
    sense_index = {s.sense_id: s for senses in senses_by_acr.values() for s in senses}
    ambiguous = tuple(sorted(k for k, v in senses_by_acr.items() if len(v) > 1))

    occs = [OccurrenceLite(o.acronym, o.start_offset, o.end_offset) for o in det_res.occurrences]
    resolutions = disambiguate_occurrences(
        text=text,
        occurrences=occs,
        senses=senses_by_acr,
        window_chars=getattr(det.cfg, "window_chars", 320),
        margin_threshold=0.20,
    )
    undecided = [r for r in resolutions if r.chosen_sense_id is None]

    extr = ExtractionResult(
        picks=picks,
        definitions=all_defs,  # <-- now includes both EMA senses, etc.
        strategy=strategy,
        coverage=coverage,
        missing_keys=missing_keys,
        senses_by_acronym=senses_by_acr,
        sense_index=sense_index,
        resolutions=resolutions,
        ambiguous_keys=ambiguous,
        undecided=undecided,
    )
    return det_res, extr
