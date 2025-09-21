import json
from dataclasses import asdict
from typing import Optional, Dict, Tuple, List

from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.detection.detector import Detector
from plainera_unacronym.nlp.common.types import (
    SCHEMA_VERSION,
    DetectorConfig, DetectorResult,
    ExtractionResult, ExtractedDefinition, InTextPick, FirstOccurrence
)
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.extract_first_occ import extract_near_firsts
from plainera_unacronym.nlp.extraction.extract import extract_iter


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
    """
    1) detect acronyms
    2) anchored extraction near FirstOccurrence for each acronym
    3) if any missing -> one global extraction to fill gaps
    4) return both the detector result and an ExtractionResult data class
    """
    det = Detector(config=det_cfg or DetectorConfig())
    det_res = det.detect(text)  # or detect_parallel per your caller

    # Step 2: anchored
    ext_cfg = ext_cfg or ExtractionConfig()
    anchored_picks = extract_near_firsts(
        text,
        firsts=det_res.unique_acronyms,
        cfg=ext_cfg,
        window_left=window_left,
        window_right=window_right,
    )

    missing = tuple(sorted(k for k, v in anchored_picks.items() if v is None))
    if not missing:
        # No need for global scan; reconstruct minimal 'definitions' from picks for completeness
        defs: List[ExtractedDefinition] = []
        for key, pick in anchored_picks.items():
            if pick is None:
                continue
            a0, a1 = pick.acr_span
            # reconstruct acronym surface from text (consistent with offsets)
            acr_surface = text[a0:a1]
            defs.append(
                ExtractedDefinition(
                    acronym=acr_surface.upper(),  # normalized acronym form for consistency
                    definition=pick.definition,
                    source="in_text",
                    confidence=pick.confidence,
                    acr_start=a0, acr_end=a1,
                    def_start=pick.def_span[0], def_end=pick.def_span[1],
                    original_definition=pick.original_definition,
                )
            )
        extr = ExtractionResult(
            picks=anchored_picks,
            definitions=defs,
            strategy="anchored",
            coverage=(len(anchored_picks) - len(missing)) / max(1, len(anchored_picks)),
            missing_keys=missing,
        )
        return det_res, extr

    # Step 3: global fallback once
    global_picks, global_defs = _nearest_from_global(text, det_res.unique_acronyms, det.cfg, ext_cfg)

    # fill anchored gaps with global picks
    merged: Dict[str, Optional[InTextPick]] = dict(anchored_picks)
    for k in missing:
        merged[k] = global_picks.get(k)

    remaining_missing = tuple(sorted(k for k, v in merged.items() if v is None))
    extr = ExtractionResult(
        picks=merged,
        definitions=global_defs,       # global provides a full set of locations
        strategy="hybrid-filled",
        coverage=(len(merged) - len(remaining_missing)) / max(1, len(merged)),
        missing_keys=remaining_missing,
    )
    return det_res, extr
