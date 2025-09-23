from typing import Mapping, Optional

from ..common.shared import normalize_acronym_key
from ..common.types import FirstOccurrence, DetectorConfig, InTextPick
from .config import ExtractionConfig
from .extract import extract_iter, ExtractedDefinition

def picks_from_global(
    text: str,
    firsts: Mapping[str, FirstOccurrence],              # key = normalized_key from detector
    det_cfg: DetectorConfig,                            # for normalize_key (allow_chars, dotted policy)
    ext_cfg: ExtractionConfig = ExtractionConfig(),
) -> dict[str, Optional[InTextPick]]:
    """
    Single global pass over `text`, then for each FirstOccurrence key choose the
    nearest matching in-text definition (parenthetical > inline via confidence).

    Returns: { normalized_key: InTextPick | None }
    """
    # 1) Run global extraction once
    defs = list(extract_iter(text, ext_cfg))

    # 2) Index definitions by detector's normalized key
    dotted_mode = getattr(det_cfg, "dotted_display", "strip")
    index: dict[str, list[ExtractedDefinition]] = {}
    for d in defs:
        k = normalize_acronym_key(d.acronym, det_cfg.allow_chars, dotted_mode=dotted_mode)
        if not k:
            continue
        index.setdefault(k, []).append(d)

    # 3) For each FO, pick nearest candidate (tie-break by higher confidence)
    picks: dict[str, Optional[InTextPick]] = {}
    for key, fo in firsts.items():
        cands = index.get(key, [])
        if not cands:
            picks[key] = None
            continue

        # nearest by acronym start to FO.start_offset; prefer higher confidence on ties
        best = min(
            cands,
            key=lambda c: (abs(c.acr_start - fo.start_offset), -c.confidence, c.acr_start),
        )
        picks[key] = InTextPick(
            definition=best.definition,
            acr_span=(best.acr_start, best.acr_end),
            def_span=(best.def_start, best.def_end),
            confidence=best.confidence,
            original_definition=best.original_definition,
        )

    return picks
