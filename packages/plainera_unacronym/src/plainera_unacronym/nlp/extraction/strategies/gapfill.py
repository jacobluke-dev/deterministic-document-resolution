from typing import Optional

from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import DetectorConfig, ExtractedDefinition, FirstOccurrence, InTextPick


def fill_missing_from_defs(
    _text: str,
    *,
    firsts: dict[str, FirstOccurrence],
    det_cfg: DetectorConfig,
    defs: list[ExtractedDefinition],
) -> dict[str, Optional[InTextPick]]:
    """Fill missing acronym picks from existing extracted definitions.

    For each acronym in ``firsts``, selects the best matching definition from
    ``defs`` using proximity to the first occurrence, then confidence, then
    earliest position. The ``text`` parameter is currently unused and kept for
    signature consistency and future span validation.

        Selection heuristic:
        - Prefer definitions whose acronym span is closest to the acronym's first
          occurrence (minimum absolute distance between ``acr_start`` and the
          first occurrence start offset).
        - Break ties by higher definition confidence (descending).
        - Break remaining ties by earlier acronym span (ascending ``acr_start``).

        Note:
            ``text`` is not currently used, but is threaded through to keep the
            helper signature consistent with other pipeline utilities and to enable
            future span validation (bounds checks, surface verification) without
            changing call sites.

        Args:
            _text:
                The full source text being processed. Currently unused.
            firsts:
                Mapping of normalized acronym key to its first occurrence metadata.
            det_cfg:
                Detector configuration used for normalizing acronym keys
                (e.g. allowed characters, dotted display mode).
            defs:
                Extracted definitions gathered from one or more strategies.

        Returns:
            A mapping from normalized acronym key to an ``InTextPick`` if a suitable
            definition is found, otherwise ``None``.
    """
    index: dict[str, list[ExtractedDefinition]] = {}
    for d in defs:
        k = normalize_acronym_key(d.acronym, det_cfg.allow_chars, dotted_mode=det_cfg.dotted_display)
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
            key=lambda c: (abs(c.acr_start - fo.start_offset), -c.definition_confidence, c.acr_start),
        )
        fills[key] = InTextPick(
            definition=best.definition,
            acr_span=(best.acr_start, best.acr_end),
            def_span=(best.def_start, best.def_end),
            definition_confidence=best.definition_confidence,
            original_definition=best.original_definition,
            reasons=best.reasons,
        )
    return fills
