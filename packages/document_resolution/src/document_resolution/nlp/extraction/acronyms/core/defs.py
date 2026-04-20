from typing import Optional

from document_resolution.nlp.common.shared import strip_trailing_punct_str
from document_resolution.nlp.common.types import ExtractedDefinition, InTextPick
from document_resolution.nlp.extraction.acronyms.core.normalise import tighten_label
from document_resolution.nlp.extraction.acronyms.matchers.tighten import tighten_label_by_acronym

_SOURCE_PRECEDENCE: dict[str, int] = {
    "parenthetical": 5,
    "inline": 4,
    "first_occurrence_anchored": 3,
    "all_occ_scan_parenthetical": 2,
    "sentence_backref": 1,
}


def _src_rank(src: str) -> int:
    return _SOURCE_PRECEDENCE.get(src, 0)


def _wins(a: ExtractedDefinition, b: ExtractedDefinition) -> bool:
    """
    Return True if `a` should replace `b` as the winner for the same meaning key.
    """
    if a.definition_confidence != b.definition_confidence:
        return a.definition_confidence > b.definition_confidence
    if _src_rank(a.source) != _src_rank(b.source):
        return _src_rank(a.source) > _src_rank(b.source)
    # stable tie-breaker: earlier definition span wins
    if (a.def_start, a.def_end) != (b.def_start, b.def_end):
        return (a.def_start, a.def_end) < (b.def_start, b.def_end)
    # final tie-breaker: earlier acronym span
    return (a.acr_start, a.acr_end) < (b.acr_start, b.acr_end)


def _meaning_key(acr: str, label: str) -> tuple[str, str]:
    """Build a canonical (acronym, label) key.

    Args:
        acr: Acronym surface form (any case).
        label: Candidate long-form label or definition.

    Returns:
        A tuple ``(ACRONYM_UPPER, tightened_label_lower)`` suitable for use as
        a stable dictionary key or join key.

    """
    return acr.upper(), tighten_label(label).lower()


def defs_from_picks(text: str, picks: dict[str, Optional[InTextPick]]) -> list[ExtractedDefinition]:
    """Convert extracted in-text picks into `ExtractedDefinition` records.

    Args:
        text (str): Original document text that the picks' spans refer to.
        picks (dict[str, Optional[InTextPick]]): Mapping of acronym key to an
            optional in-text pick. Entries with `None` are skipped.

    Returns:
        list[ExtractedDefinition]: One `ExtractedDefinition` per non-null pick,
        with acronym normalised to an uppercase key and spans mapped directly
        from the pick.

    Notes:
        - This function does not validate span bounds; it assumes `acr_span` and
          `def_span` are valid absolute offsets into `text`.
        - Dictionary ordering is preserved, so output ordering follows `picks.items()`
          in the current runtime.

    """
    out: list[ExtractedDefinition] = []
    for _, pick in picks.items():
        if pick is None:
            continue
        a0, a1 = pick.acr_span
        acr_surface = text[a0:a1]
        acr_key = strip_trailing_punct_str(acr_surface).upper()
        out.append(
            ExtractedDefinition(
                acronym=acr_key,
                definition=tighten_label_by_acronym(pick.definition, acr_key),
                source=pick.route,
                definition_confidence=pick.definition_confidence,
                acr_start=a0,
                acr_end=a1,
                def_start=pick.def_span[0],
                def_end=pick.def_span[1],
                original_definition=pick.original_definition,
            )
        )
    return out


def dedupe_defs(defs: list[ExtractedDefinition]) -> list[ExtractedDefinition]:
    """Deduplicate extracted definitions by stable meaning key.

    Args:
        defs (list[ExtractedDefinition]): Candidate definitions to deduplicate.

    Returns:
        list[ExtractedDefinition]: A filtered list containing only the first instance
        of each unique `(acronym, definition)` meaning key, preserving original order.

    Notes:
        - Deduplication is based on `_meaning_key/tighten_label`, not spans or confidence.
        - Output ordering follows the input ordering (stable dedupe).
    """
    best: dict[tuple[str, str], ExtractedDefinition] = {}
    first_idx: dict[tuple[str, str], int] = {}

    for idx, d in enumerate(defs):
        k = _meaning_key(d.acronym, d.definition)

        if k not in best:
            best[k] = d
            first_idx[k] = idx
            continue

        cur = best[k]
        if _wins(d, cur):
            best[k] = d

    # Emit winners in order of first appearance of each key
    ordered_keys = sorted(first_idx.items(), key=lambda kv: kv[1])
    return [best[k] for (k, _) in ordered_keys]
