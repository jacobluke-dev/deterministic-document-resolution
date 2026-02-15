from collections import defaultdict
from typing import Optional

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import ExtractedDefinition, InTextPick


def build_defs_index(
    all_defs: list[ExtractedDefinition],
    *,
    allow_chars: str,
    dotted_mode: str,
) -> dict[str, list[ExtractedDefinition]]:
    """
    Build an index of extracted definitions keyed by the detector's normalised acronym key.

    Args:
        all_defs: All extracted definitions collected so far (e.g., from anchored/global passes).
        allow_chars: Characters allowed internally when normalising acronym keys (passed through to
            `normalize_acronym_key`).
        dotted_mode: Dotted acronym policy used during normalisation (e.g., "strip" or "preserve").
            This should align with the detector/display configuration.

    Returns:
        Mapping of `normalised_key -> list[ExtractedDefinition]`. Keys are omitted when normalisation
        yields an empty string.

    Notes:
        - Preserves insertion order of `all_defs` within each bucket (no sorting applied here).
        - Normalisation is the sole gatekeeper: if it returns "", that definition is not indexed.
    """
    idx: dict[str, list[ExtractedDefinition]] = defaultdict(list)
    for d in all_defs:
        k = normalize_acronym_key(d.acronym, allow_chars, dotted_mode=dotted_mode)
        if k:
            idx[k].append(d)
    return idx


def backfill_missing_picks_from_defs(
    picks: dict[str, Optional[InTextPick]],
    *,
    defs_index: dict[str, list[ExtractedDefinition]],
    unique_acronyms: dict[str, FirstOccurrence],
) -> dict[str, Optional[InTextPick]]:
    """
    Backfill only missing picks (`None`) using candidate definitions from an index.

    Args:
        picks: Current pick map (`normalised_key -> InTextPick | None`). Existing non-None picks are
            left untouched.
        defs_index: Mapping of `normalised_key -> candidate ExtractedDefinition`s.
        unique_acronyms: Mapping of `normalised_key -> Occurrence` (first/unique occurrence info)
            used to choose the nearest matching definition.

    Returns:
        Updated picks dict (same object mutated, but also returned for convenience).

    Selection:
        For each missing key, choose `best = min(cands, key=(distance_to_first_occurrence,
        -definition_confidence, definition_start_offset))`.

    Provenance:
        The created `InTextPick` is populated from the winning `ExtractedDefinition`:
        - kind <- best.kind (fallback "unknown")
        - route <- best.source (fallback "unknown")
        - reasons <- best.reasons
    """
    missing = [k for k, v in picks.items() if v is None]
    for key in missing:
        fo = unique_acronyms.get(key)
        if fo is None:
            continue

        cands = defs_index.get(key, [])
        if not cands:
            continue

        best = min(
            cands,
            key=lambda c: (abs(c.acr_start - fo.start_offset), -c.definition_confidence, c.acr_start),
        )

        picks[key] = InTextPick(
            definition=best.definition,
            acr_span=(best.acr_start, best.acr_end),
            def_span=(best.def_start, best.def_end),
            definition_confidence=best.definition_confidence,
            original_definition=best.original_definition,
            kind=best.kind or "unknown",
            route=best.source or "unknown",
            reasons=best.reasons,
        )

    return picks


def patch_pick_provenance(
    picks: dict[str, Optional[InTextPick]],
    *,
    default_route: str = "first_occurrence_anchored",
) -> dict[str, Optional[InTextPick]]:
    """
    Light patch-up for pick provenance to avoid leaving route as "unknown".

    Args:
        picks: Pick map (`normalised_key -> InTextPick | None`).
        default_route: Route value to apply when an existing pick has route == "unknown".

    Returns:
        Updated picks dict (same object mutated, but also returned for convenience).

    Notes:
        This is intentionally low-noise: it only changes `route` when it is literally "unknown".
        It does not attempt to infer a more specific route.
    """
    for k, p in list(picks.items()):
        if p is None:
            continue
        if getattr(p, "route", "unknown") == "unknown":
            picks[k] = InTextPick(
                definition=p.definition,
                acr_span=p.acr_span,
                def_span=p.def_span,
                definition_confidence=p.definition_confidence,
                original_definition=p.original_definition,
                kind=p.kind,
                route=default_route,
                reasons=p.reasons,
            )
    return picks
