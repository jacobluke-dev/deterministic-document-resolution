from typing import Optional
from plainera_unacronym.nlp.extraction.core.normalise import tighten_label
from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str
from plainera_unacronym.nlp.common.types import ExtractedDefinition, InTextPick
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym


def defs_from_picks(text: str, picks: dict[str, Optional[InTextPick]]) -> list[ExtractedDefinition]:
    """Convert extracted in-text picks into `ExtractedDefinition` records.

    For each non-null `InTextPick`, this builds an `ExtractedDefinition` using:
    - the pick's acronym and definition spans (absolute offsets into `text`)
    - the pick's confidence and original definition text
    - a normalised acronym key derived from the surface acronym in `text`

    The acronym is normalised by:
    1) slicing the surface form from `text` using `pick.acr_span`
    2) stripping trailing punctuation from that surface form
    3) uppercasing the result

    The definition label is then normalised via `tighten_label_by_acronym()` using
    the computed acronym key.

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
                source="in_text",
                confidence=pick.confidence,
                acr_start=a0,
                acr_end=a1,
                def_start=pick.def_span[0],
                def_end=pick.def_span[1],
                original_definition=pick.original_definition,
            )
        )
    return out


def _sense_key(acr: str, label: str) -> tuple[str, str]:
    """Build a canonical (acronym, label) key.

    Uppercases the acronym and returns it alongside a normalized, lowercase
    label produced by :func:`tighten_label`. This function does not validate
    that the acronym and label correspond; they are treated independently.

    Args:
        acr: Acronym surface form (any case).
        label: Candidate long-form label or definition.

    Returns:
        A tuple ``(ACRONYM_UPPER, tightened_label_lower)`` suitable for use as
        a stable dictionary key or join key.

    Examples:
        >>> _sense_key("Gpu", "Graphics Processing Unit")
        ('GPU', 'graphics processing unit')
        >>> _sense_key("PDF", "And, which the Portable Document Format")
        ('PDF', 'portable document format')
        # Acronym and label do not need to match:
        >>> _sense_key("GPU", "Portable Document Format")
        ('GPU', 'portable document format')

    Notes:
        - ``tighten_label`` removes leading connectors/articles and keeps
          meaningful RHS for patterns like ``"X stands for Y"`` before lowering.
        - No punctuation/whitespace trimming is applied to ``acr`` beyond
          uppercasing; callers should pre-clean if needed.
    """
    return acr.upper(), tighten_label(label).lower()


def dedupe_defs(defs: list[ExtractedDefinition]) -> list[ExtractedDefinition]:
    """Deduplicate extracted definitions by stable sense key.

        Definitions are considered duplicates when they resolve to the same sense key,
        computed via `_sense_key(d.acronym, d.definition)`. The first occurrence is
        kept and subsequent duplicates are dropped. The `definition` field is preserved
        exactly as provided (it is assumed to have been tightened/normalised upstream).

        Args:
            defs (list[ExtractedDefinition]): Candidate definitions to deduplicate.

        Returns:
            list[ExtractedDefinition]: A filtered list containing only the first instance
            of each unique `(acronym, definition)` sense key, preserving original order.

        Notes:
            - Deduplication is based on `_sense_key/tighten_label`, not spans or confidence.
            - Output ordering follows the input ordering (stable dedupe).
    """
    seen: set[tuple[str, str]] = set()
    out: list[ExtractedDefinition] = []
    for d in defs:
        k = _sense_key(d.acronym, d.definition)
        if k in seen:
            continue
        seen.add(k)
        # keep d.definition as-is (already tightened upstream where needed)
        out.append(d)
    return out
