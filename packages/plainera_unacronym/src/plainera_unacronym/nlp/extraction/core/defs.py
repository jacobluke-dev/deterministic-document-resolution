from typing import Optional

from plainera_unacronym.nlp.common.shared import tighten_label
from plainera_unacronym.nlp.common.types import ExtractedDefinition, InTextPick
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym


def defs_from_picks(text: str, picks: dict[str, Optional[InTextPick]]) -> list[ExtractedDefinition]:
    out: list[ExtractedDefinition] = []
    for _, pick in picks.items():
        if pick is None:
            continue
        a0, a1 = pick.acr_span
        acr_surface = text[a0:a1]
        out.append(
            ExtractedDefinition(
                acronym=acr_surface.upper(),
                definition=tighten_label_by_acronym(pick.definition, acr_surface.upper()),
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
