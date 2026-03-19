from __future__ import annotations

from typing import Final

from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str

from .normalise import normalize_structural_reference_key
from .types import StructuralReference, StructuralReferenceKind

_KIND_MAP: Final[dict[str, StructuralReferenceKind]] = {
    "schedule": "Schedule",
    "exhibit": "Exhibit",
    "annex": "Annex",
    "appendix": "Appendix",
    "section": "Section",
    "clause": "Clause",
    "article": "Article",
}


def canonicalize_structural_kind(kind: str) -> StructuralReferenceKind:
    """Canonicalise a structural reference kind to its typed enum-like literal.

    Normalises the raw detected structural kind by trimming surrounding
    whitespace, lowercasing it, and resolving it through ``_KIND_MAP`` to one of
    the canonical ``StructuralReferenceKind`` literal values.

    Args:
        kind: Raw structural kind text to canonicalise, for example
            ``"section"``, ``" Section "``, or ``"ARTICLE"``.

    Returns:
        The canonical ``StructuralReferenceKind`` value, for example
        ``"Section"`` or ``"Article"``.

    Raises:
        KeyError: If ``kind`` does not map to a supported structural reference
            kind.
    """
    value = kind.strip().lower()
    return _KIND_MAP[value]


def build_structural_reference(
    *,
    kind: str,
    label: str,
    start_offset: int,
    end_offset: int,
    provenance: str,
) -> StructuralReference:
    """Build a canonical structural reference from a detected structural span.

    The raw structural kind and label are trimmed and cleaned of trailing
    punctuation noise before being normalised into a stable lookup key. The
    resulting object preserves the original source offsets and provenance for
    downstream extraction and traceability.

    Args:
        kind: Raw detected structural keyword, for example ``"Section"`` or
            ``"Schedule"``.
        label: Raw detected structural label, for example ``"4.2"``, ``"A"``,
            or ``"III"``.
        start_offset: Inclusive start offset of the detected structural reference
            in the source text.
        end_offset: Exclusive end offset of the detected structural reference in
            the source text.
        provenance: Source label describing how the structural reference was
            produced.

    Returns:
        A ``StructuralReference`` containing the cleaned structural kind and
        label, preserved source offsets, canonical normalised lookup key, and
        provenance.
    """
    cleaned_kind_raw = strip_trailing_punct_str(kind.strip())
    cleaned_label = strip_trailing_punct_str(label.strip())
    cleaned_kind = canonicalize_structural_kind(cleaned_kind_raw)

    return StructuralReference(
        kind=cleaned_kind,
        label=cleaned_label,
        start_offset=start_offset,
        end_offset=end_offset,
        normalized_key=normalize_structural_reference_key(cleaned_kind, cleaned_label),
        provenance=provenance,
    )
