from __future__ import annotations

import re

from plainera_unacronym.nlp.detection.structural.types import StructuralReference
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolution,
)


_ROMAN_STRICT_RE = re.compile(
    r"^(M{0,3})(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)


def _is_strict_roman_numeral(value: str) -> bool:
    """Return whether a value is a well-formed Roman numeral.

    Validates the input against a strict Roman numeral pattern covering the
    conventional subtractive forms up to 3999, for example ``III``, ``IV``,
    ``IX``, ``XL``, ``XC``, ``CD``, and ``CM``.

    Args:
        value: Candidate Roman numeral text to validate.

    Returns:
        ``True`` if ``value`` is a well-formed Roman numeral; otherwise
        ``False``.
    """
    if not value:
        return False
    return _ROMAN_STRICT_RE.fullmatch(value.upper()) is not None


def _roman_to_int(value: str) -> int:
    """Convert a well-formed Roman numeral string to an integer.

    The input is first validated using ``_is_strict_roman_numeral``. Conversion
    then proceeds right-to-left using standard Roman numeral subtraction rules.

    Args:
        value: Roman numeral text to convert, for example ``"III"``,
            ``"IV"``, or ``"MCMXCIV"``.

    Returns:
        Integer value of the Roman numeral.

    Raises:
        ValueError: If ``value`` is not a well-formed Roman numeral.
    """
    if not _is_strict_roman_numeral(value):
        raise ValueError(f"Invalid Roman numeral: {value}")

    roman_map = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    total = 0
    prev = 0
    for ch in reversed(value.upper()):
        curr = roman_map[ch]
        if curr < prev:
            total -= curr
        else:
            total += curr
            prev = curr
    return total


def _canonicalize_structural_reference(
    ref: StructuralReference,
    cfg: StructuralReferenceExtractionConfig,
) -> tuple[str, str]:
    """Return canonical label and canonical key for a detected structural reference.

    This extraction-stage helper owns stronger semantic canonicalisation than the
    detector. By default it preserves the detector-normalised form. When Roman
    numeral conversion is enabled, eligible structural references such as
    ``Article III`` are canonicalised to numeric form, for example
    ``("3", "article_3")``.

    Args:
        ref: Detected structural reference to canonicalise.
        cfg: Structural extraction configuration controlling canonicalisation
            behaviour.

    Returns:
        Tuple of ``(canonical_label, canonical_key)`` for downstream structural
        resolution output.

    Raises:
        ValueError: If Roman numeral conversion is enabled and the label is not a
            well-formed Roman numeral.
    """
    canonical_label = ref.label
    canonical_key = ref.normalized_key

    if cfg.convert_roman_numerals and ref.kind == "Article" and _is_strict_roman_numeral(ref.label):
        numeric = str(_roman_to_int(ref.label))
        canonical_label = numeric
        canonical_key = f"{ref.kind.lower()}_{numeric}"

    return canonical_label, canonical_key


def build_structural_reference_resolutions(
    *,
    references: list[StructuralReference],
    cfg: StructuralReferenceExtractionConfig,
) -> list[StructuralReferenceResolution]:
    """Build structural-reference resolution entries from detected references.

    Each detected structural reference is transformed into a resolution entry
    carrying both the detector-normalised key and the extraction-stage canonical
    key. This allows the extraction layer to preserve source-close data while
    also exposing stronger canonicalisation for downstream consumers.

    Args:
        references: Detected structural references to transform.
        cfg: Structural extraction configuration controlling canonicalisation
            behaviour.

    Returns:
        List of ``StructuralReferenceResolution`` objects in input order.
    """
    out: list[StructuralReferenceResolution] = []

    for ref in references:
        canonical_label, canonical_key = _canonicalize_structural_reference(ref, cfg)

        out.append(
            StructuralReferenceResolution(
                kind=ref.kind,
                label=ref.label,
                canonical_label=canonical_label,
                normalized_key=ref.normalized_key,
                canonical_key=canonical_key,
                start_offset=ref.start_offset,
                end_offset=ref.end_offset,
                provenance=ref.provenance,
            )
        )

    return out
