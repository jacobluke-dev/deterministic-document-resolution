from __future__ import annotations

from document_resolution.nlp.detection.structural.types import StructuralReference
from document_resolution.nlp.extraction.structural.common import is_strict_roman_numeral, roman_to_int
from document_resolution.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from document_resolution.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
)


def _canonicalize_structural_reference(
    ref: StructuralReference,
    cfg: StructuralReferenceExtractionConfig,
) -> tuple[str, str]:
    """Return canonical label and canonical key for a detected structural reference.

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

    if cfg.convert_roman_numerals and ref.kind == "Article" and is_strict_roman_numeral(ref.label):
        numeric = str(roman_to_int(ref.label))
        canonical_label = numeric
        canonical_key = f"{ref.kind.lower()}_{numeric}"

    return canonical_label, canonical_key


def build_structural_reference_resolutions(
    *,
    references: list[StructuralReference],
    cfg: StructuralReferenceExtractionConfig,
) -> list[StructuralReferenceEntry]:
    """Build structural-reference resolution entries from detected references.

    Args:
        references: Detected structural references to transform.
        cfg: Structural extraction configuration controlling canonicalisation
            behaviour.

    Returns:
        List of ``StructuralReferenceResolution`` objects in input order.
    """
    out: list[StructuralReferenceEntry] = []

    for ref in references:
        canonical_label, canonical_key = _canonicalize_structural_reference(ref, cfg)

        out.append(
            StructuralReferenceEntry(
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
