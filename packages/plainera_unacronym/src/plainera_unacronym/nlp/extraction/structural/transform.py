from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.types import StructuralReference
from plainera_unacronym.nlp.extraction.structural.common import is_strict_roman_numeral, roman_to_int
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
)


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
