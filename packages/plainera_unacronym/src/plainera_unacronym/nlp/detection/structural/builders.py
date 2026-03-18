from __future__ import annotations

from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str

from .normalise import normalize_structural_reference_key
from .types import StructuralReference


def build_structural_reference(
    *,
    kind: str,
    label: str,
    start_offset: int,
    end_offset: int,
    provenance: str,
) -> StructuralReference:
    cleaned_kind = strip_trailing_punct_str(kind.strip())
    cleaned_label = strip_trailing_punct_str(label.strip())

    return StructuralReference(
        kind=cleaned_kind,
        label=cleaned_label,
        start_offset=start_offset,
        end_offset=end_offset,
        normalized_key=normalize_structural_reference_key(cleaned_kind, cleaned_label),
        provenance=provenance,
    )
