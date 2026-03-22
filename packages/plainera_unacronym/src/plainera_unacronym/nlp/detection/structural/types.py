from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type StructuralReferenceKind = Literal[
    "Schedule",
    "Exhibit",
    "Annex",
    "Appendix",
    "Section",
    "Clause",
    "Article",
]


@dataclass(frozen=True)
class StructuralReference:
    """Canonical structural reference detected in document text.

    Structural references represent navigational elements in a document rather
    than semantic entities. Examples include ``Section 4.2``, ``Clause 7``,
    ``Schedule A``, and ``Article III``.

    Attributes:
        kind: Structural reference kind constrained to
            ``StructuralReferenceKind``, one of ``"Schedule"``,
            ``"Exhibit"``, ``"Annex"``, ``"Appendix"``, ``"Section"``,
            ``"Clause"``, or ``"Article"``.
        label: Structural reference label associated with the kind, for example
            ``"4.2"``, ``"A"``, or ``"III"``.
        start_offset: Inclusive start offset of the detected reference in the
            source text.
        end_offset: Exclusive end offset of the detected reference in the source
            text.
        normalized_key: Deterministic canonical key derived from the structural
            kind and label, for example ``"section_4_2"`` or ``"schedule_a"``.
        provenance: Source label describing how the structural reference was
            produced.
    """

    kind: StructuralReferenceKind
    label: str
    start_offset: int
    end_offset: int
    normalized_key: str
    provenance: str


@dataclass(frozen=True)
class StructuralReferenceDetectorResult:
    """Container for structural references detected in a text run.

    Attributes:
        references: Structural references detected in source order.
    """

    references: list[StructuralReference]
