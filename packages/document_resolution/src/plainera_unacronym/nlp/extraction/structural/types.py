from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from plainera_unacronym.nlp.common.types import Span
from plainera_unacronym.nlp.detection.structural.types import StructuralReferenceKind


@dataclass(frozen=True)
class StructuralReferenceEntry:
    """Occurrence-level structural reference extracted from source text.

    Represents a single detected structural-reference occurrence, preserving
    source-local details such as the original label, character offsets, and
    detector provenance. This is the canonicalized extraction-layer form used
    before in-document target linking.

    Attributes:
        kind: Structural reference kind, for example ``Section``, ``Schedule``,
            or ``Article``.
        label: Source-close detected label text, for example ``"III"`` or
            ``"4.2"``.
        canonical_label: Canonicalized label used downstream for consistent
            lookup, for example ``"3"`` when Roman numeral conversion is
            enabled.
        normalized_key: Detector-normalized structural key for the occurrence.
        canonical_key: Extraction-stage canonical lookup key used for anchor
            matching and downstream linking.
        start_offset: Inclusive start character offset of the occurrence in the
            source text.
        end_offset: Exclusive end character offset of the occurrence in the
            source text.
        provenance: Provenance tag describing how the occurrence was produced.
    """

    kind: StructuralReferenceKind
    label: str
    canonical_label: str
    normalized_key: str
    canonical_key: str
    start_offset: int
    end_offset: int
    provenance: str


type MATCH_STRATEGY = Literal["forward", "backward", "overlap", "unresolved"]


@dataclass(frozen=True)
class StructuralReferenceLink:
    """Occurrence-level link result for a structural reference.

    Represents the linking outcome for a single structural-reference
    occurrence. A link may be resolved to an in-document target span or remain
    unresolved with ``target_span=None``. This object preserves occurrence-level
    information even when multiple occurrences share the same canonical key.

    Attributes:
        kind: Structural reference kind for the linked occurrence.
        label: Source-close detected label text for the occurrence.
        canonical_label: Canonicalized label used for deterministic matching.
        normalized_key: Detector-normalized structural key for the occurrence.
        canonical_key: Canonical lookup key used to match the occurrence
            against structural anchors.
        reference_span: Source span of the structural-reference occurrence.
        target_span: Source span of the resolved target heading, or ``None`` if
            no in-document target was resolved.
        strength: Deterministic link-strength score for the selected match
        strategy. This is a heuristic ordinal signal rather than a probabilistic
        confidence estimate.
        match_strategy: Deterministic positional strategy used to resolve the
            link, for example ``"forward"``, ``"backward"``, ``"overlap"``, or
            ``"unresolved"``.
        provenance: Provenance tag describing how the occurrence was produced.
    """

    kind: StructuralReferenceKind
    label: str
    canonical_label: str
    normalized_key: str
    canonical_key: str
    reference_span: Span
    target_span: Span | None
    match_strategy: MATCH_STRATEGY
    strength: float
    provenance: str


@dataclass(frozen=True)
class StructuralAnchor:
    """Heading-like in-document target candidate for structural linking.

    Anchors are extracted from document structure, typically from heading-like
    lines such as ``Schedule A: Services Description`` or ``4.2 Termination``.
    They form the target universe used by the structural linker.

    Attributes:
        label: Source-close structural label extracted from the heading, for
            example ``"A"`` or ``"4.2"``.
        normalized_key: Deterministic lookup key used to group anchors and
            match them against structural reference canonical keys.
        start_offset: Inclusive start character offset of the anchor in the
            source text.
        end_offset: Exclusive end character offset of the anchor in the source
            text.
        ordinal: Zero-based document-order ordinal assigned to extracted
            anchors. Used for deterministic tie-breaking.
    """

    label: str
    normalized_key: str
    start_offset: int
    end_offset: int
    ordinal: int
    title: str | None


@dataclass(frozen=True)
class StructuralReferenceResolutionResult:
    """Final structural-reference resolution output for a document.

    This result preserves both occurrence-level outputs and deduplicated views:

    * ``references`` contains all canonicalized structural-reference
      occurrences in source order.
    * ``links`` contains all occurrence-level link results in source order.
    * ``unique_keys`` contains the first structural-reference entry encountered
      for each canonical key.
    * ``unique_links`` contains the representative structural link for each
      canonical key, typically preferring a resolved link when available.

    This split allows downstream consumers to choose between:
    - full occurrence-level traceability, and
    - a deduplicated semantic view of resolved structural targets.

    Attributes:
        references: All canonicalized structural-reference occurrences in
            source order.
        links: All occurrence-level structural link results in source order.
        unique_keys: Mapping from canonical key to the first corresponding
            structural-reference entry encountered in source order.
        unique_links: Mapping from canonical key to the representative
            structural link for that key in the final deduplicated view.
    """

    references: list[StructuralReferenceEntry]
    links: list[StructuralReferenceLink]
    unique_keys: dict[str, StructuralReferenceEntry]
    unique_links: dict[str, StructuralReferenceLink]
