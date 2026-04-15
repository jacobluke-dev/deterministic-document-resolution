from __future__ import annotations

import re

from document_resolution.nlp.extraction.structural.common import is_strict_roman_numeral, roman_to_int
from document_resolution.nlp.extraction.structural.config import StructuralReferenceExtractionConfig
from document_resolution.nlp.extraction.structural.types import StructuralAnchor

_LINE_RE = re.compile(r"(?m)^[^\n]*\S[^\n]*$")

_NAMED_HEADING_RE = re.compile(
    r"""
    ^
    (?P<kind>Schedule|Section|Clause|Article|Appendix|Annex|Exhibit)
    \s+
    (?P<label>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)
    \b
    (?:\s*[:.\-–]\s*(?P<title>\S.*))?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NUMBERED_SECTION_HEADING_RE = re.compile(
    r"""
    ^
    (?P<label>\d+(?:\.\d+)+|\d+)
    \s+
    (?P<title>\S.*)
    $
    """,
    re.VERBOSE,
)


def _slug(value: str) -> str:
    """Normalize free-form text into a lowercase underscore slug.

    Non-alphanumeric runs are collapsed to underscores, repeated underscores
    are squashed, and leading or trailing underscores are removed.

    Args:
        value: Raw text to normalize.

    Returns:
        Normalized slug value suitable for structural lookup keys.
    """
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _build_anchor_key(
    *,
    kind: str,
    label: str,
    cfg: StructuralReferenceExtractionConfig,
) -> str:
    """Build the deterministic lookup key for a structural anchor.

    The key format mirrors structural-reference canonical keys so that
    downstream linking can match references to anchors directly. When Roman
    numeral conversion is enabled, article labels such as ``III`` are
    converted to numeric form, for example ``article_3``.

    Args:
        kind: Structural heading kind, for example ``"Schedule"`` or
            ``"Article"``.
        label: Heading label extracted from the source line.
        cfg: Extraction configuration controlling Roman numeral handling.

    Returns:
        Canonicalized lookup key for the anchor.
    """
    normalized_kind = kind.lower()
    normalized_label = label.strip()

    if cfg.convert_roman_numerals and normalized_kind == "article" and is_strict_roman_numeral(normalized_label):
        normalized_label = str(roman_to_int(normalized_label))

    return f"{normalized_kind}_{_slug(normalized_label)}"


def _build_anchor(
    *,
    label: str,
    title: str | None,
    kind: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor:
    """Build a structural anchor from parsed heading components.

    Normalizes optional heading title text and derives the deterministic
    structural lookup key from heading kind + label only. Title text is
    preserved for display and traceability, but does not affect anchor
    matching.

    Args:
        label: Structural label extracted from the heading, for example
            ``"4.2"`` or ``"A"``.
        title: Optional descriptive heading suffix, for example
            ``"Termination"`` or ``"Services Description"``.
        kind: Structural heading kind used for lookup-key construction, for
            example ``"Section"`` or ``"Schedule"``.
        start_offset: Inclusive start character offset of the heading.
        end_offset: Exclusive end character offset of the heading.
        ordinal: Zero-based document-order ordinal assigned to the anchor.
        cfg: Extraction configuration controlling key canonicalization.

    Returns:
        A ``StructuralAnchor`` with normalized title text and deterministic
        lookup key.
    """
    normalized_title = title.strip() if title is not None else None
    if normalized_title == "":
        normalized_title = None

    return StructuralAnchor(
        label=label,
        title=normalized_title,
        normalized_key=_build_anchor_key(kind=kind, label=label, cfg=cfg),
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
    )


def _named_anchor_from_line(
    *,
    line: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor | None:
    """Build an anchor from a named structural heading line.

    Supports headings such as ``Schedule A``, ``Section 4.2``,
    ``Article III``, and variants with trailing descriptive titles.

    Args:
        line: Candidate heading line with surrounding whitespace removed.
        start_offset: Start character offset of the stripped line in the
            original text.
        end_offset: End character offset of the stripped line in the
            original text.
        ordinal: Zero-based ordinal of the anchor within document order.
        cfg: Extraction configuration controlling key canonicalization.

    Returns:
        A ``StructuralAnchor`` if the line matches a named heading pattern;
        otherwise ``None``.
    """
    match = _NAMED_HEADING_RE.match(line)
    if match is None:
        return None

    return _build_anchor(
        kind=match.group("kind"),
        label=match.group("label"),
        title=match.group("title"),
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
        cfg=cfg,
    )


def _numbered_section_anchor_from_line(
    *,
    line: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor | None:
    """Build an anchor from a bare numbered section heading line.

    Supports headings such as ``4.2 Termination`` by applying the current
    deterministic default rule that bare numbered headings are interpreted as
    ``Section`` anchors. This default preserves stable structural lookup
    behaviour, but does not imply that explicit cross-kind linking between
    ``Clause`` and ``Section`` references is allowed.

    Args:
        line: Candidate heading line with surrounding whitespace removed.
        start_offset: Start character offset of the stripped line in the
            original text.
        end_offset: End character offset of the stripped line in the
            original text.
        ordinal: Zero-based ordinal of the anchor within document order.
        cfg: Extraction configuration controlling key canonicalization.

    Returns:
        A ``StructuralAnchor`` if the line matches a numbered heading pattern;
        otherwise ``None``.
    """
    match = _NUMBERED_SECTION_HEADING_RE.match(line)
    if match is None:
        return None

    return _build_anchor(
        kind="Section",
        label=match.group("label"),
        title=match.group("title"),
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
        cfg=cfg,
    )


def _anchor_from_line(
    *,
    line: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor | None:
    """Build a structural anchor from a candidate line.

    Named structural headings are attempted first. If no named heading is
    found, the line is tested as a bare numbered section heading.

    Args:
        line: Candidate heading line with surrounding whitespace removed.
        start_offset: Start character offset of the stripped line in the
            original text.
        end_offset: End character offset of the stripped line in the
            original text.
        ordinal: Zero-based ordinal of the anchor within document order.
        cfg: Extraction configuration controlling key canonicalization.

    Returns:
        A ``StructuralAnchor`` when the line represents a supported structural
        heading; otherwise ``None``.
    """
    anchor = _named_anchor_from_line(
        line=line,
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
        cfg=cfg,
    )
    if anchor is not None:
        return anchor

    return _numbered_section_anchor_from_line(
        line=line,
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
        cfg=cfg,
    )


def extract_structural_anchors(
    *,
    text: str,
    cfg: StructuralReferenceExtractionConfig,
) -> list[StructuralAnchor]:
    """Extract heading-like structural anchors from document text.

    Anchors are derived from line-oriented heading patterns such as
    ``Schedule A: Services Description`` and ``4.2 Termination``.
    Returned anchors preserve document order, and ordinals are assigned
    sequentially across matched headings only.

    Args:
        text: Source document text to scan.
        cfg: Extraction configuration controlling key canonicalization.

    Returns:
        Ordered list of extracted structural anchors.
    """
    anchors: list[StructuralAnchor] = []
    ordinal = 0

    for match in _LINE_RE.finditer(text):
        raw_line = match.group(0)
        stripped = raw_line.strip()
        if not stripped:
            continue

        leading_ws = len(raw_line) - len(raw_line.lstrip())
        start_offset = match.start() + leading_ws
        end_offset = start_offset + len(stripped)

        anchor = _anchor_from_line(
            line=stripped,
            start_offset=start_offset,
            end_offset=end_offset,
            ordinal=ordinal,
            cfg=cfg,
        )
        if anchor is None:
            continue

        anchors.append(anchor)
        ordinal += 1

    return anchors


def build_structural_anchor_index(
    anchors: list[StructuralAnchor],
) -> dict[str, list[StructuralAnchor]]:
    """Group structural anchors by lookup key in document order.

    Anchors are indexed by the lookup key stored in ``normalized_key``.
    Input order is preserved within each group to support deterministic
    downstream linking and proximity-based tie-breaking.

    Args:
        anchors: Structural anchors in document order.

    Returns:
        Mapping from lookup key to the ordered anchors sharing that key.
    """
    out: dict[str, list[StructuralAnchor]] = {}

    for anchor in anchors:
        out.setdefault(anchor.normalized_key, []).append(anchor)

    return out
