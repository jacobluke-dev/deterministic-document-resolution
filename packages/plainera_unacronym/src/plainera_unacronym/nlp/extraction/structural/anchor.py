from __future__ import annotations

import re

from plainera_unacronym.nlp.extraction.structural.common import is_strict_roman_numeral, roman_to_int
from plainera_unacronym.nlp.extraction.structural.config import StructuralReferenceExtractionConfig
from plainera_unacronym.nlp.extraction.structural.types import StructuralAnchor

_LINE_RE = re.compile(r"(?m)^[^\n]*\S[^\n]*$")

_NAMED_HEADING_RE = re.compile(
    r"""
    ^
    (?P<kind>Schedule|Section|Clause|Article|Appendix|Annex|Exhibit)
    \s+
    (?P<label>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)
    \b
    (?:\s*[:.\-–]\s*.*)?   # optional title suffix
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
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _build_anchor_key(
    *,
    kind: str,
    label: str,
    cfg: StructuralReferenceExtractionConfig,
) -> str:
    normalized_kind = kind.lower()
    normalized_label = label.strip()

    if cfg.convert_roman_numerals and normalized_kind == "article" and is_strict_roman_numeral(normalized_label):
        normalized_label = str(roman_to_int(normalized_label))

    return f"{normalized_kind}_{_slug(normalized_label)}"


def _named_anchor_from_line(
    *,
    line: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor | None:
    match = _NAMED_HEADING_RE.match(line)
    if match is None:
        return None

    kind = match.group("kind")
    label = match.group("label")
    key = _build_anchor_key(kind=kind, label=label, cfg=cfg)

    return StructuralAnchor(
        label=label,
        normalized_key=key,
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
    )


def _numbered_section_anchor_from_line(
    *,
    line: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor | None:
    match = _NUMBERED_SECTION_HEADING_RE.match(line)
    if match is None:
        return None

    label = match.group("label")
    key = _build_anchor_key(kind="Section", label=label, cfg=cfg)

    return StructuralAnchor(
        label=label,
        normalized_key=key,
        start_offset=start_offset,
        end_offset=end_offset,
        ordinal=ordinal,
    )


def _anchor_from_line(
    *,
    line: str,
    start_offset: int,
    end_offset: int,
    ordinal: int,
    cfg: StructuralReferenceExtractionConfig,
) -> StructuralAnchor | None:
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
    ``Schedule A: Services Description`` or ``4.2 Termination``.
    Output order is document order and ordinals are assigned sequentially.
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

    Anchors are indexed by ``normalized_key`` and preserved in input order.
    This supports deterministic downstream linking and proximity-based
    tie-breaking across repeated headings.

    Args:
        anchors: Structural anchors in document order.

    Returns:
        Mapping from anchor lookup key to the ordered anchors sharing that key.
    """
    out: dict[str, list[StructuralAnchor]] = {}

    for anchor in anchors:
        out.setdefault(anchor.normalized_key, []).append(anchor)

    return out
