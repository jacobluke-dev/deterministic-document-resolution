from __future__ import annotations

import re
from dataclasses import dataclass

from plainera_unacronym.nlp.common.types import Span, TextSpanTuple

_SCHEDULE_RE = re.compile(
    r"^(?P<raw>(?P<kind>schedule|appendix|annex)\s+(?P<label>[A-Z0-9IVX]+)\b.*)$",
    re.IGNORECASE,
)

_SECTION_RE = re.compile(r"^(?P<raw>(?:(?:section|clause)\s+)?(?P<label>\d+(?:\.\d+)*)(?:[.)])?\s+[A-Z][^\n]{0,160})$")


@dataclass(frozen=True, slots=True)
class StructurePathEntry:
    """Map a contiguous text span to a lightweight structural document path.

    Attributes:
        span: Half-open ``(start, end)`` character span covered by the path.
        path: Structural path labels describing the block, for example
            ``("document",)``, ``("section:2",)``, or
            ``("schedule:A", "section:1.2")``.
    """
    span: Span
    path: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class TermStructureIndex:
    """Lookup index mapping source offsets to lightweight structural paths.

    Attributes:
        paths_by_span: Ordered span-to-path entries covering the detected
            structural blocks of the document.
    """
    paths_by_span: tuple[StructurePathEntry, ...]

    def path_for_offset(self, offset: int) -> tuple[str, ...]:
        """Return the structural path containing a given character offset.

        Args:
            offset: Character offset in the source document.

        Returns:
            The structural path for the matching span, or ``("document",)``
            when no more specific path is available.
        """
        for entry in self.paths_by_span:
            start, end = entry.span
            if start <= offset < end:
                return entry.path
        return ("document",)


def _line_spans(text: str) -> list[TextSpanTuple]:
    """Split text into line spans while preserving source offsets.

    Args:
        text: Full source text to segment into lines.

    Returns:
        A list of ``(line_text, start, end)`` tuples for each line. When the
        text contains no line breaks, a single span covering the whole text is
        returned.
    """
    spans: list[TextSpanTuple] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        start = pos
        pos += len(line)
        spans.append((line, start, pos))
    if not spans:
        spans.append((text, 0, len(text)))
    return spans


def _close_entry(
    entries: list[StructurePathEntry],
    current_start: int,
    current_end: int,
    current_path: tuple[str, ...],
) -> None:
    """Append a structure entry for the current open span when non-empty.

    Args:
        entries: Output list of accumulated structure entries.
        current_start: Inclusive start offset of the current structural block.
        current_end: Exclusive end offset of the current structural block.
        current_path: Structural path associated with the current block.
    """
    if current_end > current_start:
        entries.append(StructurePathEntry(span=(current_start, current_end), path=current_path))


def build_term_structure_index(text: str) -> TermStructureIndex:
    """Build a lightweight structure index from section- and schedule-like headings.

    The parser is intentionally conservative. It scans line-oriented headings and
    opens new structural blocks for recognised schedules, annexes, appendices,
    sections, and clauses. If no headings are found, the entire document is
    represented as a single ``("document",)`` span.

    Args:
        text: Full source text to analyse.

    Returns:
        A ``TermStructureIndex`` containing ordered span-to-path mappings for the
        detected structural blocks.
    """
    lines = _line_spans(text)
    entries: list[StructurePathEntry] = []

    current_schedule: str | None = None
    current_path: tuple[str, ...] = ("document",)
    current_start = 0

    for raw_line, start, _end in lines:
        line = raw_line.strip()
        if not line:
            continue

        m_sched = _SCHEDULE_RE.match(line)
        if m_sched:
            _close_entry(entries, current_start, start, current_path)

            label = m_sched.group("label").upper()
            current_schedule = label
            current_path = (f"schedule:{label}",)
            current_start = start
            continue

        m_sec = _SECTION_RE.match(line)
        if m_sec:
            _close_entry(entries, current_start, start, current_path)

            label = m_sec.group("label")
            if current_schedule:
                current_path = (f"schedule:{current_schedule}", f"section:{label}")
            else:
                current_path = (f"section:{label}",)
            current_start = start
            continue

    _close_entry(entries, current_start, len(text), current_path)

    if not entries:
        entries.append(StructurePathEntry(span=(0, len(text)), path=("document",)))

    return TermStructureIndex(paths_by_span=tuple(entries))
