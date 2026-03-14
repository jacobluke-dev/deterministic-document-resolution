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
    span: Span
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TermStructureIndex:
    paths_by_span: tuple[StructurePathEntry, ...]

    def path_for_offset(self, offset: int) -> tuple[str, ...]:
        for entry in self.paths_by_span:
            start, end = entry.span
            if start <= offset < end:
                return entry.path
        return ("document",)


def _line_spans(text: str) -> list[TextSpanTuple]:
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
    if current_end > current_start:
        entries.append(StructurePathEntry(span=(current_start, current_end), path=current_path))


def build_term_structure_index(text: str) -> TermStructureIndex:
    """
    Build a lightweight structure index from section/schedule-like headings.

    This is intentionally conservative. If no headings are found, the whole
    document becomes a single ('document',) block.
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
