from __future__ import annotations

import re

from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetectorResult
from plainera_unacronym.nlp.extraction.defined_terms.structure import TermStructureIndex
from plainera_unacronym.nlp.extraction.defined_terms.types import TermDefinitionEntry

_MEANS_START_RE = re.compile(
    r'^\s*"?\s*(?:shall\s+mean|means)\b\s*',
    re.IGNORECASE,
)


def _as_text_span(text: str, start: int, end: int):
    return text[start:end], start, end


def _resolve_definition_start(
    text: str,
    intro_end: int,
    *,
    intro_kind: str,
) -> int | None:
    """
    Return the start offset of the actual definition body, or None if this
    introduction form should not produce a trailing definition span.
    """
    if intro_kind == "parenthetical_alias":
        return None

    tail = text[intro_end:]

    m = _MEANS_START_RE.match(tail)
    if not m:
        return None

    return intro_end + m.end()


def _find_definition_end(
    text: str,
    start: int,
    *,
    max_chars: int = 400,
) -> int | None:
    if start >= len(text):
        return None

    limit = min(len(text), start + max_chars)
    chunk = text[start:limit]

    if not chunk.strip():
        return None

    stop_candidates: list[int] = []

    for marker in ("\n\n", ";", "."):
        idx = chunk.find(marker)
        if idx != -1:
            stop_candidates.append(idx)

    end = start + (min(stop_candidates) if stop_candidates else len(chunk))
    return end if end > start else None


def extract_term_definitions(
    *,
    text: str,
    detector_result: DefinedTermDetectorResult,
    structure_index: TermStructureIndex | None,
    max_definition_chars: int = 400,
) -> list[TermDefinitionEntry]:
    entries: list[TermDefinitionEntry] = []

    for intro in detector_result.introductions:
        start = intro.start_offset
        end = intro.end_offset
        intro_kind = intro.intro_kind

        definition_start = _resolve_definition_start(
            text,
            end,
            intro_kind=intro_kind,
        )

        if definition_start is None:
            definition_span = None
            definition_text = None
        else:
            definition_end = _find_definition_end(
                text,
                definition_start,
                max_chars=max_definition_chars,
            )
            definition_span = (
                _as_text_span(text, definition_start, definition_end) if definition_end is not None else None
            )
            definition_text = definition_span[0] if definition_span is not None else None

        section_path = structure_index.path_for_offset(start) if structure_index else ("document",)

        entries.append(
            TermDefinitionEntry(
                surface=intro.term,
                normalized_key=intro.normalized_key,
                intro_span=_as_text_span(text, start, end),
                definition_span=definition_span,
                definition_text=definition_text,
                intro_kind=intro_kind,
                section_path=section_path,
            )
        )

    entries.sort(key=lambda e: (e.intro_span[1], e.intro_span[2]))
    return entries
