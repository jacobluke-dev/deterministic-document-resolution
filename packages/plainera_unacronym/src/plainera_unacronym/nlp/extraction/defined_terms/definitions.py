from __future__ import annotations

from plainera_unacronym.nlp.common.types import TextSpanTuple, Span
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetectorResult
from plainera_unacronym.nlp.extraction.defined_terms.structure import TermStructureIndex
from plainera_unacronym.nlp.extraction.defined_terms.types import TermDefinitionEntry


def _as_text_span(text: str, start: int, end: int) -> TextSpanTuple:
    return text[start:end], start, end


def _get_intro_start(rec: object) -> int:
    intro_span = getattr(rec, "intro_span", None)
    if intro_span is not None:
        return int(intro_span[1])

    for name in ("start_offset", "intro_start_offset"):
        value = getattr(rec, name, None)
        if value is not None:
            return int(value)

    raise AttributeError("Could not resolve introduction start offset")


def _get_intro_end(rec: object) -> int:
    intro_span = getattr(rec, "intro_span", None)
    if intro_span is not None:
        return int(intro_span[2])

    for name in ("end_offset", "intro_end_offset"):
        value = getattr(rec, name, None)
        if value is not None:
            return int(value)

    raise AttributeError("Could not resolve introduction end offset")


def _get_surface(rec: object) -> str:
    for name in ("term", "surface"):
        value = getattr(rec, name, None)
        if value:
            return str(value)

    intro_span = getattr(rec, "intro_span", None)
    if intro_span is not None:
        return str(intro_span[0])

    raise AttributeError("Could not resolve term surface")


def _get_normalized_key(rec: object) -> str:
    value = getattr(rec, "normalized_key", None)
    if value:
        return str(value)
    raise AttributeError("Could not resolve normalized_key")


def _get_intro_kind(rec: object) -> str:
    for name in ("intro_kind", "introduction_kind", "kind"):
        value = getattr(rec, name, None)
        if value:
            return str(value)
    return "unknown"


def _find_definition_bounds(text: str, intro_end: int, *, max_chars: int = 400) -> Span | None:
    """
    Extract a conservative definition tail starting immediately after the intro.

    Stops at the first strong legal-ish boundary:
      * newline newline
      * ';'
      * '.'
    """
    if intro_end >= len(text):
        return None

    start = intro_end
    limit = min(len(text), intro_end + max_chars)
    chunk = text[start:limit]

    stripped = chunk.lstrip()
    if not stripped:
        return None

    leading_ws = len(chunk) - len(stripped)
    start += leading_ws
    chunk = stripped

    stop_candidates: list[int] = []

    for marker in ("\n\n", ";", "."):
        idx = chunk.find(marker)
        if idx != -1:
            stop_candidates.append(idx)

    end = start + (min(stop_candidates) if stop_candidates else len(chunk))
    if end <= start:
        return None

    return start, end


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

        def_bounds = _find_definition_bounds(text, end, max_chars=max_definition_chars)
        definition_span = _as_text_span(text, *def_bounds) if def_bounds is not None else None
        definition_text = definition_span[0] if definition_span is not None else None

        section_path = structure_index.path_for_offset(start) if structure_index else ("document",)

        entries.append(
            TermDefinitionEntry(
                surface=intro.term,
                normalized_key=intro.normalized_key,
                intro_span=_as_text_span(text, start, end),
                definition_span=definition_span,
                definition_text=definition_text,
                intro_kind="unknown",
                section_path=section_path,
            )
        )

    entries.sort(key=lambda e: (e.intro_span[1], e.intro_span[2]))
    return entries
