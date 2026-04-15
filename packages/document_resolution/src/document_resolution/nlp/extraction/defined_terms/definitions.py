from __future__ import annotations

import re

from document_resolution.nlp.common.types import TextSpanTuple
from document_resolution.nlp.detection.defined_terms import DefinedTermDetectorResult
from document_resolution.nlp.extraction.defined_terms.structure import TermStructureIndex
from document_resolution.nlp.extraction.defined_terms.types import TermDefinitionEntry

_MEANS_START_RE = re.compile(
    r'^\s*"?\s*(?:shall\s+mean|means)\b\s*',
    re.IGNORECASE,
)


def _as_text_span(text: str, start: int, end: int) -> TextSpanTuple:
    """Return a text span tuple for a slice of the source text.

    Args:
        text: Full source text.
        start: Inclusive start offset of the span.
        end: Exclusive end offset of the span.

    Returns:
        A tuple of ``(span_text, start, end)`` for the requested slice.
    """
    return text[start:end], start, end


def _resolve_definition_start(
    text: str,
    intro_end: int,
    *,
    intro_kind: str,
) -> int | None:
    """Resolve the starting offset of the trailing definition body.

    For introduction kinds that support a trailing ``means`` or ``shall mean``
    clause, this helper skips the definitional anchor and returns the start of
    the actual definition text. Parenthetical aliases do not produce trailing
    definition spans and therefore return ``None``.

    Args:
        text: Full source text.
        intro_end: Exclusive end offset of the introduction term span.
        intro_kind: Introduction kind recorded for the detected term.

    Returns:
        The start offset of the definition body, or ``None`` when the
        introduction form does not support trailing definition extraction.
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
    """Find the end offset of a trailing definition fragment.

    Scans forward from ``start`` up to ``max_chars`` and stops at the earliest
    recognised boundary marker such as a blank line, semicolon, or period.

    Args:
        text: Full source text.
        start: Inclusive start offset of the definition body.
        max_chars: Maximum number of characters to scan when searching for a
            definition boundary.

    Returns:
        The exclusive end offset of the extracted definition fragment, or
        ``None`` when no non-whitespace definition content is available.
    """
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


def _extract_parenthetical_alias_target(
    text: str,
    intro_span: TextSpanTuple,
) -> tuple[TextSpanTuple | None, str | None]:
    """Extract the antecedent phrase for a parenthetical alias introduction.

    Walks left from the alias parenthetical opening bracket and captures the
    immediately preceding phrase up to the nearest hard boundary. This is used
    for introduction forms such as:

        This Master Services Agreement (the "Agreement")
        Acme Limited (the "Supplier")

    The extraction is intentionally conservative. It does not attempt full
    noun-phrase parsing and stops at simple sentence or line boundaries to
    avoid swallowing trailing clause text from earlier context.

    Args:
        text: Full source text containing the introduction.
        intro_span: Span of the alias term itself, for example
            ``("Agreement", 37, 46)``.

    Returns:
        A tuple of:
            - the extracted antecedent span as ``(span_text, start, end)``, or
              ``None`` if no plausible antecedent could be found
            - the extracted antecedent text, or ``None`` if not found
    """
    _, intro_start, _ = intro_span

    open_paren = text.rfind("(", 0, intro_start)
    if open_paren == -1:
        return None, None

    right = open_paren
    left = right - 1

    while left >= 0 and text[left].isspace():
        left -= 1

    if left < 0:
        return None, None

    hard_stops = {".", ";", ":", "\n"}
    while left >= 0 and text[left] not in hard_stops:
        left -= 1

    start = left + 1
    candidate = text[start:right].strip()
    if not candidate:
        return None, None

    trimmed_start = text.index(candidate, start, right)
    trimmed_end = trimmed_start + len(candidate)

    return (candidate, trimmed_start, trimmed_end), candidate


def extract_term_definitions(
    *,
    text: str,
    detector_result: DefinedTermDetectorResult,
    structure_index: TermStructureIndex | None,
    max_definition_chars: int = 400,
) -> list[TermDefinitionEntry]:
    """Extract definition entries from detected defined-term introductions.

    For each introduction, this function records the introduction span, attempts
    to extract a trailing definition span and text when the introduction form
    supports it, and attaches structural section-path context from the document
    structure index.

    Args:
        text: Full source text containing the detected introductions.
        detector_result: Detector output containing introduced defined terms.
        structure_index: Optional structure index used to map introduction
            offsets to section paths.
        max_definition_chars: Maximum number of characters to scan when
            extracting trailing definition text.

    Returns:
        A list of ``TermDefinitionEntry`` objects sorted by introduction span
        order.
    """
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
        alias_target_span = None
        alias_target_text = None

        if intro_kind == "parenthetical_alias":
            alias_target_span, alias_target_text = _extract_parenthetical_alias_target(
                text=text,
                intro_span=(intro.term, start, end),
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
                alias_target_span=alias_target_span,
                alias_target_text=alias_target_text,
                section_path=section_path,
            )
        )

    entries.sort(key=lambda e: (e.intro_span[1], e.intro_span[2]))
    return entries
