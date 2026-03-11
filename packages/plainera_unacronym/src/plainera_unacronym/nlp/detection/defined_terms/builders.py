from __future__ import annotations

from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str

from .normalise import normalize_defined_term_key
from .types import DefinedTermOccurrence, DefinedTermIntroduction


def build_defined_term_intro(
    *,
    term: str,
    term_start: int,
    term_end: int,
    provenance: str,
) -> DefinedTermIntroduction:
    """Build a canonical defined-term intro from an introduced term span.

    The raw term text is trimmed, stripped of surrounding straight quotes, cleaned
    of trailing punctuation noise, and normalised into a stable lookup key. A
    deterministic sense identifier is then derived from the normalised key and
    start offset.

    Args:
        term: Raw detected term text from an introduction pattern.
        term_start: Inclusive start offset of the detected term in the source text.
        term_end: Exclusive end offset of the detected term in the source text.
        provenance: Source label describing how the term was produced.

    Returns:
        A ``DefinedTermIntroduction`` containing the cleaned term text, source offsets,
        normalised lookup key, and provenance.
    """
    cleaned_term = strip_trailing_punct_str(term.strip().strip('"'))
    normalized_key = normalize_defined_term_key(cleaned_term)

    return DefinedTermIntroduction(
        term=cleaned_term,
        start_offset=term_start,
        end_offset=term_end,
        normalized_key=normalized_key,
        provenance=provenance,
    )


def build_defined_term_occurrence(
    *,
    term: str,
    start_offset: int,
    end_offset: int,
    segment_window: str | None = None,
    confidence: float = 1.0,
) -> DefinedTermOccurrence:
    """Build a defined-term occurrence from a detected reference span.

    The raw term text is trimmed, stripped of surrounding straight quotes, cleaned
    of trailing punctuation noise, and normalised into a stable lookup key before
    being packaged as an occurrence object.

    Args:
        term: Raw detected term text from an occurrence pattern.
        start_offset: Inclusive start offset of the detected occurrence in the
            source text.
        end_offset: Exclusive end offset of the detected occurrence in the source
            text.
        segment_window: Optional surrounding text window to retain for debugging,
            ranking, or downstream context.
        confidence: Confidence score assigned to the occurrence.

    Returns:
        A ``DefinedTermOccurrence`` containing the cleaned term text, source
        offsets, normalised lookup key, confidence score, and optional segment
        window.
    """
    cleaned_term = strip_trailing_punct_str(term.strip().strip('"'))
    normalized_key = normalize_defined_term_key(cleaned_term)

    return DefinedTermOccurrence(
        term=cleaned_term,
        start_offset=start_offset,
        end_offset=end_offset,
        normalized_key=normalized_key,
        occurrence_confidence=confidence,
        segment_window=segment_window,
    )
