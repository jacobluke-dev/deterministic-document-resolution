from __future__ import annotations

from document_resolution.nlp.common.shared import strip_trailing_punct_str

from .normalise import normalize_defined_term_key
from .types import DefinedTermIntroduction, DefinedTermMention, IntroKind


def build_defined_term_intro(
    *,
    term: str,
    term_start: int,
    term_end: int,
    provenance: str,
    intro_kind: IntroKind,
) -> DefinedTermIntroduction:
    """Build a canonical defined-term introduction from a detected term span.

    Args:
        term: Raw detected term text.
        term_start: Inclusive start offset of the term.
        term_end: Exclusive end offset of the term.
        provenance: Source label describing how the term was produced.
        intro_kind: Introduction kind of the term.

    Returns:
        Defined-term introduction with cleaned term text and normalised lookup key.
    """
    cleaned_term = strip_trailing_punct_str(term.strip().strip('"'))
    normalized_key = normalize_defined_term_key(cleaned_term)

    return DefinedTermIntroduction(
        term=cleaned_term,
        start_offset=term_start,
        end_offset=term_end,
        normalized_key=normalized_key,
        provenance=provenance,
        intro_kind=intro_kind,
    )


def build_defined_term_mention(
    *,
    term: str,
    start_offset: int,
    end_offset: int,
    segment_window: str | None = None,
    confidence: float = 1.0,
) -> DefinedTermMention:
    """Build a defined-term mention from a detected reference span.

    Args:
        term: Raw detected term text.
        start_offset: Inclusive start offset of the mention.
        end_offset: Exclusive end offset of the mention.
        segment_window: Optional surrounding text window.
        confidence: Confidence score for the mention.

    Returns:
        Defined-term mention with cleaned term text and normalised lookup key.
    """
    cleaned_term = strip_trailing_punct_str(term.strip().strip('"'))
    normalized_key = normalize_defined_term_key(cleaned_term)

    return DefinedTermMention(
        term=cleaned_term,
        start_offset=start_offset,
        end_offset=end_offset,
        normalized_key=normalized_key,
        confidence=confidence,
        segment_window=segment_window,
    )
