from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str

from .normalise import normalize_defined_term_key
from .types import DefinedTermOccurrence, DefinedTermSense


def build_defined_term_sense(
    *,
    term: str,
    term_start: int,
    term_end: int,
    definition_text: str,
    definition_start: int,
    definition_end: int,
    provenance: str,
) -> DefinedTermSense:
    cleaned_term = strip_trailing_punct_str(term.strip().strip('"'))
    normalized_key = normalize_defined_term_key(cleaned_term)
    sense_id = f"term|{normalized_key}|{term_start}"

    return DefinedTermSense(
        term=cleaned_term,
        definition_text=definition_text.strip(),
        start_offset=term_start,
        end_offset=term_end,
        definition_start_offset=definition_start,
        definition_end_offset=definition_end,
        normalized_key=normalized_key,
        sense_id=sense_id,
        provenance=provenance,
        confidence=1.0,
    )


def build_defined_term_occurrence(
    *,
    term: str,
    start_offset: int,
    end_offset: int,
    segment_window: str | None = None,
    confidence: float = 1.0,
) -> DefinedTermOccurrence:
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
