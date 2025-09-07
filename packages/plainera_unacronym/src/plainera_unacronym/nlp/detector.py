from src.plainera_unacronym.nlp.heuristics import context_window, score, iter_candidates, normalize_key
from src.plainera_unacronym.nlp.types import DetectorConfig, DetectorResult, Occurrence, FirstOccurrence

DEFAULT_CONFIG = DetectorConfig()


def detect_acronyms(text: str, config: DetectorConfig = DEFAULT_CONFIG) -> DetectorResult:
    """
    One-pass detector. Returns stable schema + first-occurrence map with normalized keys.
    """
    occurrences: list[Occurrence] = []
    firsts: dict[str, FirstOccurrence] = {}

    for surface, s, e in iter_candidates(text, config):
        if blacklist_context_drop(surface, text, s, e, config):
            continue

        conf = score(surface, text, s, e, config)
        ctx = context_window(text, s, e, config.window_chars)

        occ = Occurrence(
            acronym=surface,
            start_offset=s,
            end_offset=e,
            confidence=conf,
            context_window=ctx,
        )
        occurrences.append(occ)

        key = normalize_key(surface)
        if key not in firsts:
            firsts[key] = FirstOccurrence(
                acronym=surface,
                start_offset=s,
                end_offset=e,
                confidence=conf,
            )

    return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)
