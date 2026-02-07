from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import DetectorConfig, Occurrence, FirstOccurrence


def recompute_firsts(
    text: str,
    occurrences: list[Occurrence],
    cfg: DetectorConfig,
) -> dict[str, FirstOccurrence]:
    firsts: dict[str, FirstOccurrence] = {}

    for o in occurrences:
        k = o.normalized_key
        if not k:
            k = normalize_acronym_key(o.acronym, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        if not k:
            continue

        prev = firsts.get(k)
        if prev is None or o.start_offset < prev.start_offset:
            firsts[k] = FirstOccurrence(
                acronym=o.acronym,
                start_offset=o.start_offset,
                end_offset=o.end_offset,
                confidence=o.confidence,
                normalized_key=k,
            )

    return firsts
