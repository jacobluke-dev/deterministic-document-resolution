from document_resolution.nlp.common.shared import normalize_acronym_key
from document_resolution.nlp.common.types import AcronymDetectorConfig, FirstOccurrence, Occurrence


def recompute_firsts(
    occurrences: list[Occurrence],
    cfg: AcronymDetectorConfig,
) -> dict[str, FirstOccurrence]:
    """Recompute first-occurrence metadata from kept occurrences.

    Prefers `Occurrence.normalized_key` when present, otherwise recomputes the key
    from the occurrence acronym and detector config.

    Args:
        occurrences: Occurrences to derive first occurrences from.
        cfg: Detector configuration used for key normalisation.

    Returns:
        Mapping of normalised key to the earliest occurrence for that key.
    """

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
                occurrence_confidence=o.occurrence_confidence,
                normalized_key=k,
            )

    return firsts
