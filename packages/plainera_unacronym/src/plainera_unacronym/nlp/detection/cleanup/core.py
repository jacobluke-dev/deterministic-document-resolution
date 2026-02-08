from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence


def recompute_firsts(
    occurrences: list[Occurrence],
    cfg: DetectorConfig,
) -> dict[str, FirstOccurrence]:
    """Recomputes first-occurrence metadata from a list of kept occurrences.

    Produces a mapping of `normalized_key -> FirstOccurrence`, selecting the earliest
    occurrence (lowest `start_offset`) per normalized key. This is used after cleanup
    so that `DetectorResult.unique_acronyms` is derived from the authoritative
    occurrence list rather than the pre-cleanup detector output.

    Normalization:
      - Prefer `Occurrence.normalized_key` when present.
      - Otherwise compute a key via `normalize_acronym_key()` using `cfg.allow_chars`
        and `cfg.dotted_display`.
      - If a key cannot be computed, the occurrence is ignored.

    Args:
        occurrences: Occurrences to derive first occurrences from (typically the
            post-cleanup kept list). Input ordering is not assumed.
        cfg: Detector configuration used for key normalization.

    Returns:
        A dict mapping normalized keys to `FirstOccurrence` entries, where each entry
        corresponds to the earliest occurrence for that key.

    Notes:
        - Deterministic: ties on start offset are resolved by first-seen iteration
          order (stable for a fixed input list).
        - Does not mutate the input occurrences.
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
                confidence=o.confidence,
                normalized_key=k,
            )

    return firsts
