from document_resolution.nlp.common.types import AcronymDetectorConfig, AcronymDetectorResult, Occurrence

from .common import DroppedOccurrence, RuleFn
from .core import recompute_firsts
from .rules import (
    rule_contained_suffix,
    rule_drop_mixed_case_typos,
    rule_end_suffix_micro,
    rule_inside_paren_suffix_of_left_acronym,
    rule_token_before_paren_suffix,
)

# Tier-1: narrow, deterministic, test-backed rules.
RULES_SAFE: tuple[RuleFn, ...] = (
    rule_inside_paren_suffix_of_left_acronym,  # "(RNA)" right after mRNA -> drop RNA inside parens
    rule_token_before_paren_suffix,  # "... RNA (mRNA)" -> drop RNA before parens
    rule_contained_suffix,  # RNA fully inside mRNA span -> drop RNA
    rule_end_suffix_micro,  # same end offset, shorter suffix of longer -> drop shorter
    rule_drop_mixed_case_typos,  # likely internal-case OCR/typo weirdness
)

# Optional alias if later formalise tiers across the pipeline:
# This will come later if I go with this naming convention
# RULES_TIER1 = RULES_SAFE


def post_detect_cleanup(
    text: str,
    det: AcronymDetectorResult,
    cfg: AcronymDetectorConfig,
) -> tuple[AcronymDetectorResult, str, list[DroppedOccurrence]]:
    """Applies deterministic post-detection clean up rules and recomputes first occurrences.

    This function runs a narrow, conservative rule pipeline over `det.occurrences` to
    remove obvious duplicate / fragment hits created by tokenisation or overlapping
    acronym shapes (e.g., dropping "RNA" when "mRNA" is present). After clean up, it
    recomputes `unique_acronyms` from the kept occurrences so the boundary is
    authoritative and consistent with the final occurrence list.

    Rule pipeline:
      - The active rule list is `RULES_SAFE` (ordering is significant).
      - Each rule receives `(text, occs)` and returns `(kept, dropped)` where `kept`
        is passed to the next rule.

    Args:
        text: Source text that produced the detector result. Some rules inspect the
            surrounding characters (e.g., parentheses boundaries); others ignore it.
        det: The detector output containing `occurrences` and `unique_acronyms`.
        cfg: Detector configuration used for normalisation and recomputing firsts.

    Returns:
        A tuple of:
          - cleaned: A new `DetectorResult` with occurrences after cleanup and
            `unique_acronyms` recomputed from the kept occurrences.
          - summary: A short human-readable summary (counts before/after).
          - dropped: A list of `DroppedOccurrence` records describing each dropped
            occurrence and the rule that dropped it.

    Notes:
        - Deterministic: given the same inputs, outputs are stable.
        - Does not mutate input occurrences; produces a filtered/sorted list as needed.
        - `unique_acronyms` in the returned result is derived from occurrences (single
          source of truth).
    """

    before = det.occurrences
    kept: list[Occurrence] = before
    dropped_all: list[DroppedOccurrence] = []

    for rule in RULES_SAFE:
        kept, dropped = rule(text, kept)
        dropped_all.extend(dropped)

    # Recompute unique_acronyms from kept occurrences (authoritative boundary)
    firsts = recompute_firsts(kept, cfg)

    cleaned = AcronymDetectorResult(unique_acronyms=firsts, occurrences=kept)
    summary = (
        f"cleanup occs {len(before)}→{len(kept)} "
        f"firsts={len(det.unique_acronyms)}→{len(firsts)} "
        f"dropped={len(dropped_all)}"
    )
    return cleaned, summary, dropped_all
