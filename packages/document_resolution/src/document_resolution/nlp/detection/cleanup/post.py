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


def post_detect_cleanup(
    text: str,
    det: AcronymDetectorResult,
    cfg: AcronymDetectorConfig,
) -> tuple[AcronymDetectorResult, str, list[DroppedOccurrence]]:
    """Apply deterministic post-detection cleanup rules and recompute firsts.

    Runs the configured cleanup rules over `det.occurrences`, then rebuilds
    `unique_acronyms` from the kept occurrences.

    Args:
        text: Source text used by cleanup rules.
        det: Detector result before cleanup.
        cfg: Detector configuration used when recomputing first occurrences.

    Returns:
        Cleaned detector result, a short summary string, and dropped-occurrence records.
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
