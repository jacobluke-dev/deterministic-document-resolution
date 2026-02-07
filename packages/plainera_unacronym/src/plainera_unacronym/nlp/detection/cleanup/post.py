from plainera_unacronym.nlp.common.types import DetectorResult, DetectorConfig, Occurrence

from .common import DroppedOccurrence, RuleFn
from .core import recompute_firsts
from .rules import (
    rule_inside_paren_suffix_of_left_acronym,
    rule_token_before_paren_suffix,
    rule_contained_suffix,
    rule_end_suffix_micro,
    rule_drop_mixed_case_typos,
)


# Tier-1: narrow, deterministic, test-backed rules.
RULES_TIER1: tuple[RuleFn, ...] = (
    rule_inside_paren_suffix_of_left_acronym,  # "(RNA)" right after mRNA -> drop RNA inside parens
    rule_token_before_paren_suffix,            # "... RNA (mRNA)" -> drop RNA before parens
    rule_contained_suffix,                     # RNA fully inside mRNA span -> drop RNA
    rule_end_suffix_micro,                     # same end offset, shorter suffix of longer -> drop shorter
    rule_drop_mixed_case_typos,                # likely internal-case OCR/typo weirdness
)


def post_detect_cleanup(
    text: str,
    det: DetectorResult,
    cfg: DetectorConfig,
) -> tuple[DetectorResult, str, list[DroppedOccurrence]]:
    """
    Post-detection cleanup (detect -> anchored boundary).

    Rule set: RULES_TIER1.
    """
    before = det.occurrences
    kept: list[Occurrence] = before
    dropped_all: list[DroppedOccurrence] = []

    for rule in RULES_TIER1:
        kept, dropped = rule(text, kept)
        dropped_all.extend(dropped)

    # Recompute unique_acronyms from kept occurrences (authoritative boundary)
    firsts = recompute_firsts(text, kept, cfg)

    cleaned = DetectorResult(unique_acronyms=firsts, occurrences=kept)
    summary = (
        f"cleanup occs {len(before)}→{len(kept)} "
        f"firsts={len(det.unique_acronyms)}→{len(firsts)} "
        f"dropped={len(dropped_all)}"
    )
    return cleaned, summary, dropped_all
