import re

from plainera_unacronym.nlp.common.constants_regex import INLINE_CUE_FRAGMENTS


def _compile_inline_cues_pattern(cues: tuple[str, ...]) -> re.Pattern[str]:
    # single regex, detector-friendly
    joined = "|".join(cues)
    return re.compile(
        rf"\b(?:{joined})\b(?=[\s,;:—–-])",
        flags=re.IGNORECASE,
    )


def boost_confidence_if_inline_cue(surface: str, text: str, e: int, conf: float) -> float:
    # Only bother for short acronyms
    if len(surface) > 3:
        return conf

    # Look rightwards: "AM, short for ..." is the canonical form
    right = text[e : min(len(text), e + 60)]
    if _compile_inline_cues_pattern(INLINE_CUE_FRAGMENTS).search(right):
        return min(conf + 0.20, 0.99)

    return conf
