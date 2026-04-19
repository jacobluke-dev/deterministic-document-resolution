import re

from document_resolution.nlp.common.constants_regex import INLINE_CUE_FRAGMENTS


def _compile_inline_cues_pattern(cues: tuple[str, ...]) -> re.Pattern[str]:
    """
    Compile a single, detector-friendly regex for inline definition cues.

    Args:
        cues: Tuple of regex fragments representing cue phrases (no flags, no \b wrappers).

    Returns:
        Compiled, case-insensitive regex pattern that matches any cue phrase.
    """
    joined = "|".join(cues)
    return re.compile(
        rf"\b(?:{joined})\b(?=[\s,;:—–-])",
        flags=re.IGNORECASE,
    )


_INLINE_CUES_RE = _compile_inline_cues_pattern(INLINE_CUE_FRAGMENTS)

def boost_confidence_if_inline_cue(surface: str, text: str, e: int, conf: float) -> float:
    """
    Boost confidence for short acronyms if an inline cue appears immediately to the right.

    Args:
        surface: Matched acronym surface (e.g. "NLP").
        text: Full source text containing the match.
        e: End offset (exclusive) of the acronym in `text`.
        conf: Current confidence score.

    Returns:
        Updated confidence (adds +0.20 up to a max of 0.99) when a cue is present; else `conf`.
    """
    if len(surface) > 3:
        return conf

    # Look rightwards: "AM, short for ..." is the canonical form
    right = text[e : min(len(text), e + 60)]
    if _INLINE_CUES_RE.search(right):
        return min(conf + 0.20, 0.99)

    return conf
