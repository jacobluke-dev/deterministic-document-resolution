from document_resolution.nlp.detection.domains.bio.config import STRONG, SUPPORT


def bio_signal_score(text: str, cap: int = 80_000) -> tuple[int, list[str], bool]:
    """Score biomedical/life-sciences signals in the input text.

    Scans up to `cap` characters and awards each matched cue once. Returns the
    total score, matched cue labels, and whether any strong cue was present.

    Args:
        text: Input text to score.
        cap: Maximum number of characters to scan (performance guard). Defaults to 80,000.

    Returns:
        A tuple of:
        - score: Total weighted score from all matched cues.
        - reasons: Labels of cues that matched, in scan order (STRONG first, then SUPPORT).
        - has_strong: True if any STRONG cue matched; otherwise False.
    """
    t = text[:cap]
    score, reasons, has_strong = 0, [], False

    for label, pat, w in STRONG:
        if pat.search(t):
            score += w
            reasons.append(label)
            has_strong = True

    for label, pat, w in SUPPORT:
        if pat.search(t):
            score += w
            reasons.append(label)

    return score, reasons, has_strong


def should_enable_bio(text: str, threshold: int = 5) -> tuple[bool, list[str]]:
    """
    Decide whether biomedical mode should be enabled for the input text.

    Enables when at least one strong cue is present, or when weaker cues
    accumulate to a sufficiently high score.

    Args:
        text: Input text to evaluate.
        threshold: Base score threshold used to define "high score" texts. Defaults to 5.

    Returns:
        A tuple of:
        - enable: True if bio mode should be enabled; otherwise False.
        - reasons: Labels of cues that matched (for logging / explainability).
    """
    score, reasons, has_strong = bio_signal_score(text)
    # Require at least one strong signal, OR a high total score.
    enable = has_strong or score >= threshold + 3
    return enable, reasons
