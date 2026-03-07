from plainera_unacronym.nlp.detection.domains.bio.config import SUPPORT, STRONG


def bio_signal_score(text: str, cap: int = 80_000) -> tuple[int, list[str], bool]:
    """
    Compute a lightweight heuristic score for biomedical / life-sciences signals.

    The function scans the first `cap` characters of `text` using two cue tiers:
    - STRONG: high-confidence biomedical markers (e.g., RNA types, cytokines, viruses).
    - SUPPORT: contextual markers (e.g., PCR terms, lab units, stats phrasing, section headings, Greek letters).

    Each cue contributes a fixed weight once (presence/absence), not per occurrence.

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
    Decide whether to enable "bio mode" for downstream processing based on cue scoring.

    Uses `bio_signal_score` and applies a conservative gating rule:
    - Enable if at least one STRONG cue matched, OR
    - Enable if the overall score is high even without a STRONG cue (score >= threshold + 3).

    This biases toward precision: a single strong biomedical marker is decisive,
    while multiple weaker signals must accumulate to surpass the higher score bar.

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
