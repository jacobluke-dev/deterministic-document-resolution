from .config import STRUCTURAL_STRONG_CUES, STRUCTURAL_SUPPORT_CUES


def structural_signal_score(text: str, cap: int = 80_000) -> tuple[int, list[str], bool]:
    """Score structural-reference signals found in a capped prefix of text.

    Evaluates the input against configured **strong** and **supporting**
    structural-reference cues (for example, schedules, appendices, sections,
    clauses). Each matched cue contributes a deterministic weight to the total
    score, and matched cue labels are collected into ``reasons``.

    Strong cues additionally set ``has_strong`` to ``True``. Repeated weak
    references such as multiple section or clause mentions can add a small
    bonus to improve recall for formally structured documents without requiring
    a strong cue.

    Args:
        text (str): Source document text to inspect.
        cap (int, optional): Maximum number of characters from the start of
            ``text`` to inspect. Defaults to ``80_000``.

    Returns:
        tuple[int, list[str], bool]:
            A 3-tuple of:

            - ``score``: Total weighted structural signal score.
            - ``reasons``: Labels for each matched cue contributing to the score.
            - ``has_strong``: ``True`` if any strong cue matched; otherwise
              ``False``.

    Notes:
        - Matching is deterministic and side-effect free.
        - The function operates only on the first ``cap`` characters.
        - ``reasons`` may contain both direct cue labels and derived repetition
          labels such as ``"repeated_section"``.
    """
    t = text[:cap]
    score, reasons, has_strong = 0, [], False

    for label, pat, weight in STRUCTURAL_STRONG_CUES:
        if pat.search(t):
            score += weight
            reasons.append(label)
            has_strong = True

    support_hits = 0
    for label, pat, weight in STRUCTURAL_SUPPORT_CUES:
        if pat.search(t):
            score += weight
            reasons.append(label)
            support_hits += 1

    # repeated weak cues can also indicate formal structure
    repeated_section_hits = len(next(pat for label, pat, _ in STRUCTURAL_SUPPORT_CUES if label == "section").findall(t))
    repeated_clause_hits = len(next(pat for label, pat, _ in STRUCTURAL_SUPPORT_CUES if label == "clause").findall(t))

    if repeated_section_hits >= 2:
        score += 2
        reasons.append("repeated_section")

    if repeated_clause_hits >= 2:
        score += 2
        reasons.append("repeated_clause")

    return score, reasons, has_strong


def should_enable_structural_reference(
    text: str,
    *,
    threshold: int = 4,
    cap: int = 80_000,
) -> tuple[bool, list[str]]:
    """Decide whether to enable the structural-reference domain for a document.

    Runs structural signal scoring over a capped prefix of the input text and
    returns whether the document should be considered structurally referential
    enough to enable downstream structural-reference logic.

    The domain is enabled if either:

    - at least one **strong** structural cue is present; or
    - the total weighted score meets or exceeds ``threshold``.

    Args:
        text (str): Source document text to inspect.
        threshold (int, optional): Minimum score required to enable the
            structural-reference domain when no strong cue is present.
            Defaults to ``4``.
        cap (int, optional): Maximum number of characters from the start of
            ``text`` to inspect. Defaults to ``80_000``.

    Returns:
        tuple[bool, list[str]]:
            A 2-tuple of:

            - ``enabled``: ``True`` if the structural-reference domain should
              be enabled; otherwise ``False``.
            - ``reasons``: Labels describing which cues contributed to the
              decision.

    Notes:
        This function is intended for lightweight, deterministic domain gating.
        It does not perform structural-reference extraction.
    """
    score, reasons, has_strong = structural_signal_score(text, cap=cap)
    return (has_strong or score >= threshold), reasons
