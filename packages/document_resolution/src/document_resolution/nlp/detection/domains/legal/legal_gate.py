from .config import LEGAL_STRONG_CUES, LEGAL_SUPPORT_CUES


def legal_signal_score(text: str, cap: int = 80_000) -> tuple[int, list[str], bool]:
    t = text[:cap]
    score, reasons, has_strong = 0, [], False

    for label, pat, w in LEGAL_STRONG_CUES:
        if pat.search(t):
            score += w
            reasons.append(label)
            has_strong = True

    for label, pat, w in LEGAL_SUPPORT_CUES:
        if pat.search(t):
            score += w
            reasons.append(label)

    return score, reasons, has_strong


def should_enable_legal(text: str, *, threshold: int = 6, cap: int = 80_000) -> tuple[bool, list[str]]:
    score, reasons, has_strong = legal_signal_score(text, cap=cap)
    return (has_strong or score >= threshold), reasons
