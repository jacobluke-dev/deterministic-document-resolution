import re

RNA_RE = re.compile(r"\b(?:mRNA|tRNA|rRNA|miRNA|siRNA|sgRNA|gRNA|lncRNA|snRNA|scRNA|cDNA|gDNA)\b")
CYTOKINE = re.compile(r"\b(?:IL-\d{1,3}|IFN-[\u03B1-\u03C9]|TNF-[\u03B1-\u03C9]|TGF-[\u03B1-\u03C9])\b")
VIRUS = re.compile(r"\b(?:SARS-CoV-2|MERS-CoV|H[1-9]N[1-9])\b")
PCR_RE = re.compile(r"\b(?:PCR|qPCR|RT-?qPCR|CRISPR|Cas9|ELISA|Western blot)\b", re.I)
UNITS = re.compile(r"\b(?:mg/dL|μL|uL|mM|μM|ng/mL|ug/mL|OD ?(?:600|260|280))\b")
STATS = re.compile(r"\b(?:95%\s*CI|confidence interval|odds ratio|hazard ratio|p\s*<\s*0\.\d+|HR\s*=?\s*\d)\b", re.I)
SECTIONS = re.compile(r"\b(?:Abstract|Methods?|Materials and Methods|Results|Discussion)\b")
GREEK = re.compile(r"[\u0370-\u03FF]")


STRONG = (
    ("rna", RNA_RE, 5),
    ("cytokine", CYTOKINE, 5),
    ("virus", VIRUS, 5),
)
SUPPORT = (
    ("pcr", PCR_RE, 2),
    ("units", UNITS, 1),
    ("stats", STATS, 2),
    ("sections", SECTIONS, 1),
    ("greek", GREEK, 1),
)


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
