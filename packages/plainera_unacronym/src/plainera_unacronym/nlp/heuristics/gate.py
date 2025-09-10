import re

RNA_RE   = re.compile(r"\b(?:mRNA|tRNA|rRNA|miRNA|siRNA|sgRNA|gRNA|lncRNA|snRNA|scRNA|cDNA|gDNA)\b")
CYTOKINE = re.compile(r"\b(?:IL-\d{1,3}|IFN-[\u03B1-\u03C9]|TNF-[\u03B1-\u03C9]|TGF-[\u03B1-\u03C9])\b")
VIRUS    = re.compile(r"\b(?:SARS-CoV-2|MERS-CoV|H[1-9]N[1-9])\b")
PCR_RE   = re.compile(r"\b(?:PCR|qPCR|RT-?qPCR|CRISPR|Cas9|ELISA|Western blot)\b", re.I)
UNITS    = re.compile(r"\b(?:mg/dL|μL|uL|mM|μM|ng/mL|ug/mL|OD ?(?:600|260|280))\b")
STATS    = re.compile(r"\b(?:95%\s*CI|confidence interval|odds ratio|hazard ratio|p\s*<\s*0\.\d+|HR\s*=?\s*\d)\b", re.I)
SECTIONS = re.compile(r"\b(?:Abstract|Methods?|Materials and Methods|Results|Discussion)\b")
GREEK    = re.compile(r"[\u0370-\u03FF]")

def _slice(text: str, max_chars: int = 80_000) -> str: return text[:max_chars]

def bio_signal_score(text: str) -> tuple[int,list[str]]:
    t = _slice(text)
    score, reasons = 0, []
    for label, pat, w in (
        ("rna", RNA_RE, 3), ("cytokine", CYTOKINE, 3), ("virus", VIRUS, 3),
        ("pcr", PCR_RE, 2), ("units", UNITS, 1), ("stats", STATS, 2),
        ("sections", SECTIONS, 1),
    ):
        if pat.search(t): score += w; reasons.append(label)
    if GREEK.search(t): score += 1; reasons.append("greek")
    return score, reasons

def should_enable_bio(text: str, threshold: int = 3) -> tuple[bool,list[str]]:
    score, reasons = bio_signal_score(text)
    return (score >= threshold), reasons
