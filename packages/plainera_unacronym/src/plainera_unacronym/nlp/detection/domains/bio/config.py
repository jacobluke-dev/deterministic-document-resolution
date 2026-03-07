import re
from dataclasses import dataclass
from typing import Optional

from plainera_unacronym.nlp.common.types import DetectorConfig

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


BIO_GREEK = "\u03b1-\u03c9"  # α–ω
_STATS_CI_RE = re.compile(r"\b\d{1,3}%\s*CI\b")
_STATS_OR_HR_RR_RE = re.compile(r"\b(OR|HR|RR)\s*(?:=|≈|~)?\s*\d")


@dataclass(frozen=True, slots=True)
class BioConfig(DetectorConfig):
    rna_like: frozenset[str] = frozenset({"mRNA", "miRNA", "sgRNA"})
    two_letter_keep: frozenset[str] = frozenset({"IL", "TN", "HR", "OR", "RR"})
    enable_bio: bool = False
    bio_rna_like: bool = False
    stats_window_chars: Optional[int] = 40
    bio_2letter_keep: bool = False
