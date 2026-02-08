import re
from dataclasses import dataclass
from typing import Optional

from plainera_unacronym.nlp.common.types import DetectorConfig

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
