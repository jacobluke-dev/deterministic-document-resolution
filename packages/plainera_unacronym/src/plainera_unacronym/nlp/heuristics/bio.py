from __future__ import annotations
import re
from typing import Iterator
from plainera_unacronym.nlp.types import DetectorConfig

BIO_GREEK = "\u03B1-\u03C9"  # α–ω

def _compile_bio_pattern() -> re.Pattern[str]:
    camel   = r"(?:[A-Z][a-z]+[A-Z0-9][A-Za-z0-9]{1,6}|[A-Z]{2,6}\d{0,2})"
    cytokine= rf"(?:IL-\d{{1,3}}|TNF-[{BIO_GREEK}]|IFN-[{BIO_GREEK}]|TGF-[{BIO_GREEK}\d])"
    virus   = r"(?:SARS-CoV-2|MERS-CoV|H\dN\d)"
    prime   = r"(?:[35][\'′″]-?\s?UTR)"
    return re.compile(rf"(?P<bio>{cytokine}|{virus}|{prime}|{camel})")

def extra_candidates(text: str, cfg: DetectorConfig) -> Iterator[tuple[str,int,int]]:
    if not cfg.enable_bio: return
    pat = _compile_bio_pattern()
    for m in pat.finditer(text):
        s, e = m.span("bio")
        yield text[s:e], s, e

STATS_CI_RE = re.compile(r"\b\d{1,3}%\s*CI\b")
STATS_OR_HR_RR_RE = re.compile(r"\b(OR|HR|RR)\s*(?:=|≈|~)?\s*\d")

def bio_keep_guard(surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
    if not cfg.enable_bio: return False
    if surface in cfg.bio_rna_like:  # e.g., mRNA, miRNA, sgRNA
        return True
    if len(surface) == 2 and surface in cfg.bio_2letter_keep:
        r = text[max(0, s-20):min(len(text), e+20)]
        if STATS_CI_RE.search(r) or STATS_OR_HR_RR_RE.search(r):
            return True
    return False
