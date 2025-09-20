import re

from plainera_unacronym.nlp.domains.bio.config import BioConfig
from plainera_unacronym.nlp.domains.bio.patterns import bio_pattern
from plainera_unacronym.nlp.domains.bio.rules import keep_guard as bio_keep_guard
from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.detection.heuristics.gate import should_enable_bio

from plainera_unacronym.nlp.plugins.registry import register_plugin

_BIO_SNIFF_RE = re.compile(r"\b(?:mRNA|miRNA|sgRNA|SARS-CoV-2|MERS-CoV|H\d{1,2}N\d{1,2}|IL-\d{1,3}|[35][′'\"]-?UTR)\b")


class BioPlugin:
    name = "bio"

    def _cfg(self, cfg: DetectorConfig) -> BioConfig:
        return cfg.domain_cfg.get(self.name, BioConfig())

    @staticmethod
    def sniff(self, text: str) -> bool:
        t = text[:80_000]  # cap scanning for speed
        if _BIO_SNIFF_RE.search(t):
            return True
        ok, _ = should_enable_bio(t)
        return ok

    def extra_candidates(self, text: str, cfg: DetectorConfig):
        if self.name not in cfg.enabled_domains:
            return
        pat = bio_pattern()
        for m in pat.finditer(text):
            s, e = m.span("bio")
            yield text[s:e], s, e

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
        if self.name not in cfg.enabled_domains:
            return False
        return bio_keep_guard(surface, text, s, e, self._cfg(cfg))


register_plugin(BioPlugin())
