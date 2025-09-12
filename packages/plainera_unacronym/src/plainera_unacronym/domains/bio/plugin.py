import re

from plainera_unacronym.domains.bio.config import BioConfig, _STATS_CI_RE, _STATS_OR_HR_RR_RE
from plainera_unacronym.domains.bio.patterns import bio_pattern
from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.plugins.registry import register_plugin


_BIO_SNIFF_RE = re.compile(
    r"(?:\bmRNA\b|\bmiRNA\b|\bsgRNA\b|SARS-CoV-2|MERS-CoV|\bH\dN\d\b|\bIL-\d{1,3}\b|IFN-|TNF-|TGF-|[35][′'\"]-?UTR)"
)

class BioPlugin:
    name = "bio"

    def _cfg(self, cfg: DetectorConfig) -> BioConfig:
        return cfg.domain_cfg.get(self.name, BioConfig())

    @staticmethod
    def sniff(text: str) -> bool:
        return bool(_BIO_SNIFF_RE.search(text))

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
        bcfg = self._cfg(cfg)
        if surface in bcfg.rna_like:
            return True
        if len(surface) == 2 and surface in bcfg.two_letter_keep:
            r = text[max(0, s-20):min(len(text), e+20)]
            return bool(_STATS_CI_RE.search(r) or _STATS_OR_HR_RR_RE.search(r))
        return False


register_plugin(BioPlugin())
