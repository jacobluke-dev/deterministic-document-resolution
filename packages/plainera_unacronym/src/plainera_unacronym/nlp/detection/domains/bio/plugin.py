import re

from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.detection.domains.bio.config import BioConfig
from plainera_unacronym.nlp.detection.domains.bio.patterns import bio_pattern
from plainera_unacronym.nlp.detection.domains.bio.rules import keep_guard as bio_keep_guard
from plainera_unacronym.nlp.detection.heuristics.gate import should_enable_bio
from plainera_unacronym.nlp.plugins.interface import DomainPlugin
from plainera_unacronym.nlp.plugins.registry import register_plugin

_BIO_SNIFF_RE = re.compile(r"\b(?:mRNA|miRNA|sgRNA|SARS-CoV-2|MERS-CoV|H\d{1,2}N\d{1,2}|IL-\d{1,3}|[35][′'\"]-?UTR)\b")

# TODO
# One more architectural heads-up (so future Jacob doesn’t swear at past Jacob)
#
# Your registry.py currently holds DomainPlugin used by detection (bio sniffing etc.).
# build_plan() is an extraction concern (inline cues, parenthetical allow hooks).
#
# So either:
#
# you’re intentionally reusing one registry for both (fine, but then DomainPlugin should expose extraction hooks too), or
#
# you should split registries: domain_registry.py vs extraction_registry.py.
#
# Right now, you’ve wired extraction planning to a registry that (so far) only supports detection plugins.
# That mismatch is exactly how “it compiles but does nothing” bugs are born. (The classic.)
#
# If you want, paste DomainPlugin (interface) and your intended extraction plugin hook shape, and I’ll tell you whether to split or extend cleanly.

class BioPlugin(DomainPlugin):
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
