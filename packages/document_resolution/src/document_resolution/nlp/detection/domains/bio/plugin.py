import re

from document_resolution.nlp.common.types import AcronymDetectorConfig
from document_resolution.nlp.detection.domains.bio.bio_gate import should_enable_bio
from document_resolution.nlp.detection.domains.bio.config import BioConfig
from document_resolution.nlp.detection.domains.bio.patterns import bio_pattern
from document_resolution.nlp.detection.domains.bio.rules import bio_keep_guard
from document_resolution.nlp.plugins.interface import DomainPlugin
from document_resolution.nlp.plugins.registry import register_plugin

_BIO_SNIFF_RE = re.compile(r"\b(?:mRNA|miRNA|sgRNA|SARS-CoV-2|MERS-CoV|H\d{1,2}N\d{1,2}|IL-\d{1,3}|[35][′'\"]-?UTR)\b")

# TODO: If/when domain-specific *extraction* hooks are added (e.g. inline cue tweaks),
# decide whether to extend DomainPlugin or introduce a separate ExtractionPlugin + registry.


class BioPlugin(DomainPlugin):
    """Domain plugin providing biology-specific detection hooks.

    Adds a fast domain sniff for auto-enabling, plus a dedicated candidate regex and
    a keep-guard to rescue borderline spans in bio-heavy text. Intended to be
    side-effect free and cheap per-document.
    """

    name = "bio"

    def _cfg(self, cfg: AcronymDetectorConfig) -> BioConfig:
        """Return the active `BioConfig` for this plugin.

        Looks up `cfg.domain_cfg["bio"]` and falls back to a default `BioConfig()`.
        This isolates callers from the storage details of per-domain configuration.

        Args:
            cfg (AcronymDetectorConfig): Active detector configuration.

        Returns:
            BioConfig: Per-document biology configuration for this plugin.
        """
        return cfg.domain_cfg.get(self.name) or BioConfig()

    @staticmethod
    def sniff(text: str) -> bool:
        """Heuristically detect whether a document is likely biology/biomed.

        Scans a capped prefix for strong bio cues (RNA types, viruses, cytokines,
        UTR patterns) and falls back to a weighted bio-signal scorer.

        Args:
            text (str): Source document text (caller may pass a truncated prefix).

        Returns:
            bool: True if biology signals are present; otherwise False.
        """
        t = text[:80_000]  # cap scanning for speed
        if _BIO_SNIFF_RE.search(t):
            return True
        ok, _ = should_enable_bio(t)
        return ok

    def extra_candidates(self, text: str, cfg: AcronymDetectorConfig):
        """Yield additional biology-specific candidate spans.

        When the bio domain is enabled, runs the domain regex and yields each match
        as a `(surface, start, end)` tuple using end-exclusive offsets.

        Args:
            text (str): Source document text.
            cfg (AcronymDetectorConfig): Active detector configuration.

        Yields:
            TextSpanTuple: (surface, start, end) for each domain match.
        """
        if self.name not in cfg.enabled_domains:
            return
        pat = bio_pattern()
        for m in pat.finditer(text):
            s, e = m.span("bio")
            yield text[s:e], s, e

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg: AcronymDetectorConfig) -> bool:
        """Decide whether to keep a generic candidate based on biology context.

        Used to rescue tokens that the generic pipeline might drop (e.g., short or
        ambiguous forms) when bio context suggests they are meaningful.

        Args:
            surface (str): Candidate surface text (`text[s:e]`).
            text (str): Full source document text.
            s (int): Start offset (inclusive) of the candidate.
            e (int): End offset (exclusive) of the candidate.
            cfg (AcronymDetectorConfig): Active detector configuration (may contain BioConfig).

        Returns:
            bool: True to keep the candidate; False to let generic logic decide.
        """
        if self.name not in cfg.enabled_domains:
            return False
        return bio_keep_guard(surface, text, s, e, self._cfg(cfg))


register_plugin(BioPlugin())
