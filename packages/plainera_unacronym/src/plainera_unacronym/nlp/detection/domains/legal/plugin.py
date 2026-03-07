import re

from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.detection.domains.legal.config import LegalConfig
from plainera_unacronym.nlp.detection.domains.legal.patterns import legal_pattern
from plainera_unacronym.nlp.plugins.interface import DomainPlugin
from plainera_unacronym.nlp.plugins.registry import register_plugin

# Strong-ish cues that are relatively characteristic in contracts/regulatory docs.
# Keep this conservative to reduce false positives.
_LEGAL_SNIFF_RE = re.compile(
    r"(?:"
    r"\"[A-Z][^\"]{1,80}\"\s+(?:shall\s+)?mean(?:s)?\b"   # "Term" means / shall mean
    r"|hereinafter\b"
    r"|pursuant\s+to\b"
    r"|this\s+Agreement\b"
    r"|\b(?:Schedule|Appendix)\s+[A-Z0-9]+\b"
    r"|\b(?:clause|section)\s+\d+(?:\.\d+)*\b"
    r"|\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Act\s+\d{4}\b"
    r"|\bRegulation\s*\(\s*EU\s*\)\s*\d{3,4}/\d{2,4}\b"
    r")",
    re.IGNORECASE,
)


class LegalPlugin(DomainPlugin):
    """Domain plugin providing legal/regulatory domain sniffing.

    UN-86 scope: only auto-enable via sniff + registration.
    Extra candidate / keep-guard hooks can be added later when term detection lands.
    """

    name = "legal"

    def _cfg(self, cfg: DetectorConfig) -> LegalConfig:
        """Return the active `Legal` for this plugin.

        Looks up `cfg.domain_cfg["bio"]` and falls back to a default `LegalConfig()`.
        This isolates callers from the storage details of per-domain configuration.

        Args:
            cfg (DetectorConfig): Active detector configuration.

        Returns:
            LegalConfig: Per-document legal configuration for this plugin.
        """
        return cfg.domain_cfg.get(self.name) or LegalConfig()

    @staticmethod
    def sniff(text: str) -> bool:
        """Heuristically detect whether a document is likely legal document

        Scans a capped prefix for strong bio cues checkiing using the _LEGAL_SNIFF_RE.

        Args:
            text (str): Source document text (caller may pass a truncated prefix).

        Returns:
            bool: True if legal signals are present; otherwise False.
        """
        t = text[:80_000]
        return bool(_LEGAL_SNIFF_RE.search(t))

    def extra_candidates(self, text: str, cfg: DetectorConfig):
        """Yield additional biology-specific candidate spans.

        When the legaL domain is enabled, runs the domain regex and yields each match
        as a `(surface, start, end)` tuple using end-exclusive offsets.

        Args:
            text (str): Source document text.
            cfg (DetectorConfig): Active detector configuration.

        Yields:
            TextSpanTuple: (surface, start, end) for each domain match.
        """
        if self.name not in cfg.enabled_domains:
            return
        pat = legal_pattern()
        for m in pat.finditer(text):
            s, e = m.span("legal")
            yield text[s:e], s, e

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
        """Domain-specific “rescue” hook for borderline candidates.

        This method is consulted by the generic detection pipeline when deciding whether
        to *keep* a candidate that might otherwise be dropped by general heuristics
        (e.g., short tokens, ambiguous forms). Returning ``True`` signals that the
        domain plugin has strong evidence the candidate is meaningful in this domain
        and should be retained.

        For the **legal** domain plugin (UN-86), we currently return ``False`` for all
        inputs on purpose:

        - UN-86’s scope is **domain identification** (sniffing) only. We are not yet
          introducing domain-specific extraction/ranking behaviour for acronyms.
        - Legal documents do not require special “acronym rescue” logic analogous to
          biology/statistics (e.g., OR/HR/RR in Bio). Adding keep-guards prematurely
          increases false positives and can change outputs in ways that are hard to
          justify/audit.
        - Once defined-term extraction and Tier-1/Tier-2 resolution land (UN-68.x),
          legal-specific keep/rescue logic (if needed) should be derived from concrete
          failure cases and backed by deterministic tests.

        In short: returning ``False`` preserves current detector behaviour while still
        allowing the legal domain to be auto-enabled for downstream components.

        Args:
            surface: Candidate surface text (`text[s:e]`).
            text: Full source document text.
            s: Start offset (inclusive).
            e: End offset (exclusive).
            cfg: Active detector configuration.

        Returns:
            bool: Always ``False`` for now (no legal-domain rescue applied).
        """
        return False

register_plugin(LegalPlugin())
