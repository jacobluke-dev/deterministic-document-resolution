from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.detection.domains.legal.legal_gate import should_enable_legal
from document_resolution.nlp.detection.domains.legal.patterns import legal_pattern
from document_resolution.nlp.plugins.interface import DomainPlugin
from document_resolution.nlp.plugins.registry import register_plugin


class LegalPlugin(DomainPlugin):
    """Domain plugin providing legal/regulatory domain sniffing.

    UN-86 scope: only auto-enable via sniff + registration.
    Extra candidate / keep-guard hooks can be added later when term detection lands.
    """

    name = "legal"
    _SNIFF_CAP = 80_000

    def sniff(self, text: str) -> bool:
        """Heuristically detect whether a document is likely legal document

        Scans a capped prefix for strong legal cues checking using the _LEGAL_SNIFF_RE.

        Args:
            text (str): Source document text (caller may pass a truncated prefix).

        Returns:
            bool: True if legal signals are present; otherwise False.
        """
        t = text[: self._SNIFF_CAP]
        ok, _reasons = should_enable_legal(t)
        return ok

    def extra_candidates(self, text: str, cfg: DefinedTermDetectorConfig):
        """Yield additional legal specific candidate spans.

        When the legaL domain is enabled, runs the domain regex and yields each match
        as a `(surface, start, end)` tuple using end-exclusive offsets.

        Args:
            text (str): Source document text.
            cfg (DefinedTermDetectorConfig): Active detector configuration.

        Yields:
            TextSpanTuple: (surface, start, end) for each domain match.
        """
        if self.name not in cfg.enabled_domains:
            return
        pat = legal_pattern()
        for m in pat.finditer(text):
            s, e = m.span("legal")
            yield text[s:e], s, e

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg: DefinedTermDetectorConfig) -> bool:
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
