from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.detection.domains.legal.legal_gate import should_enable_legal
from document_resolution.nlp.detection.domains.legal.patterns import legal_pattern
from document_resolution.nlp.plugins.interface import DomainPlugin
from document_resolution.nlp.plugins.registry import register_plugin


class LegalPlugin(DomainPlugin):
    """Domain plugin providing legal/regulatory domain sniffing.

    """

    name = "legal"
    _SNIFF_CAP = 80_000

    def sniff(self, text: str) -> bool:
        """Heuristically detect whether a document is likely legal document


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


        returning ``False`` preserves current detector behaviour while still
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
