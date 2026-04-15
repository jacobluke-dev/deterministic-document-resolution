from collections.abc import Iterator

from document_resolution.nlp.common.types import TextSpanTuple
from document_resolution.nlp.detection.domains.structural_reference.structural_gate import (
    should_enable_structural_reference,
)
from document_resolution.nlp.plugins.interface import DomainPlugin
from document_resolution.nlp.plugins.registry import register_plugin


class StructuralReferencePlugin(DomainPlugin):
    """Domain plugin providing structural-reference document sniffing.

    This plugin participates in domain auto-detection by checking whether the
    document contains enough structural-reference cues to justify enabling the
    ``"structural_reference"`` domain.

    - ``sniff`` determines whether the domain should be enabled.
    - ``extra_candidates`` yields no spans.
    - ``keep_guard`` does not rescue any borderline candidates.

    This preserves current acronym/defined-term extraction behaviour while
    allowing downstream structural-reference components to be activated only
    for documents that appear formally structured.
    """

    name = "structural_reference"
    _SNIFF_CAP = 80_000

    def sniff(self, text: str) -> bool:
        """Heuristically detect whether a document uses structural references.

        Inspects a capped prefix of the source text and delegates enablement
        logic to ``should_enable_structural_reference``. A positive result
        indicates that the document contains sufficient structural-reference
        cues (for example, sections, clauses, schedules, appendices, annexes)
        to enable the ``"structural_reference"`` domain.

        Args:
            text (str): Source document text. The plugin inspects only the
                first ``self._SNIFF_CAP`` characters.

        Returns:
            bool: ``True`` if the structural-reference domain should be
            enabled for this document; otherwise ``False``.
        """
        t = text[: self._SNIFF_CAP]
        ok, _reasons = should_enable_structural_reference(t, cap=self._SNIFF_CAP)
        return ok

    def extra_candidates(self, text: str, cfg) -> Iterator[TextSpanTuple]:
        """Yield no additional candidates.

        This plugin exists only to support deterministic domain auto-detection,
        so it does not contribute candidate spans to the generic detection pipeline.

        Args:
            text (str): Source document text.
            cfg: Active detector configuration.

        Returns:
            Iterator[TextSpanTuple]: An empty iterator.
        """
        return iter(())

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg) -> bool:
        """Do not rescue borderline candidates.

        Structural-reference sniffing in UN-90 must not alter acronym or
        generic candidate retention behaviour. This hook therefore always
        returns ``False``.

        Args:
            surface (str): Candidate surface text (``text[s:e]``).
            text (str): Full source document text.
            s (int): Start offset, inclusive.
            e (int): End offset, exclusive.
            cfg: Active detector configuration.

        Returns:
            bool: Always ``False``.
        """
        return False


register_plugin(StructuralReferencePlugin())
