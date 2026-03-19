from __future__ import annotations

from observability.logger.decorator import logger

from plainera_unacronym.nlp.detection.base import BaseDetector
from plainera_unacronym.wiring.composition import sink

from .builders import build_structural_reference, canonicalize_structural_kind
from .structural_reference_compiler import compile_structural_reference_patterns
from .types import StructuralReference, StructuralReferenceDetectorResult


class StructuralReferenceDetector(BaseDetector[StructuralReferenceDetectorResult]):
    """Detect structural document references such as Section 4.2 and Schedule A.

    The detector scans text for a conservative, bounded set of known structural
    reference forms used in legal and similarly structured documents. Each match
    is emitted directly as a canonical ``StructuralReference`` with preserved
    source offsets and deterministic normalisation.

    Detection is intentionally narrow:
        * only known structural keywords are recognised
        * a valid label is required after the keyword
        * ordinary capitalised phrases are ignored

    This detector returns:
        * ``references``: structural references detected in document order
    """

    def __init__(self, config, max_workers=None):
        """Initialise the structural-reference detector.

        Args:
            config: Detector configuration object stored on the shared base
                detector. The current implementation does not require any
                structural-reference-specific config fields.
            max_workers: Optional maximum number of worker processes for future
                parallel execution support.
        """
        super().__init__(config=config, max_workers=max_workers)
        self._patterns = compile_structural_reference_patterns()

    @staticmethod
    def _is_invalid_appendix_alpha_continuation(text: str, label: str, end_offset: int) -> bool:
        """Return whether an appendix alpha label is immediately continued by '.X'.

        Rejects cases such as ``Appendix C.A`` where a shorter valid alpha label would
        otherwise be emitted as a prefix match.
        """
        if not label.isalpha():
            return False

        if end_offset + 1 >= len(text):
            return False

        if text[end_offset] != ".":
            return False

        next_char = text[end_offset + 1]
        return next_char.isalpha() and next_char.isupper()

    def _iter_structural_references(self, text: str) -> list[StructuralReference]:
        """Collect structural references from supported keyword+label patterns.

        The scan runs across all compiled structural-reference patterns,
        deduplicates exact repeated matches, builds canonical output objects, and
        returns results sorted by source order.

        Args:
            text: Full source text to scan.

        Returns:
            A list of ``StructuralReference`` objects detected in the text, sorted
            by ``start_offset``, ``end_offset``, and ``normalized_key``.
        """
        refs: list[StructuralReference] = []
        seen: set[tuple[int, int, str]] = set()

        patterns = (
            self._patterns.schedule_reference,
            self._patterns.exhibit_reference,
            self._patterns.annex_reference,
            self._patterns.appendix_reference,
            self._patterns.section_reference,
            self._patterns.clause_reference,
            self._patterns.article_reference,
        )

        for pat in patterns:
            for match in pat.finditer(text):
                kind = match.group("kind")
                label = match.group("label")
                start_offset, end_offset = match.span()

                canonical_kind = canonicalize_structural_kind(kind)

                if canonical_kind == "Appendix" and self._is_invalid_appendix_alpha_continuation(
                    text, label, end_offset
                ):
                    continue

                dedupe_key = (start_offset, end_offset, canonical_kind)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                refs.append(
                    build_structural_reference(
                        kind=kind,
                        label=label,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        provenance="structural_reference_detector",
                    )
                )

        refs.sort(key=lambda ref: (ref.start_offset, ref.end_offset, ref.normalized_key))
        return refs

    @logger(message="structural_reference_detector.detect", db_sink=sink)
    def detect(self, text: str) -> StructuralReferenceDetectorResult:
        """Detect structural references in a text run.

        Args:
            text: Full source text to analyse.

        Returns:
            A ``StructuralReferenceDetectorResult`` containing all detected
            structural references.
        """
        return StructuralReferenceDetectorResult(references=self._iter_structural_references(text))

    def detect_parallel(
        self, text: str, threshold: int = 1000, chunk_size: int = 256
    ) -> StructuralReferenceDetectorResult:
        """Detect structural references using the current single-pass implementation.

        This method currently delegates directly to ``detect``. The parallel entry
        point exists to preserve a stable detector interface and allow future
        structure-aware chunking if needed.

        Args:
            text: Full source text to analyse.
            threshold: Minimum text-size threshold at which a parallel strategy may
                be considered in future.
            chunk_size: Target chunk size that may be used by a future parallel
                implementation.

        Returns:
            A ``StructuralReferenceDetectorResult`` containing all detected
            structural references.
        """
        return self.detect(text)
