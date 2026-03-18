from __future__ import annotations

from observability.logger.decorator import logger

from plainera_unacronym.nlp.detection.base import BaseDetector

from .builders import build_structural_reference
from .structural_reference_compiler import compile_structural_reference_patterns
from .types import StructuralReferenceDetectorResult, StructuralReference


class StructuralReferenceDetector(BaseDetector[StructuralReferenceDetectorResult]):
    """Detect structural document references such as Section 4.2 and Schedule A."""

    def __init__(self, config, max_workers=None):
        super().__init__(config=config, max_workers=max_workers)
        self._patterns = compile_structural_reference_patterns()

    def _iter_structural_references(self, text: str) -> list[StructuralReference]:
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

                dedupe_key = (start_offset, end_offset, kind.lower())
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

    @logger(message="structural_reference_detector.detect", db_sink="sink")
    def detect(self, text: str) -> StructuralReferenceDetectorResult:
        return StructuralReferenceDetectorResult(
            references=self._iter_structural_references(text)
        )

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> StructuralReferenceDetectorResult:
        return self.detect(text)
