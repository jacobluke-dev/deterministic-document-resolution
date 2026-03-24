from __future__ import annotations

from plainera_unacronym.nlp.common.types import AcronymDetectorConfig, DefinedTermDetectorConfig
from plainera_unacronym.nlp.extraction.acronyms.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.execute import (
    detect_and_resolve_terms,
)
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceDetectorConfig,
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.execute import (
    detect_and_resolve_structural_references,
)
from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineRequest,
    PipelineRunner,
    PipelineRunResult,
)


class AcronymPipelineRunner(PipelineRunner):
    """Adapter for the acronym pipeline execute entry point.

    Translates orchestration-layer pipeline options into the keyword arguments
    expected by ``detect_and_extract``.
    """

    key = PIPELINE_ACRONYMS

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        """Run the acronym pipeline for a single orchestration request.

        Args:
            request: Pipeline-specific request containing the source text and
                acronym pipeline options.

        Returns:
            Opaque top-level pipeline result containing the acronym pipeline's
            native payload.
        """
        options = request.options

        raw_det_cfg = options.get("det_cfg")
        det_cfg = raw_det_cfg if isinstance(raw_det_cfg, AcronymDetectorConfig) else None

        raw_ext_cfg = options.get("ext_cfg")
        ext_cfg = raw_ext_cfg if isinstance(raw_ext_cfg, ExtractionConfig) else None

        raw_window_left = options.get("window_left", 320)
        raw_window_right = options.get("window_right", 280)

        window_left = raw_window_left if isinstance(raw_window_left, int) else 320
        window_right = raw_window_right if isinstance(raw_window_right, int) else 280

        payload = detect_and_extract(
            request.text,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
            tier2_model=options.get("tier2_model"),
            window_left=window_left,
            window_right=window_right,
            return_reports=bool(options.get("return_reports", False)),
            trace=bool(options.get("trace", False)),
            return_state=bool(options.get("return_state", False)),
            trace_filter=options.get("trace_filter"),
        )

        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )


class DefinedTermsPipelineRunner(PipelineRunner):
    """Adapter for the defined-terms pipeline execute entry point.

    Translates orchestration-layer pipeline options into the keyword arguments
    expected by ``detect_and_resolve_terms``.
    """

    key = PIPELINE_DEFINED_TERMS

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        """Run the defined-terms pipeline for a single orchestration request.

        Args:
            request: Pipeline-specific request containing the source text and
                defined-term pipeline options.

        Returns:
            Opaque top-level pipeline result containing the defined-terms
            pipeline's native payload.
        """
        options = request.options

        raw_det_cfg = options.get("det_cfg")
        det_cfg = raw_det_cfg if isinstance(raw_det_cfg, DefinedTermDetectorConfig) else None

        raw_ext_cfg = options.get("ext_cfg")
        ext_cfg = raw_ext_cfg if isinstance(raw_ext_cfg, DefinedTermExtractionConfig) else None

        raw_disambig_margin_threshold = options.get("disambig_margin_threshold")
        disambig_margin_threshold = (
            raw_disambig_margin_threshold if isinstance(raw_disambig_margin_threshold, float) else None
        )

        payload = detect_and_resolve_terms(
            request.text,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
            return_reports=bool(options.get("return_reports", False)),
            disambig_margin_threshold=disambig_margin_threshold,
            trace=bool(options.get("trace", False)),
            return_state=bool(options.get("return_state", False)),
            trace_filter=options.get("trace_filter"),
        )

        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )


class StructuralReferencesPipelineRunner(PipelineRunner):
    """Adapter for the structural-reference pipeline execute entry point.

    Translates orchestration-layer pipeline options into the keyword arguments
    expected by ``detect_and_resolve_structural_references``.
    """

    key = PIPELINE_STRUCTURAL_REFERENCES

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        """Run the structural-reference pipeline for a single orchestration request.

        Args:
            request: Pipeline-specific request containing the source text and
                structural-reference pipeline options.

        Returns:
            Opaque top-level pipeline result containing the structural-reference
            pipeline's native payload.
        """
        options = request.options

        raw_det_cfg = options.get("det_cfg")
        det_cfg = raw_det_cfg if isinstance(raw_det_cfg, StructuralReferenceDetectorConfig) else None

        raw_ext_cfg = options.get("ext_cfg")
        ext_cfg = raw_ext_cfg if isinstance(raw_ext_cfg, StructuralReferenceExtractionConfig) else None

        payload = detect_and_resolve_structural_references(
            request.text,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
            return_reports=bool(options.get("return_reports", False)),
            return_state=bool(options.get("return_state", False)),
        )

        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )
