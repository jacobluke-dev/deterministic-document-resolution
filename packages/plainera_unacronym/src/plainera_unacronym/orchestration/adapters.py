from __future__ import annotations

from collections.abc import Mapping

from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.nlp.extraction.defined_terms.execute import (
    detect_and_resolve_terms,
)
from plainera_unacronym.nlp.extraction.structural.execute import (
    detect_and_resolve_structural_references,
)
from plainera_unacronym.orchestration import (PipelineRunner,
                                              PIPELINE_ACRONYMS,
                                              PipelineRequest,
                                              PipelineRunResult,
                                              PIPELINE_DEFINED_TERMS,
                                              PIPELINE_STRUCTURAL_REFERENCES)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


class AcronymPipelineRunner(PipelineRunner):
    key = PIPELINE_ACRONYMS

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        options = _mapping(request.options)

        raw_window_left = options.get("window_left", 320)
        raw_window_right = options.get("window_right", 280)

        window_left = raw_window_left if isinstance(raw_window_left, int) else 320
        window_right = raw_window_right if isinstance(raw_window_right, int) else 280

        payload = detect_and_extract(
            request.text,
            det_cfg=options.get("det_cfg"),
            ext_cfg=options.get("ext_cfg"),
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
    key = PIPELINE_DEFINED_TERMS

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        options = _mapping(request.options)

        payload = detect_and_resolve_terms(
            request.text,
            det_cfg=options.get("det_cfg"),
            ext_cfg=options.get("ext_cfg"),
            return_reports=bool(options.get("return_reports", False)),
            disambig_margin_threshold=options.get("disambig_margin_threshold"),
            trace=bool(options.get("trace", False)),
            return_state=bool(options.get("return_state", False)),
            trace_filter=options.get("trace_filter"),
        )

        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )


class StructuralReferencesPipelineRunner(PipelineRunner):
    key = PIPELINE_STRUCTURAL_REFERENCES

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        options = _mapping(request.options)

        payload = detect_and_resolve_structural_references(
            request.text,
            det_cfg=options.get("det_cfg"),
            ext_cfg=options.get("ext_cfg"),
            return_reports=bool(options.get("return_reports", False)),
            return_state=bool(options.get("return_state", False)),
        )

        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )
