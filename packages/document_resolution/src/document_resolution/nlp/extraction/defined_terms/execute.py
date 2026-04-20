from __future__ import annotations

from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.extraction import run_flow_with_options
from document_resolution.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from document_resolution.nlp.extraction.defined_terms.extract_flow import DefinedTermResolutionFlow
from document_resolution.nlp.extraction.defined_terms.state import TermFlowState


def detect_and_resolve_terms(
    text: str,
    *,
    det_cfg: DefinedTermDetectorConfig | None = None,
    ext_cfg: DefinedTermExtractionConfig | None = None,
    return_reports: bool = False,
    trace: bool = False,
    return_state: bool = False,
    trace_filter=None,
):
    """Run the full defined-term detection + resolution pipeline over a single input text.

    Args:
        text: Source document text to process.
        det_cfg: Optional `DefinedTermDetectorConfig` override. If None, the flow
            default is used.
        ext_cfg: Optional `DefinedTermExtractionConfig` override. If None, the
            flow default is used.
        return_reports: If True, include per-stage `StageReport` objects in the
            return tuple.
        trace: If True, enable tracing for the underlying flow and return trace
            events when requested.
        return_state: If True, include the final `TermFlowState` in the return
            tuple so callers/tests can inspect intermediate artefacts.
        trace_filter: Optional regex string used to filter trace output.

    Returns:
        The return shape depends on `return_reports`, `trace`, and `return_state`:

        - Default:
            (det_res, extr)

        - If return_reports:
            (det_res, extr, reports)

        - If return_reports and trace:
            (det_res, extr, reports, trace_events)

        - If return_state:
            (det_res, extr, state)

        - If return_state and return_reports:
            (det_res, extr, reports, state)

        - If return_state and return_reports and trace:
            (det_res, extr, reports, state, trace_events)

    Raises:
        AssertionError: If the pipeline completes without producing a detector
            result or resolution result.
    """
    flow = DefinedTermResolutionFlow(
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
        trace=trace,
        trace_filter=trace_filter,
    )

    state = TermFlowState(
        text=text,
        det_cfg=flow.det_cfg,
        ext_cfg=flow.ext_cfg,
    )

    return run_flow_with_options(
        flow=flow,
        state=state,
        return_reports=return_reports,
        return_state=return_state,
        trace=trace,
    )
