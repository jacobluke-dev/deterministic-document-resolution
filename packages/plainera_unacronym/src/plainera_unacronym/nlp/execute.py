from plainera_unacronym.nlp.extraction.engine.detect_flow import ExtractionFlow
from plainera_unacronym.nlp.extraction.engine.state import FlowState


def detect_and_extract(
    text: str,
    *,
    det_cfg=None,
    ext_cfg=None,
    tier2_model=None,
    window_left: int = 320,
    window_right: int = 280,
    return_reports: bool = False,
    trace: bool = False,
    return_state: bool = False,
    trace_filter=None,
):
    """Run the full detection + extraction pipeline over a single input text.

    This is a convenience wrapper around `ExtractionFlow` that:
      1) constructs an `ExtractionFlow` (optionally with custom configs),
      2) initialises a `FlowState`,
      3) executes the configured stage chain, and
      4) returns the final `DetectorResult` and `ExtractionResult`, with optional
         stage reports, trace events, and/or the final `FlowState`.

    Args:
        text: Source document text to process.
        det_cfg: Optional `DetectorConfig` override. If None, the flow default
            (`DetectorConfig()`) is used.
        ext_cfg: Optional `ExtractionConfig` override. If None, the flow default
            (`ExtractionConfig()`) is used.
        tier2_model: Optional `Tier2Model` override. If None, the flow default
        window_left: Number of characters to include to the left of each first
            occurrence when building the anchored extraction window.
        window_right: Number of characters to include to the right of each first
            occurrence when building the anchored extraction window.
        return_reports: If True, include per-stage `StageReport` objects in the
            return tuple.
        trace: If True, enable tracing for the underlying flow and return trace
            events when requested. Tracing may add overhead.
        return_state: If True, include the final `FlowState` in the return tuple.
            This is useful for inspecting intermediate work products such as
            Tier-1/Tier-2 disambiguation structures.
        trace_filter: Optional regex string used to filter which acronym keys
            are traced (e.g. r"^(GPU|API)$"). Only applied when `trace=True`.

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

        Where:
          - det_res is a `DetectorResult`
          - extr is an `ExtractionResult`
          - reports is a list of `StageReport`
          - state is the final `FlowState`
          - trace_events is a list of `TraceEvent`

    Raises:
        AssertionError: If the pipeline completes without producing a detector
            result or extraction result (indicates an internal pipeline fault).

    Notes:
        This function executes the stage chain directly via `flow.build_chain().run(...)`
        so that callers/tests can optionally capture stage reports and/or access
        the final `FlowState` without relying on `ExtractionFlow.run(...)`.
    """
    flow = ExtractionFlow(
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
        window_left=window_left,
        window_right=window_right,
        trace=trace,
        trace_filter=trace_filter,
    )

    state = FlowState(text=text, det_cfg=flow.det_cfg, ext_cfg=flow.ext_cfg, tier2_model=tier2_model)

    state, reports = flow.build_chain().run(state, tracer=flow._tracer)

    assert state.det_res is not None and state.extr is not None

    if return_state and return_reports and trace:
        return state.det_res, state.extr, reports, state, flow.trace_events

    if return_state and return_reports:
        return state.det_res, state.extr, reports, state

    if return_state:
        return state.det_res, state.extr, state

    if return_reports and trace:
        return state.det_res, state.extr, reports, flow.trace_events

    if return_reports:
        return state.det_res, state.extr, reports

    return state.det_res, state.extr
