from __future__ import annotations


def run_flow_with_options(
    *,
    flow,
    state,
    return_reports: bool = False,
    return_state: bool = False,
    trace: bool = False,
):
    """Run a configured flow/state pair and return outputs with optional extras."""
    tracer = getattr(flow, "_tracer", None)
    state, reports = flow.build_chain().run(state, tracer=tracer)

    assert state.det_res is not None and state.extr is not None

    trace_events = tracer.events if tracer else None
    if hasattr(flow, "trace_events"):
        flow.trace_events = trace_events

    if return_state and return_reports and trace:
        return state.det_res, state.extr, reports, state, trace_events

    if return_state and return_reports:
        return state.det_res, state.extr, reports, state

    if return_state:
        return state.det_res, state.extr, state

    if return_reports and trace:
        return state.det_res, state.extr, reports, trace_events

    if return_reports:
        return state.det_res, state.extr, reports

    return state.det_res, state.extr
