from plainera_unacronym.nlp.extraction.engine.detect_flow import ExtractionFlow
from plainera_unacronym.nlp.extraction.engine.state import FlowState


def detect_and_extract(
    text: str,
    *,
    det_cfg=None,
    ext_cfg=None,
    window_left: int = 320,
    window_right: int = 280,
    return_reports: bool = False,
    trace: bool = False,
    return_state: bool = False,
    trace_filter=None,
):
    flow = ExtractionFlow(
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
        window_left=window_left,
        window_right=window_right,
        trace=trace,
        trace_filter=trace_filter,
    )

    state = FlowState(text=text, det_cfg=flow.det_cfg, ext_cfg=flow.ext_cfg)

    # IMPORTANT: run the chain on *this* state
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
