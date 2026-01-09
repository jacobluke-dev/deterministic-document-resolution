from plainera_unacronym.nlp.extraction.engine.detect_flow import ExtractionFlow


def detect_and_extract(
    text: str,
    *,
    det_cfg=None,
    ext_cfg=None,
    window_left: int = 320,
    window_right: int = 280,
    return_reports: bool = False,
    trace:bool=False,
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
    det_res, extr, reports = flow.run(text)
    if return_reports and trace:
        return det_res, extr, reports, flow.trace_events
    if return_reports:
        return det_res, extr, reports
    return det_res, extr
