from plainera_unacronym.nlp.extraction.engine.detect_flow import ExtractionFlow


def detect_and_extract(
    text: str,
    *,
    det_cfg=None,
    ext_cfg=None,
    window_left: int = 320,
    window_right: int = 280,
    return_reports: bool = False,
):
    flow = ExtractionFlow(
        det_cfg=det_cfg,
        ext_cfg=ext_cfg,
        window_left=window_left,
        window_right=window_right,
    )
    det_res, extr, reports = flow.run(text)
    return (det_res, extr, reports) if return_reports else (det_res, extr)
