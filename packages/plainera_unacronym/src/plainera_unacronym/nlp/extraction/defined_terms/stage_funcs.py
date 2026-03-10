from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector
from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.engine.stages import StageResult


def st_detect_terms(s: TermFlowState) -> StageResult[TermFlowState]:
    """Run detection to find defined terms occurrences and unique occurrences.

    Populates `s.det_res` with the detector output and records a short summary
    into `s.last_info`.

    Args:
        s (FlowState): Mutable flow state containing input text and detector config.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.
    """
    det = DefinedTermDetector(config=s.det_cfg).detect(s.text)
    s.det_res = det
    s.last_info = f"unique terms:{len(det.unique_terms)} occurrences:{len(det.occurrences)}"
    return StageResult(s, s.last_info)


def st_build_term_sense_index(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)


def st_tier1_score_term_occurrences(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)


def st_tier2_term_semantic_rerank(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)


def st_assemble_term_resolutions(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)
