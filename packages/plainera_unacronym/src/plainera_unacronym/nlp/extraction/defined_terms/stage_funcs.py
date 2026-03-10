from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.engine.stages import StageResult


def st_detect_terms(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)



def st_build_term_sense_index(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)


def st_tier1_score_term_occurrences(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)



def st_tier2_term_semantic_rerank(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)


def st_assemble_term_resolutions(s: TermFlowState) -> StageResult[TermFlowState]:
    return StageResult(s, s.last_info)
