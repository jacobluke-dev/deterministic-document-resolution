from __future__ import annotations

from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector
from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.engine.stages import StageResult


def st_detect_terms(s: TermFlowState) -> StageResult[TermFlowState]:
    """Run defined-term detection and store the detector result."""
    det = DefinedTermDetector(config=s.det_cfg).detect(s.text)
    s.det_res = det
    s.last_info = f"terms={len(det.unique_terms)} occurrences={len(det.occurrences)}"
    return StageResult(s, s.last_info)


def st_build_structure_index(s: TermFlowState) -> StageResult[TermFlowState]:
    """Build the term sense inventory from detected introductions."""
    # TODO: replace with real builder
    # s.tier_1.term_sense_index = build_term_sense_index(...)
    # s.tier_1.sense_index = ...
    s.last_info = (
        f"keys={len(s.tier_1.term_sense_index)} "
        f"senses={len(s.tier_1.sense_index)}"
    )
    return StageResult(s, s.last_info)

def st_extract_term_definitions(s: TermFlowState) -> StageResult[TermFlowState]:
    """Extract definition spans/text for detected term introductions."""
    assert s.det_res is not None

    # TODO: replace with real extractor
    # s.definition_entries = extract_term_definitions(
    #     text=s.text,
    #     detector_result=s.det_res,
    #     structure_index=s.structure_index,
    # )

    s.last_info = f"definitions={len(s.definition_entries)}"
    return StageResult(s, s.last_info)

def st_build_term_sense_index(s: TermFlowState) -> StageResult[TermFlowState]:
    """Build term senses from extracted definition entries."""
    # TODO: replace with real builder
    # s.tier_1.term_sense_index, s.tier_1.sense_index = build_term_sense_index(
    #     definition_entries=s.definition_entries,
    # )

    if s.det_res is not None:
        s.tier_1.occurrences = list(s.det_res.occurrences)

    s.last_info = (
        f"keys={len(s.tier_1.term_sense_index)} "
        f"senses={len(s.tier_1.sense_index)} "
        f"occurrences={len(s.tier_1.occurrences)}"
    )
    return StageResult(s, s.last_info)


def st_tier1_score_term_occurrences(s: TermFlowState) -> StageResult[TermFlowState]:
    """Run deterministic Tier-1 scoring over term occurrences."""
    # TODO: replace with real scorer
    # s.tier_1.occurrences = list(s.det_res.occurrences) if s.det_res else []
    # s.tier_1.ranked = score_term_candidates_tier1(...)
    s.last_info = (
        f"occurrences={len(s.tier_1.occurrences)} "
        f"ranked={len(s.tier_1.ranked)}"
    )
    return StageResult(s, s.last_info)


def st_tier2_term_semantic_rerank(s: TermFlowState) -> StageResult[TermFlowState]:
    """Optionally rerank ambiguous term occurrences using Tier-2 semantics."""
    # TODO: replace with real rerank step
    applied = sum(1 for r in s.tier_2.ranked if r.applied)
    s.last_info = f"tier2_ranked={len(s.tier_2.ranked)} applied={applied}"
    return StageResult(s, s.last_info)


def st_assemble_term_resolutions(s: TermFlowState) -> StageResult[TermFlowState]:
    """Assemble the final term resolution result."""
    # TODO: replace with real assembler
    # s.extr = TermResolutionResult(...)
    n_res = len(s.extr.term_resolutions) if s.extr else 0
    s.last_info = f"term_resolutions={n_res}"
    return StageResult(s, s.last_info)
