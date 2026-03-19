from __future__ import annotations

from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector
from plainera_unacronym.nlp.extraction.defined_terms.definitions import extract_term_definitions
from plainera_unacronym.nlp.extraction.defined_terms.senses import build_term_sense_index
from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.defined_terms.structure import build_term_structure_index
from plainera_unacronym.nlp.extraction.defined_terms.tiers.assemble import assemble_term_resolution_result
from plainera_unacronym.nlp.extraction.defined_terms.tiers.tier_1_score import score_term_occurrences_tier1
from plainera_unacronym.nlp.extraction.defined_terms.tiers.tier_2 import rerank_term_occurrences_tier2
from plainera_unacronym.nlp.extraction.base.stages import StageResult


def st_detect_terms(s: TermFlowState) -> StageResult[TermFlowState]:
    """Run defined-term detection and store the detector result."""
    det = DefinedTermDetector(config=s.det_cfg).detect(s.text)
    s.det_res = det
    s.last_info = (
        f"introductions={len(det.introductions)} "
        f"unique_terms={len(det.unique_terms)} "
        f"occurrences={len(det.mentions)}"
    )
    return StageResult(s, s.last_info)


def st_build_structure_index(s: TermFlowState) -> StageResult[TermFlowState]:
    """Build lightweight structural context for the document."""
    s.structure_index = build_term_structure_index(s.text)
    s.last_info = f"structures={len(s.structure_index.paths_by_span)}"
    return StageResult(s, s.last_info)


def st_extract_term_definitions(s: TermFlowState) -> StageResult[TermFlowState]:
    """Extract definition spans/text for detected term introductions."""
    assert s.det_res is not None

    s.definition_entries = extract_term_definitions(
        text=s.text,
        detector_result=s.det_res,
        structure_index=s.structure_index,
    )

    n_with_text = sum(1 for e in s.definition_entries if e.definition_text)
    s.last_info = f"definitions={len(s.definition_entries)} with_text={n_with_text}"
    return StageResult(s, s.last_info)


def st_build_term_sense_index(s: TermFlowState) -> StageResult[TermFlowState]:
    """Build term senses from extracted definition entries."""
    s.tier_1.term_sense_index, s.tier_1.sense_index = build_term_sense_index(
        definition_entries=s.definition_entries,
    )

    if s.det_res is not None:
        s.tier_1.occurrences = list(s.det_res.mentions)

    s.last_info = (
        f"keys={len(s.tier_1.term_sense_index)} "
        f"senses={len(s.tier_1.sense_index)} "
        f"occurrences={len(s.tier_1.occurrences)}"
    )
    return StageResult(s, s.last_info)


def st_tier1_score_term_occurrences(s: TermFlowState) -> StageResult[TermFlowState]:
    """Run deterministic Tier-1 scoring over term occurrences."""
    if s.det_res is not None and not s.tier_1.occurrences:
        s.tier_1.occurrences = list(s.det_res.mentions)

    s.tier_1.ranked = score_term_occurrences_tier1(
        text=s.text,
        occurrences=s.tier_1.occurrences,
        term_sense_index=s.tier_1.term_sense_index,
        structure_index=s.structure_index,
        cfg=s.ext_cfg,
    )

    decided = sum(1 for r in s.tier_1.ranked if r.chosen_sense_id is not None)
    s.last_info = f"occurrences={len(s.tier_1.occurrences)} " f"ranked={len(s.tier_1.ranked)} " f"decided={decided}"
    return StageResult(s, s.last_info)


def st_tier2_term_semantic_rerank(s: TermFlowState) -> StageResult[TermFlowState]:
    """Optionally rerank ambiguous term occurrences using Tier-2 semantics."""
    ranked, report = rerank_term_occurrences_tier2(
        text=s.text,
        t1_ranked=s.tier_1.ranked,
        sense_index=s.tier_1.sense_index,
        cfg=s.ext_cfg,
    )
    s.tier_2.ranked = ranked
    s.tier_2.report = report

    s.last_info = f"tier2_ranked={len(ranked)} applied={report.applied}"
    return StageResult(s, s.last_info)


def st_assemble_term_resolutions(s: TermFlowState) -> StageResult[TermFlowState]:
    """Assemble final term resolutions from Tier-1 and Tier-2 outputs."""
    s.extr = assemble_term_resolution_result(s)
    s.last_info = (
        f"term_resolutions={len(s.extr.term_resolutions)} "
        f"undecided={len(s.extr.undecided)} "
        f"ambiguous_keys={len(s.extr.ambiguous_keys)}"
    )
    return StageResult(s, s.last_info)
