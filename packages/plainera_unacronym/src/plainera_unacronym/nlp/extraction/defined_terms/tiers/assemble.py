from __future__ import annotations

from typing import Literal

from plainera_unacronym.nlp.extraction.defined_terms.state import TermFlowState
from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermCandidateScore,
    TermResolution,
    TermResolutionResult,
)


def _select_final_candidate_scores_and_method(
    s: TermFlowState,
    idx: int,
) -> tuple[tuple[TermCandidateScore, ...], Literal["tier1", "tier2_blend"]]:
    r1 = s.tier_1.ranked[idx]
    r2 = s.tier_2.ranked[idx] if idx < len(s.tier_2.ranked) else None

    use_tier2 = bool(r2 is not None and r2.applied and r2.blended_scores is not None)
    method: Literal["tier1", "tier2_blend"] = "tier2_blend" if use_tier2 else "tier1"

    candidate_scores: list[TermCandidateScore] = []
    for sense_id, tier1_score in r1.candidate_scores.items():
        sense = s.tier_1.sense_index.get(sense_id)
        tier2_score = r2.tier2_sims.get(sense_id) if use_tier2 and r2 and r2.tier2_sims else None
        total_score = (
            float(r2.blended_scores[sense_id]) if use_tier2 and r2 and r2.blended_scores else float(tier1_score)
        )

        candidate_scores.append(
            TermCandidateScore(
                sense_id=sense_id,
                total_score=total_score,
                tier1_score=float(tier1_score),
                tier2_score=float(tier2_score) if tier2_score is not None else None,
                definition_span=sense.definition_span if sense is not None else None,
                components={},
            )
        )

    candidate_scores.sort(key=lambda c: (-c.total_score, c.sense_id))
    return tuple(candidate_scores), method


def _choose_from_candidate_scores(
    candidate_scores: tuple[TermCandidateScore, ...],
    *,
    margin_threshold: float,
) -> str | None:
    if not candidate_scores:
        return None
    if len(candidate_scores) == 1:
        return candidate_scores[0].sense_id

    top = candidate_scores[0]
    second = candidate_scores[1]
    gap = top.total_score - second.total_score
    margin = gap / max(abs(top.total_score), 1.0)
    return top.sense_id if margin >= margin_threshold else None


def assemble_term_resolution_result(s: TermFlowState) -> TermResolutionResult:
    margin_threshold = float(s.ext_cfg.multi_tier.select_margin_threshold)
    resolutions: list[TermResolution] = []
    undecided: list[TermResolution] = []

    for idx, r1 in enumerate(s.tier_1.ranked):
        candidate_scores, method = _select_final_candidate_scores_and_method(s, idx)
        chosen_sense_id = _choose_from_candidate_scores(candidate_scores, margin_threshold=margin_threshold)
        chosen = next((c for c in candidate_scores if c.sense_id == chosen_sense_id), None)

        resolution = TermResolution(
            occurrence_span=(r1.occ.term, r1.occ.start_offset, r1.occ.end_offset),
            term=r1.occ.term,
            normalized_key=r1.occ.normalized_key,
            chosen_sense_id=chosen_sense_id,
            chosen_definition_span=chosen.definition_span if chosen else None,
            candidate_scores=candidate_scores,
            resolution_method="unresolved" if chosen_sense_id is None else method,
        )
        resolutions.append(resolution)
        if chosen_sense_id is None:
            undecided.append(resolution)

    ambiguous_keys = tuple(sorted(key for key, senses in s.tier_1.term_sense_index.items() if len(senses) > 1))

    return TermResolutionResult(
        term_sense_index=s.tier_1.term_sense_index,
        sense_index=s.tier_1.sense_index,
        term_resolutions=resolutions,
        ambiguous_keys=ambiguous_keys,
        undecided=undecided,
        tier2_report=s.tier_2.report,
        tier2_ranked=tuple(s.tier_2.ranked),
    )
