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
    """Build final candidate scores for one occurrence and record the source method.

    For the occurrence at ``idx``, this helper combines Tier-1 candidate scores
    with Tier-2 semantic information when Tier-2 was actually applied. When no
    applied Tier-2 result exists, the returned candidate scores remain pure
    Tier-1 scores.

    Args:
        s: Current term-resolution flow state containing Tier-1 and Tier-2
            ranking outputs plus the meaning index.
        idx: Index of the occurrence to assemble candidate scores for.

    Returns:
        A tuple of:
            - candidate scores sorted by descending final score, and
            - the method label indicating whether the final scores came from
              pure Tier-1 scoring or Tier-2 blended scoring.
    """
    r1 = s.tier_1.ranked[idx]
    r2 = s.tier_2.ranked[idx] if idx < len(s.tier_2.ranked) else None

    use_tier2 = bool(r2 is not None and r2.applied and r2.blended_scores is not None)
    method: Literal["tier1", "tier2_blend"] = "tier2_blend" if use_tier2 else "tier1"

    candidate_scores: list[TermCandidateScore] = []
    for meaning_id, tier1_score in r1.candidate_scores.items():
        meaning = s.tier_1.meaning_index.get(meaning_id)
        tier2_score = r2.tier2_sims.get(meaning_id) if use_tier2 and r2 and r2.tier2_sims else None
        total_score = (
            float(r2.blended_scores[meaning_id]) if use_tier2 and r2 and r2.blended_scores else float(tier1_score)
        )

        candidate_scores.append(
            TermCandidateScore(
                meaning_id=meaning_id,
                total_score=total_score,
                tier1_score=float(tier1_score),
                tier2_score=float(tier2_score) if tier2_score is not None else None,
                definition_span=meaning.definition_span if meaning is not None else None,
                components={},
            )
        )

    candidate_scores.sort(key=lambda c: (-c.total_score, c.meaning_id))
    return tuple(candidate_scores), method


def _choose_from_candidate_scores(
    candidate_scores: tuple[TermCandidateScore, ...],
    *,
    margin_threshold: float,
) -> str | None:
    """Choose a winning meaning from assembled candidate scores.

    The top candidate is selected only when its score margin over the runner-up
    meets or exceeds ``margin_threshold``. Single-candidate cases resolve
    immediately. Empty candidate sets remain unresolved.

    Args:
        candidate_scores: Final assembled candidate scores for one occurrence,
            sorted in descending score order.
        margin_threshold: Minimum normalised margin required to resolve a winner.

    Returns:
        The chosen meaning ID when the winning margin is sufficient, otherwise
        ``None``.
    """
    if not candidate_scores:
        return None
    if len(candidate_scores) == 1:
        return candidate_scores[0].meaning_id

    top = candidate_scores[0]
    second = candidate_scores[1]
    gap = top.total_score - second.total_score
    margin = gap / max(abs(top.total_score), 1.0)
    return top.meaning_id if margin >= margin_threshold else None


def assemble_term_resolution_result(s: TermFlowState) -> TermResolutionResult:
    """Assemble final defined-term resolutions from Tier-1 and Tier-2 outputs.

    For each Tier-1-ranked occurrence, this function builds the final candidate
    score list, selects a winning meaning when the configured margin threshold is
    met, and records unresolved cases explicitly. Tier-2 outputs are incorporated
    only when Tier-2 was applied for that occurrence.

    Args:
        s: Current term-resolution flow state containing Tier-1 rankings,
            optional Tier-2 rerank outputs, and the meaning indexes.

    Returns:
        A ``TermResolutionResult`` containing:
            - the term and meaning indexes,
            - final per-occurrence resolutions,
            - ambiguous multi-meaning term keys,
            - unresolved occurrences, and
            - Tier-2 report metadata.
    """
    margin_threshold = float(s.ext_cfg.multi_tier.select_margin_threshold)
    resolutions: list[TermResolution] = []
    undecided: list[TermResolution] = []

    for idx, r1 in enumerate(s.tier_1.ranked):
        candidate_scores, method = _select_final_candidate_scores_and_method(s, idx)
        chosen_meaning_id = _choose_from_candidate_scores(candidate_scores, margin_threshold=margin_threshold)
        chosen = next((c for c in candidate_scores if c.meaning_id == chosen_meaning_id), None)

        resolution = TermResolution(
            occurrence_span=(r1.occ.term, r1.occ.start_offset, r1.occ.end_offset),
            term=r1.occ.term,
            normalized_key=r1.occ.normalized_key,
            chosen_meaning_id=chosen_meaning_id,
            chosen_definition_span=chosen.definition_span if chosen else None,
            candidate_scores=candidate_scores,
            resolution_method="unresolved" if chosen_meaning_id is None else method,
        )
        resolutions.append(resolution)
        if chosen_meaning_id is None:
            undecided.append(resolution)

    ambiguous_keys = tuple(sorted(key for key, meanings in s.tier_1.term_meaning_index.items() if len(meanings) > 1))

    return TermResolutionResult(
        term_meaning_index=s.tier_1.term_meaning_index,
        meaning_index=s.tier_1.meaning_index,
        term_resolutions=resolutions,
        ambiguous_keys=ambiguous_keys,
        undecided=undecided,
        tier2_report=s.tier_2.report,
        tier2_ranked=tuple(s.tier_2.ranked),
    )
