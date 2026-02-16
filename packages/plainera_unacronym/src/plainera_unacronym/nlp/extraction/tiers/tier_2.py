from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from plainera_unacronym.nlp.extraction.tiers.semantic import embed_texts, cosine_sim01
from plainera_unacronym.nlp.extraction.tiers.types import Tier1OccurrenceRanking, Tier2SkipReason, \
    Tier2OccurrenceRanking


@dataclass(frozen=True)
class _EligibleRerank:
    """
    Work item describing a Tier-1 occurrence eligible for Tier-2 reranking.

    Attributes:
        idx: Position of the occurrence in `t1.ranked` (and therefore in the final Tier-2 list).
        r1: Tier-1 ranking record for the occurrence.
        context: Context window string used for semantic embedding.
        cand_ids: Candidate sense IDs in Tier-1 insertion order.
        cand_texts: Candidate text strings aligned 1:1 with `cand_ids`.
    """
    idx: int
    r1: Tier1OccurrenceRanking
    context: str
    cand_ids: list[str]
    cand_texts: list[str]


@dataclass(frozen=True)
class _EmbeddingsBatch:
    """
    Bundled embeddings required to rerank a batch of eligible occurrences.

    Attributes:
        cand_texts: Deterministic, globally unique candidate text list used for embedding.
        cand_mat: Candidate embedding matrix of shape (N_candidates, D).
        ctx_mat: Context embedding matrix of shape (N_eligible, D).
        cand_row: Mapping from candidate text -> row index in `cand_mat`.
    """
    cand_texts: list[str]
    cand_mat: FloatMat
    ctx_mat: FloatMat
    cand_row: dict[str, int]


def _skip_tier2(r1: Tier1OccurrenceRanking, reason: Tier2SkipReason) -> Tier2OccurrenceRanking:
    """
    Create a Tier-2 ranking record representing a skipped rerank.

    Args:
        r1: Tier-1 ranking for the occurrence being skipped.
        reason: Skip reason label.

    Returns:
        A Tier2OccurrenceRanking with `applied=False` and no Tier-2 score payloads.
    """
    return Tier2OccurrenceRanking(
        occ=r1.occ,
        applied=False,
        skip_reason=reason,
        tier2_sims=None,
        blended_scores=None,
    )


def _slice_context(text: str, start: int, end: int, window_chars: int) -> str:
    """
    Deterministically extract a context window around an occurrence span.

    Args:
        text: Source document text.
        start: Occurrence start offset (inclusive).
        end: Occurrence end offset (exclusive).
        window_chars: Characters to include on each side of the span.

    Returns:
        The context substring clamped to the bounds of `text`.
    """
    L = max(0, start - window_chars)
    R = min(len(text), end + window_chars)
    return text[L:R]


def collect_tier2_inputs(
    *,
    text: str,
    t1_ranked: Sequence[Tier1OccurrenceRanking],
    sense_index: dict[str, object],  # object to avoid importing AcronymSense here; only `.definition` is used.
    window_chars: int,
    reasons: Counter[Tier2SkipReason],
) -> tuple[list[Tier2OccurrenceRanking], list[_EligibleRerank]]:
    """
    Build the default Tier-2 output list and collect eligible rerank work items.

    This function guarantees the returned Tier-2 ranking list is aligned 1:1 with
    `t1_ranked` and fully populated (no Optionals). Eligible occurrences are
    represented as placeholders (model_unavailable) until rerank is applied.

    Args:
        text: Source document text.
        t1_ranked: Tier-1 rankings to evaluate.
        sense_index: Sense lookup by id; used to format candidate texts.
        window_chars: Context window size for Tier-2.
        reasons: Counter to be updated with skip reasons for ineligible occurrences.

    Returns:
        A tuple of:
          - `ranked2`: pre-filled Tier-2 records aligned with Tier-1 order
          - `eligible`: occurrences eligible for semantic reranking
    """
    ranked2: list[Tier2OccurrenceRanking] = []
    eligible: list[_EligibleRerank] = []

    for i, r1 in enumerate(t1_ranked):
        scores = r1.candidate_scores

        if len(scores) < 2:
            reasons["single_candidate"] += 1
            ranked2.append(_skip_tier2(r1, "single_candidate"))
            continue

        if r1.chosen_sense_id is not None:
            reasons["tier1_decided"] += 1
            ranked2.append(_skip_tier2(r1, "tier1_decided"))
            continue

        context = _slice_context(text, r1.occ.start, r1.occ.end, window_chars)

        cand_ids = list(scores.keys())  # Tier-1 insertion order
        cand_texts: list[str] = []
        for sid in cand_ids:
            sense = sense_index[sid]
            # mypy: `sense` is object; we only rely on `.definition` existing.
            definition = getattr(sense, "definition")
            cand_texts.append(f"{r1.occ.acronym.upper()}: {definition}")

        eligible.append(_EligibleRerank(i, r1, context, cand_ids, cand_texts))

        # Placeholder: will be overwritten if rerank applies successfully.
        ranked2.append(_skip_tier2(r1, "model_unavailable"))

    return ranked2, eligible


def _embed_for_tier2(model_name: str, eligible: Sequence[_EligibleRerank]) -> _EmbeddingsBatch | None:
    """
    Embed all unique candidate texts and all eligible contexts in two batches.

    Args:
        model_name: Embedding model identifier passed to `embed_texts`.
        eligible: Eligible rerank work items.

    Returns:
        A batch containing candidate/context embeddings and a text->row index map,
        or None if embedding/model is unavailable.
    """
    uniq_cands: set[str] = set()
    contexts: list[str] = []

    for e in eligible:
        uniq_cands.update(e.cand_texts)
        contexts.append(e.context)

    cand_texts = sorted(uniq_cands)  # determinism
    cand_mat = embed_texts(model_name, cand_texts)
    ctx_mat = embed_texts(model_name, contexts)

    if cand_mat is None or ctx_mat is None:
        return None

    # If your semantic module already returns NDArray, these types line up.
    cand_mat = np.asarray(cand_mat)
    ctx_mat = np.asarray(ctx_mat)

    cand_row = {txt: idx for idx, txt in enumerate(cand_texts)}
    return _EmbeddingsBatch(cand_texts=cand_texts, cand_mat=cand_mat, ctx_mat=ctx_mat, cand_row=cand_row)


def apply_tier2_reranks(
    *,
    ranked2: list[Tier2OccurrenceRanking],
    eligible: Sequence[_EligibleRerank],
    batch: _EmbeddingsBatch,
    weight: float,
) -> int:
    """
    Apply semantic reranking to eligible occurrences and overwrite placeholders.

    Args:
        ranked2: Tier-2 rankings aligned with Tier-1 order; eligible slots are placeholders.
        eligible: Eligible rerank work items aligned with `batch.ctx_mat` rows.
        batch: Embedded candidate/context batch returned by `_embed_for_tier2`.
        weight: Blend weight for Tier-2 similarity in [0,1].

    Returns:
        Number of occurrences for which Tier-2 rerank was applied.
    """
    applied = 0

    for k, e in enumerate(eligible):
        ctx_vec: FloatVec = batch.ctx_mat[k]

        rows = [batch.cand_row[txt] for txt in e.cand_texts]
        cand_mat: FloatMat = batch.cand_mat[rows]

        sims01 = cosine_sim01(ctx_vec, cand_mat)  # shape (K,)

        tier2_sims: dict[str, float] = {}
        blended: dict[str, float] = {}

        for sid, sim in zip(e.cand_ids, sims01, strict=True):
            sim_f = float(sim)
            tier2_sims[sid] = sim_f
            t1_score = float(e.r1.candidate_scores[sid])
            blended[sid] = (1.0 - weight) * t1_score + weight * sim_f

        ranked2[e.idx] = Tier2OccurrenceRanking(
            occ=e.r1.occ,
            applied=True,
            skip_reason=None,
            tier2_sims=tier2_sims,
            blended_scores=blended,
        )
        applied += 1

    return applied
