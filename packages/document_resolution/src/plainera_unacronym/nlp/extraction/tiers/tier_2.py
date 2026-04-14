from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from plainera_unacronym.nlp.common.types import AcronymMeaning
from plainera_unacronym.nlp.extraction.tiers.semantic import cosine_sim01, embed_texts
from plainera_unacronym.nlp.extraction.tiers.types import (
    FloatMat,
    FloatVec,
    Tier1OccurrenceRanking,
    Tier2OccurrenceRanking,
    Tier2SkipReason,
)

# Internal safety rails only.
# These are not "policy knobs" exposed to callers.
_T2_MIN_CONTEXT_CHARS = 80
_T2_HARD_CAP_CHARS = 320


@dataclass(frozen=True)
class _EligibleRerank:
    """
    Work item describing a Tier-1 occurrence eligible for Tier-2 reranking.

    Attributes:
        idx: Position of the occurrence in `t1.ranked` (and therefore in the final Tier-2 list).
        r1: Tier-1 ranking record for the occurrence.
        context: Context window string used for semantic embedding.
        cand_ids: Candidate meaning IDs in Tier-1 insertion order.
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


def _clamp_span(start: int, end: int, text_len: int) -> tuple[int, int]:
    start = max(0, min(start, text_len))
    end = max(start, min(end, text_len))
    return start, end


def _trim_to_whitespace(text: str, start: int, end: int, max_adjust: int = 40) -> tuple[int, int]:
    text_len = len(text)
    start, end = _clamp_span(start, end, text_len)

    moved = 0
    while start > 0 and moved < max_adjust and not text[start].isspace():
        start -= 1
        moved += 1

    moved = 0
    while end < text_len and moved < max_adjust and not text[end - 1].isspace():
        end += 1
        moved += 1

    return _clamp_span(start, end, text_len)


def _find_sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Return the sentence-like span containing the occurrence.

    Sentence boundaries are approximated deterministically using:
      - newline boundaries
      - '.', '!', '?', ';'
    """
    text_len = len(text)
    start, end = _clamp_span(start, end, text_len)

    left = start
    while left > 0:
        ch = text[left - 1]
        if ch in ".!?;\n":
            break
        left -= 1

    right = end
    while right < text_len:
        ch = text[right]
        if ch in ".!?;\n":
            right += 1  # include boundary punctuation/newline
            break
        right += 1

    left, right = _trim_to_whitespace(text, left, right)
    return left, right


def _find_prev_sentence_span(text: str, sentence_start: int) -> tuple[int, int] | None:
    if sentence_start <= 0:
        return None

    i = sentence_start - 1
    while i > 0 and text[i - 1].isspace():
        i -= 1
    if i <= 0:
        return None

    end = i
    start, _ = _find_sentence_span(text, max(0, end - 1), end)
    return start, end


def _find_next_sentence_span(text: str, sentence_end: int) -> tuple[int, int] | None:
    text_len = len(text)
    i = sentence_end
    while i < text_len and text[i].isspace():
        i += 1
    if i >= text_len:
        return None

    _, end = _find_sentence_span(text, i, min(i + 1, text_len))
    return i, end


def _enforce_hard_cap(text: str, start: int, end: int, occ_start: int, occ_end: int) -> tuple[int, int]:
    """
    If the span is too large, clamp it around the occurrence while preserving locality.
    """
    text_len = len(text)
    start, end = _clamp_span(start, end, text_len)

    if (end - start) <= _T2_HARD_CAP_CHARS:
        return start, end

    half = max(1, _T2_HARD_CAP_CHARS // 2)
    clipped_start = max(0, occ_start - half)
    clipped_end = min(text_len, occ_end + half)
    return _trim_to_whitespace(text, clipped_start, clipped_end)


def _resolve_tier2_context_span(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Resolve a deterministic local context span for a Tier-2 occurrence.

    Policy:
      1) use the containing sentence
      2) if too short, expand with a neighbouring sentence
      3) apply a hard cap only as a safety rail
    """
    sent_start, sent_end = _find_sentence_span(text, start, end)
    span_start, span_end = sent_start, sent_end

    if (span_end - span_start) < _T2_MIN_CONTEXT_CHARS:
        prev_span = _find_prev_sentence_span(text, sent_start)
        next_span = _find_next_sentence_span(text, sent_end)

        candidates: list[tuple[int, int]] = [(span_start, span_end)]

        if prev_span is not None:
            candidates.append((prev_span[0], span_end))
        if next_span is not None:
            candidates.append((span_start, next_span[1]))
        if prev_span is not None and next_span is not None:
            candidates.append((prev_span[0], next_span[1]))

        # Choose the smallest span that reaches the minimum target, else the largest available.
        valid = [c for c in candidates if (c[1] - c[0]) >= _T2_MIN_CONTEXT_CHARS]
        if valid:
            span_start, span_end = min(valid, key=lambda x: (x[1] - x[0]))
        else:
            span_start, span_end = max(candidates, key=lambda x: (x[1] - x[0]))

    span_start, span_end = _enforce_hard_cap(text, span_start, span_end, start, end)
    return span_start, span_end


def _slice_tier2_context(text: str, start: int, end: int) -> str:
    """
    Return the Tier-2 local context slice for an acronym occurrence.

    The slice is derived from the occurrence's containing sentence. If that
    sentence is too short, an adjacent sentence may be included to provide
    sufficient local context. A hard cap is applied only as a safety rail.

    Args:
        text: Full source text.
        start: Start offset of the occurrence.
        end: End offset (exclusive) of the occurrence.

    Returns:
        A deterministic local context slice for Tier-2 semantic reranking.
    """
    span_start, span_end = _resolve_tier2_context_span(text, start, end)
    return text[span_start:span_end]


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


def collect_tier2_inputs(
    *,
    text: str,
    t1_ranked: Sequence[Tier1OccurrenceRanking],
    meaning_index: dict[str, AcronymMeaning],
    auto_margin_ceiling: float,
    mode: Literal["off", "auto", "on"],
    only_when_undecided: bool,
    ambiguous_acrs: set[str],  # EXPECTED UPPERCASE
    reasons: Counter[Tier2SkipReason],
) -> tuple[list[Tier2OccurrenceRanking], list[_EligibleRerank]]:
    """
    Collect Tier-2 eligible occurrences and return:
      - ranked2: 1:1 aligned with t1_ranked (placeholders for eligible)
      - eligible: rerank work items (subset)

    Eligibility:
      - >=2 candidates
      - acronym is multi-meaning (in ambiguous_acrs)
      - if only_when_undecided: Tier-1 must not have chosen (chosen_meaning_id is None)
      - AUTO mode: skip if Tier-1 margin >= auto_margin_ceiling
      - ON mode: ignore margin ceiling (still respects only_when_undecided if set)
    """
    ranked2: list[Tier2OccurrenceRanking] = []
    eligible: list[_EligibleRerank] = []

    for i, r1 in enumerate(t1_ranked):
        scores = r1.candidate_scores

        if len(scores) < 2:
            reasons["single_candidate"] += 1
            ranked2.append(_skip_tier2(r1, "single_candidate"))
            continue

        acr = r1.occ.acronym.upper()
        if acr not in ambiguous_acrs:
            reasons["not_ambiguous"] += 1
            ranked2.append(_skip_tier2(r1, "not_ambiguous"))
            continue

        if mode != "on" and only_when_undecided and r1.chosen_meaning_id is not None:
            reasons["tier1_decided"] += 1
            ranked2.append(_skip_tier2(r1, "tier1_decided"))
            continue

        if mode == "auto" and r1.margin >= auto_margin_ceiling:
            reasons["tier1_confident"] += 1
            ranked2.append(_skip_tier2(r1, "tier1_confident"))
            continue

        context = _slice_tier2_context(text, r1.occ.start, r1.occ.end)

        cand_ids = list(scores.keys())
        cand_texts: list[str] = []
        for sid in cand_ids:
            meaning = meaning_index.get(sid)
            definition = getattr(meaning, "definition", None) if meaning is not None else None
            if not definition:
                reasons["no_meanings"] += 1
                cand_texts = []
                break
            cand_texts.append(f"{acr}: {definition}")

        if not cand_texts:
            ranked2.append(_skip_tier2(r1, "no_meanings"))
            continue

        eligible.append(_EligibleRerank(i, r1, context, cand_ids, cand_texts))
        ranked2.append(_skip_tier2(r1, "pending"))

    return ranked2, eligible


def embed_for_tier2(
    eligible: Sequence[_EligibleRerank],
    *,
    tier2_model: Any | None = None,
    model_name: str | None = None,
) -> _EmbeddingsBatch | None:
    """
    Embed all unique candidate texts and all eligible contexts in two batches.

    Args:
        eligible: Eligible rerank work items.
        model_name: Embedding model identifier passed to `embed_texts`.
        tier2_model: Tier2 model identifier passed to `embed_texts`.

    Returns:
        A batch containing candidate/context embeddings and a text->row index map,
        or None if embedding/model is unavailable.
    """
    uniq_cands: set[str] = set()
    ctx_texts: list[str] = []  # keep duplicates; must align 1:1 with eligible

    for e in eligible:
        uniq_cands.update(e.cand_texts)
        ctx_texts.append(e.context)

    cand_texts = sorted(uniq_cands)  # determinism

    cand_mat = embed_texts(cand_texts, model=tier2_model, model_name=model_name)
    ctx_mat = embed_texts(ctx_texts, model=tier2_model, model_name=model_name)

    if cand_mat is None or ctx_mat is None:
        return None

    cand_mat = np.asarray(cand_mat, dtype=np.float32)
    ctx_mat = np.asarray(ctx_mat, dtype=np.float32)

    cand_row = {txt: idx for idx, txt in enumerate(cand_texts)}
    return _EmbeddingsBatch(
        cand_texts=cand_texts,
        cand_mat=cand_mat,
        ctx_mat=ctx_mat,
        cand_row=cand_row,
    )


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
        batch: Embedded candidate/context batch returned by `embed_for_tier2`.
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
