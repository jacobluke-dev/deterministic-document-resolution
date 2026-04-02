from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import  Sequence

from plainera_unacronym.nlp.detection.defined_terms.types import DefinedTermMention
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermMeaning,
    TermTier1OccurrenceRanking,
    TermTier2OccurrenceRanking,
    TermTier2SkipReason,
)
from plainera_unacronym.nlp.extraction.tiers.semantic import cosine_sim01, embed_texts
from plainera_unacronym.nlp.extraction.tiers.types import Tier2Report


@dataclass(frozen=True)
class _EligibleTermRerank:
    """Intermediate payload for a Tier-2-eligible occurrence.

    Stores the minimum information required to batch-embed candidate definition
    texts and occurrence contexts before constructing the final Tier-2 ranking
    outputs.

    Attributes:
        idx: Index of the occurrence in the original Tier-1-ranked sequence.
        r1: Tier-1 ranking output for the occurrence.
        context: Local context text used as the semantic query.
        cand_ids: Candidate meaning IDs aligned with ``cand_texts``.
        cand_texts: Candidate definition texts to embed and compare against the
            occurrence context.
    """

    idx: int
    r1: TermTier1OccurrenceRanking
    context: str
    cand_ids: list[str]
    cand_texts: list[str]


def _term_context(
    text: str,
    occ: DefinedTermMention,
    *,
    window_chars: int,
) -> str:
    """Return the local context text used for Tier-2 semantic reranking.

    Prefers a precomputed ``segment_window`` on the occurrence when available.
    Otherwise, extracts a symmetric character window around the occurrence span.

    Args:
        text: Full source text containing the occurrence.
        occ: Defined-term occurrence to extract context for.
        window_chars: Number of characters to include on each side of the
            occurrence when no precomputed segment window is present.

    Returns:
        A context string surrounding the occurrence, suitable for embedding and
        semantic similarity comparison.
    """
    if occ.segment_window:
        return occ.segment_window

    start = max(0, occ.start_offset - window_chars)
    end = min(len(text), occ.end_offset + window_chars)
    return text[start:end]


def _skip(
    r1: TermTier1OccurrenceRanking,
    reason: TermTier2SkipReason,
) -> TermTier2OccurrenceRanking:
    """Build a skipped Tier-2 output row for a Tier-1-ranked occurrence.

    Args:
        r1: Tier-1 ranking output for the occurrence being skipped.
        reason: Structured reason explaining why Tier-2 was not applied.

    Returns:
        A ``TermTier2OccurrenceRanking`` with ``applied=False`` and no semantic
        similarity or blended-score payloads.
    """
    return TermTier2OccurrenceRanking(
        occ=r1.occ,
        applied=False,
        skip_reason=reason,
        tier2_sims=None,
        blended_scores=None,
    )


def _collect_tier2_eligible(
    *,
    text: str,
    t1_ranked: Sequence[TermTier1OccurrenceRanking],
    meaning_index: dict[str, TermMeaning],
    cfg: DefinedTermExtractionConfig,
) -> tuple[list[TermTier2OccurrenceRanking], list[_EligibleTermRerank], Counter[TermTier2SkipReason]]:
    """Build initial Tier-2 outputs, skipping occurrences that are not eligible.

    This pass evaluates Tier-2 gating rules such as disabled mode, single-candidate
    cases, confident Tier-1 decisions, and missing meaning definition text. Eligible
    occurrences are recorded for later semantic reranking, while ineligible ones are
    emitted immediately as skipped Tier-2 results.

    Args:
        text: Full source text containing the term occurrences.
        t1_ranked: Tier-1 ranking outputs for each detected occurrence.
        meaning_index: Mapping from meaning ID to resolved term meaning metadata.
        cfg: Active extraction configuration controlling Tier-2 behaviour.

    Returns:
        A tuple of:
            - initial Tier-2 ranking outputs containing skip placeholders,
            - eligible occurrences to be semantically reranked, and
            - skip-reason counts accumulated during eligibility filtering.
    """
    mode = cfg.tier2.mode
    auto_margin_ceiling = cfg.tier2.auto_margin_ceiling
    only_when_undecided = cfg.tier2.only_when_undecided

    reasons: Counter[TermTier2SkipReason] = Counter()
    ranked2: list[TermTier2OccurrenceRanking] = []
    eligible: list[_EligibleTermRerank] = []

    if mode == "off":
        for r1 in t1_ranked:
            ranked2.append(_skip(r1, "disabled"))
            reasons["disabled"] += 1
        return ranked2, eligible, reasons

    for idx, r1 in enumerate(t1_ranked):
        scores = r1.candidate_scores

        if len(scores) < 2:
            ranked2.append(_skip(r1, "single_candidate"))
            reasons["single_candidate"] += 1
            continue

        if only_when_undecided and r1.chosen_meaning_id is not None:
            ranked2.append(_skip(r1, "tier1_decided"))
            reasons["tier1_decided"] += 1
            continue

        if mode == "auto" and r1.margin >= auto_margin_ceiling:
            ranked2.append(_skip(r1, "tier1_confident"))
            reasons["tier1_confident"] += 1
            continue

        cand_ids = list(scores.keys())
        cand_texts: list[str] = []

        for sid in cand_ids:
            meaning = meaning_index.get(sid)
            if meaning is None or not meaning.definition_text:
                cand_texts = []
                break
            cand_texts.append(f"{r1.occ.term}: {meaning.definition_text}")

        if not cand_texts:
            ranked2.append(_skip(r1, "no_meanings"))
            reasons["no_meanings"] += 1
            continue

        context = _term_context(
            text,
            r1.occ,
            window_chars=cfg.tier_1_window_chars,
        )

        eligible.append(
            _EligibleTermRerank(
                idx=idx,
                r1=r1,
                context=context,
                cand_ids=cand_ids,
                cand_texts=cand_texts,
            )
        )
        ranked2.append(_skip(r1, "pending"))
        reasons["pending"] += 1

    return ranked2, eligible, reasons


def _apply_model_unavailable_fallback(
    *,
    ranked2: list[TermTier2OccurrenceRanking],
    eligible: list[_EligibleTermRerank],
    reasons: Counter[TermTier2SkipReason],
) -> tuple[list[TermTier2OccurrenceRanking], Tier2Report]:
    """Replace pending eligible rows with model-unavailable skips.

    This helper is used when Tier-2 embeddings cannot be produced. Each eligible
    occurrence is downgraded to a skipped Tier-2 result with reason
    ``"model_unavailable"``.

    Args:
        ranked2: Current Tier-2 output rows, containing pending placeholders for
            eligible occurrences.
        eligible: Eligible occurrences that were awaiting semantic reranking.
        reasons: Skip-reason counter to update.

    Returns:
        A tuple of:
            - the updated Tier-2 ranking outputs, and
            - a ``Tier2Report`` summarising the fallback outcome.
    """
    reasons["model_unavailable"] += len(eligible)
    reasons.pop("pending", None)

    for e in eligible:
        ranked2[e.idx] = _skip(e.r1, "model_unavailable")

    return ranked2, Tier2Report(
        applied=0,
        skipped=len(ranked2),
        reasons=dict(reasons),
    )


def _apply_semantic_rerank(
    *,
    ranked2: list[TermTier2OccurrenceRanking],
    eligible: list[_EligibleTermRerank],
    reasons: Counter[TermTier2SkipReason],
    weight: float,
    uniq_cands: list[str],
    cand_mat,
    ctx_mat,
) -> tuple[list[TermTier2OccurrenceRanking], Tier2Report]:
    """Apply Tier-2 semantic similarity and blended scoring to eligible rows.

    For each eligible occurrence, this helper computes semantic similarity between
    the occurrence context and each candidate definition, blends the similarity
    with the existing Tier-1 score, and writes an applied Tier-2 ranking result.

    Args:
        ranked2: Current Tier-2 output rows, containing pending placeholders for
            eligible occurrences.
        eligible: Eligible occurrences awaiting semantic reranking.
        reasons: Skip-reason counter to finalise.
        weight: Tier-2 blending weight in the range ``[0, 1]``.
        uniq_cands: Unique candidate-definition texts embedded for this rerank run.
        cand_mat: Embedded candidate-definition matrix.
        ctx_mat: Embedded occurrence-context matrix.

    Returns:
        A tuple of:
            - the updated Tier-2 ranking outputs, and
            - a ``Tier2Report`` summarising applied and skipped counts.
    """
    cand_row = {txt: i for i, txt in enumerate(uniq_cands)}
    applied = 0

    for k, e in enumerate(eligible):
        ctx_vec = ctx_mat[k]
        rows = [cand_row[txt] for txt in e.cand_texts]
        sims01 = cosine_sim01(ctx_vec, cand_mat[rows])

        tier2_sims: dict[str, float] = {}
        blended: dict[str, float] = {}

        for sid, sim in zip(e.cand_ids, sims01, strict=True):
            sim_f = float(sim)
            tier2_sims[sid] = sim_f
            t1_score = float(e.r1.candidate_scores[sid])
            blended[sid] = (1.0 - weight) * t1_score + weight * sim_f

        ranked2[e.idx] = TermTier2OccurrenceRanking(
            occ=e.r1.occ,
            applied=True,
            skip_reason=None,
            tier2_sims=tier2_sims,
            blended_scores=blended,
        )
        applied += 1

    reasons.pop("pending", None)

    return ranked2, Tier2Report(
        applied=applied,
        skipped=len(ranked2) - applied,
        reasons=dict(reasons),
    )


def rerank_term_occurrences_tier2(
    *,
    text: str,
    t1_ranked: Sequence[TermTier1OccurrenceRanking],
    meaning_index: dict[str, TermMeaning],
    cfg: DefinedTermExtractionConfig,
) -> tuple[list[TermTier2OccurrenceRanking], Tier2Report]:
    """Optionally rerank ambiguous defined-term occurrences using semantic similarity.

    Tier-2 reranking is only attempted for occurrences that remain eligible after
    deterministic Tier-1 filtering. Ineligible occurrences are emitted with a skip
    reason. For eligible cases, semantic similarity between local occurrence context
    and candidate definition text is blended with Tier-1 scores to produce final
    Tier-2 outputs.

    Args:
        text: Full source text containing the term occurrences.
        t1_ranked: Tier-1 ranking outputs for each detected occurrence.
        meaning_index: Mapping from meaning ID to resolved term meaning metadata.
        cfg: Active extraction configuration controlling Tier-2 mode, thresholds,
            model name, and blend weight.

    Returns:
        A tuple of:
            - Tier-2 ranking outputs aligned to ``t1_ranked``, and
            - a ``Tier2Report`` summarising how many rows were applied or skipped
              and why.
    """
    weight = cfg.tier2.weight
    model_name = cfg.tier2.model_name

    ranked2, eligible, reasons = _collect_tier2_eligible(
        text=text,
        t1_ranked=t1_ranked,
        meaning_index=meaning_index,
        cfg=cfg,
    )

    if not eligible:
        reasons.pop("pending", None)
        return ranked2, Tier2Report(
            applied=0,
            skipped=len(ranked2),
            reasons=dict(reasons),
        )

    uniq_cands = sorted({txt for e in eligible for txt in e.cand_texts})
    ctx_texts = [e.context for e in eligible]

    cand_mat = embed_texts(uniq_cands, model_name=model_name)
    ctx_mat = embed_texts(ctx_texts, model_name=model_name)

    if cand_mat is None or ctx_mat is None:
        return _apply_model_unavailable_fallback(
            ranked2=ranked2,
            eligible=eligible,
            reasons=reasons,
        )

    return _apply_semantic_rerank(
        ranked2=ranked2,
        eligible=eligible,
        reasons=reasons,
        weight=weight,
        uniq_cands=uniq_cands,
        cand_mat=cand_mat,
        ctx_mat=ctx_mat,
    )
