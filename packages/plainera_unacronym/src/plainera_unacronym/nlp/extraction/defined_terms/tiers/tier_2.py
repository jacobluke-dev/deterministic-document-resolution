from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from plainera_unacronym.nlp.detection.defined_terms.types import DefinedTermMention
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermSense,
    TermTier1OccurrenceRanking,
    TermTier2OccurrenceRanking,
    TermTier2SkipReason,
)
from plainera_unacronym.nlp.extraction.tiers.semantic import cosine_sim01, embed_texts
from plainera_unacronym.nlp.extraction.tiers.types import Tier2Report


@dataclass(frozen=True)
class _EligibleTermRerank:
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
    if occ.segment_window:
        return occ.segment_window

    start = max(0, occ.start_offset - window_chars)
    end = min(len(text), occ.end_offset + window_chars)
    return text[start:end]


def _skip(
    r1: TermTier1OccurrenceRanking,
    reason: TermTier2SkipReason,
) -> TermTier2OccurrenceRanking:
    return TermTier2OccurrenceRanking(
        occ=r1.occ,
        applied=False,
        skip_reason=reason,
        tier2_sims=None,
        blended_scores=None,
    )


def rerank_term_occurrences_tier2(
    *,
    text: str,
    t1_ranked: Sequence[TermTier1OccurrenceRanking],
    sense_index: dict[str, TermSense],
    cfg: DefinedTermExtractionConfig,
) -> tuple[list[TermTier2OccurrenceRanking], Tier2Report]:
    mode = cfg.tier2.mode
    weight = cfg.tier2.weight
    auto_margin_ceiling = cfg.tier2.auto_margin_ceiling
    only_when_undecided = cfg.tier2.only_when_undecided
    model_name = cfg.tier2.model_name

    reasons: Counter[TermTier2SkipReason] = Counter()
    ranked2: list[TermTier2OccurrenceRanking] = []
    eligible: list[_EligibleTermRerank] = []

    if mode == "off":
        for r1 in t1_ranked:
            ranked2.append(_skip(r1, "disabled"))
            reasons["disabled"] += 1

        return ranked2, Tier2Report(
            applied=0,
            skipped=len(ranked2),
            reasons=dict(reasons),
        )

    for idx, r1 in enumerate(t1_ranked):
        scores = r1.candidate_scores

        if len(scores) < 2:
            ranked2.append(_skip(r1, "single_candidate"))
            reasons["single_candidate"] += 1
            continue

        if only_when_undecided and r1.chosen_sense_id is not None:
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
            sense = sense_index.get(sid)
            if sense is None or not sense.definition_text:
                cand_texts = []
                break
            cand_texts.append(f"{r1.occ.term}: {sense.definition_text}")

        if not cand_texts:
            ranked2.append(_skip(r1, "no_senses"))
            reasons["no_senses"] += 1
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
        reasons["model_unavailable"] += len(eligible)
        reasons.pop("pending", None)

        for e in eligible:
            ranked2[e.idx] = _skip(e.r1, "model_unavailable")

        return ranked2, Tier2Report(
            applied=0,
            skipped=len(ranked2),
            reasons=dict(reasons),
        )

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
