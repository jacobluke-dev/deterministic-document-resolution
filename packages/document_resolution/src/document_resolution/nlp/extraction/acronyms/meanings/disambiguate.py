"""
Acronym meaning disambiguation for documents with multiple in-text definitions.

This module resolves each acronym occurrence to the most likely meaning
(`AcronymMeaning`) when a document defines the same acronym more than once.
Meanings are built from extracted definitions and tracked with definition spans.
Each occurrence is scored against candidate meanings using proximity to definition
spans and local context token overlap, with a conservative margin-based tiebreak.
If no meaning is clearly dominant, the occurrence is left undecided rather than
assigned incorrectly.

This stage enables per-occurrence correctness and ambiguity detection beyond
a single global glossary pick.
"""

from __future__ import annotations

import re

from document_resolution.nlp.common.types import AcronymMeaning, OccurrenceLite, OccurrenceResolution, Span

NEAR_TIE_GAP = 0.06


def _ascii_tokens(s: str) -> list[str]:
    """
    Tokenize ASCII-ish words/numbers with optional internal apostrophes/hyphens.

    Pattern:
        - First char must be ASCII letter or digit: [A-Za-z0-9]
        - Followed by zero or more of letters/digits/apostrophe/hyphen: [A-Za-z0-9'-]*
        - Case-insensitive: input is lowercased before matching
        - Underscores, dots, and non-ASCII letters (e.g., é, ï) break tokens

    Args:
        s: Raw input string.

    Returns:
        list[str]: Lowercased tokens (e.g., "rock'n'roll", "state-of-the-art", "a1b2").

    Notes:
        - Leading apostrophes/hyphens are not allowed (e.g., "'tis" → "tis", "-dash" → "dash").
        - Trailing apostrophes/hyphens are kept if present in the match (e.g., "james'").
        - Emails/URLs are split at punctuation (e.g., "email@example.com" → ["email","example","com"]).
    """
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", s.lower())


def _center(s: int, e: int) -> float:
    """Return the midpoint of an interval.
    """
    return (s + e) / 2.0


def _min_distance_to_spans(pos: float, spans: list[Span]) -> int:
    """
    Return the minimum integer distance from a position to the centers of spans.

    Args:
        pos: Position (float) to measure from.
        spans: List of (start, end) integer pairs representing spans. The order of
            `start` and `end` does not matter; centers are symmetric.

    Returns:
        int: Minimum floored distance to any span center, or 1_000_000_000 if none.

    Notes:
        - Uses floor via `int(...)`, so distances in [0.0, 1.0) map to 0.
        - Reversed endpoints (e.g., (5, 1)) are handled naturally by `_center`.
        - Intended for coarse tiebreaking; prefer smaller distances.

    """
    best = 10**9
    for s, e in spans:
        c = _center(s, e)
        d = abs(c - pos)
        if d < best:
            best = int(d)
    return best


# Require a clear spatial advantage (≥3 chars) to break near-score ties
DIST_TIEBREAK_MIN_ADVANTAGE = 3


def choose_with_tiebreak(
    occ: OccurrenceLite,
    cand_scores: dict[str, float],
    meanings_by_id: dict[str, AcronymMeaning],
    *,
    margin_threshold: float = 0.10,
    near_tie_margin: float = 0.06,
) -> tuple[None, float, float] | tuple[str, float, float]:
    """
    Pick a winning meaning from candidate scores using a relative-margin rule first,
    then (if needed) a spatial tiebreak using definition-span proximity.

    Args:
        occ:
            Occurrence to resolve (expects integer start/end offsets).
        cand_scores:
            Mapping meaning_id -> score. Scores are heuristic composites (distance/overlap/prior)
            and are not calibrated probabilities.
        meanings_by_id:
            Mapping meaning_id -> AcronymMeaning. Used for:
              - definition spans (distance tiebreak)
              - optional meaning_confidence as a deterministic sorting key
        margin_threshold:
            Minimum relative margin to accept the top score:
                (p1 - p2) / max(p1, 1e-9) >= margin_threshold
        near_tie_margin:
            Absolute score gap threshold to engage distance tiebreak:
                (p1 - p2) <= near_tie_margin

    Returns:
        tuple[str | None, float]:
            (chosen_meaning_id_or_none, gap)

            gap is the absolute top-two score gap (p1 - p2).
            It is 0.0 when there are no candidates, and equals p1 when there is only one candidate
            (since p2 is treated as 0.0).

    """
    items = sorted(
        cand_scores.items(),
        key=lambda kv: (kv[1], getattr(meanings_by_id.get(kv[0]), "meaning_confidence", 0.0), kv[0]),
        reverse=True,
    )

    if not items:
        return None, 0.0, 0.0

    sid1, p1 = items[0]
    p2 = items[1][1] if len(items) > 1 else 0.0

    gap = p1 - p2
    rel_margin = gap / max(p1, 1e-9)

    # 1) accept probabilistic winner if relative margin is strong
    if rel_margin >= margin_threshold:
        return sid1, rel_margin, gap

    # 2) near tie -> engage distance tiebreak
    if len(items) > 1 and gap <= near_tie_margin:
        pos = _center(occ.start, occ.end)

        def dist_for(sid: str) -> int:
            spans = getattr(meanings_by_id.get(sid), "def_spans", None) or []
            return _min_distance_to_spans(pos, spans)

        sid2 = items[1][0]
        d1, d2 = dist_for(sid1), dist_for(sid2)

        if d1 - d2 >= DIST_TIEBREAK_MIN_ADVANTAGE:
            return sid2, rel_margin, gap
        if d2 - d1 >= DIST_TIEBREAK_MIN_ADVANTAGE:
            return sid1, rel_margin, gap

    # 3) unresolved
    return None, rel_margin, gap


def prior_weight_for_gap(gap: float, *, max_w: float = 0.08, engage_gap: float = 0.06) -> float:
    """Return dynamic prior weight based on top-two score gap.

    Args:
        gap: Absolute (p1 - p2) gap (>=0).
        max_w: Maximum weight applied at exact tie.
        engage_gap: Gap threshold above which the prior is disabled.

    Returns:
        A weight in [0, max_w].
    """
    if gap >= engage_gap or max_w <= 0.0:
        return 0.0
    g = max(0.0, gap)
    return max_w * (1.0 - (g / engage_gap))


def meaning_prior_term(*, meaning_confidence: float, weight: float) -> float:
    """Return additive prior term `weight * meaning_confidence` (clamped).

    Args:
        meaning_confidence: Meaning confidence in [0, 1].
        weight: Prior weight in [0, 1] (typically small).

    Returns:
        Additive score contribution.
    """
    if weight <= 0.0:
        return 0.0
    c = 0.0 if meaning_confidence < 0.0 else (1.0 if meaning_confidence > 1.0 else meaning_confidence)
    return weight * c


def dynamic_prior_weight(
    base_scores: dict[str, float],
    *,
    max_w: float,
    engage_gap: float,
) -> float:
    """
    Compute a dynamic prior weight for near-ties based on the top-two score gap.

    Args:
        base_scores: Mapping of meaning_id -> base score (no prior).
        max_w: Maximum prior weight applied at exact tie; set 0.0 to disable.
        engage_gap: Absolute gap threshold above which prior is disabled.

    Returns:
        A weight in [0.0, max_w] used to scale `meaning_confidence` as an additive prior.
    """
    if max_w == 0:
        return 0.0
    if max_w <= 0.0 or len(base_scores) < 2:
        return 0.0

    items = sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)
    p1 = items[0][1]
    p2 = items[1][1]
    gap = p1 - p2
    return prior_weight_for_gap(gap, max_w=max_w, engage_gap=engage_gap)


def base_scores_for_occurrence(
    *,
    occ: OccurrenceLite,
    meanings_list: list[AcronymMeaning],
    ctx_tokens: set[str],
    dist_weight: float,
    overlap_weight: float,
) -> dict[str, float]:
    """
    Compute per-meaning base scores for a single occurrence.
    Args:
        occ: Occurrence being resolved.
        meanings_list: Candidate meanings for the occurrence acronym.
        ctx_tokens: Tokens from the occurrence context window.
        dist_weight: Weight for proximity score.
        overlap_weight: Weight for overlap score.

    Returns:
        Mapping of meaning_id -> base score.
    """
    out: dict[str, float] = {}

    for s in meanings_list:
        # distance score to nearest def span centre
        if s.def_spans:
            d = min(abs(occ.start - ((a + b) // 2)) for (a, b) in s.def_spans)
            dist_score = 1.0 / (1.0 + d)
        else:
            dist_score = 0.0

        # label overlap scores
        label_tokens = set(_ascii_tokens(s.definition))
        overlap = len(label_tokens & ctx_tokens) / max(1, len(label_tokens)) if label_tokens else 0.0

        out[s.meaning_id] = dist_weight * dist_score + overlap_weight * overlap

    return out


def disambiguate_occurrences(
    text: str,
    occurrences: list[OccurrenceLite],
    meanings: dict[str, list[AcronymMeaning]],
    *,
    window_chars: int = 300,
    meanings_prior_weight: float = 0.02,
    margin_threshold: float = 0.10,
    dist_weight: float = 0.75,
    overlap_weight: float = 0.25,
    meanings_by_id: dict[str, AcronymMeaning] | None = None,
) -> list[OccurrenceResolution]:
    """
    Resolve acronym occurrences to the most likely meaning.

    Args:
        text: Full source text.
        occurrences: OccurrenceLite list (acronym, start, end).
        meanings: Mapping from UPPER(acronym) to candidate AcronymMeaning list.
        window_chars: Half-window (chars) around each occurrence to form ctx_tokens.
        meanings_prior_weight: Maximum weight for an optional confidence prior.
        margin_threshold: Relative margin needed to accept the probabilistic winner.
        dist_weight: Weight for distance score.
        overlap_weight: Weight for overlap score.
        meanings_by_id: Optional {meaning_id → AcronymMeaning}; if None, built from `meanings`.

    Returns:
        List of OccurrenceResolution in the same order as `occurrences`. Each item
        includes chosen_meaning_id (or None), per-meaning scores, and the top-two margin.
    """
    results: list[OccurrenceResolution] = []
    meanings_by_id = meanings_by_id or {s.meaning_id: s for lst in meanings.values() for s in lst}

    for occ in occurrences:
        meanings_list = meanings.get(occ.acronym.upper(), [])
        if not meanings_list:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0, 0.0))
            continue

        L = max(0, occ.start - window_chars)
        R = min(len(text), occ.end + window_chars)
        ctx_tokens = set(_ascii_tokens(text[L:R]))

        base_scores = base_scores_for_occurrence(
            occ=occ,
            meanings_list=meanings_list,
            ctx_tokens=ctx_tokens,
            dist_weight=dist_weight,
            overlap_weight=overlap_weight,
        )

        if not base_scores:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0, 0.0))
            continue

        w = dynamic_prior_weight(base_scores, max_w=meanings_prior_weight, engage_gap=NEAR_TIE_GAP)

        cand_scores: dict[str, float] = dict(base_scores)
        if w:
            for sid in cand_scores:
                sc = getattr(meanings_by_id.get(sid), "meaning_confidence", 0.0)
                cand_scores[sid] += meaning_prior_term(meaning_confidence=sc, weight=w)

        chosen, rel_margin, gap = choose_with_tiebreak(
            occ,
            cand_scores,
            meanings_by_id,
            margin_threshold=margin_threshold,
            near_tie_margin=NEAR_TIE_GAP,
        )

        results.append(
            OccurrenceResolution(
                occ.acronym,
                occ.start,
                occ.end,
                chosen,
                cand_scores,
                gap=gap,
                margin=rel_margin,
            )
        )

    return results
