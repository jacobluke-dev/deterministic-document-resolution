"""
Acronym sense disambiguation for documents with multiple in-text definitions.

This module resolves each acronym occurrence to the most likely meaning
(`AcronymSense`) when a document defines the same acronym more than once.
Senses are built from extracted definitions and tracked with definition spans.
Each occurrence is scored against candidate senses using proximity to definition
spans and local context token overlap, with a conservative margin-based tiebreak.
If no sense is clearly dominant, the occurrence is left undecided rather than
assigned incorrectly.

This stage enables per-occurrence correctness and ambiguity detection beyond
a single global glossary pick.
"""

import re

from plainera_unacronym.nlp.common.types import AcronymSense, OccurrenceLite, OccurrenceResolution, Span

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

    Args:
        s: Interval start.
        e: Interval end.

    Returns:
        The midpoint `(s + e) / 2.0` as a float.
    """
    return (s + e) / 2.0


def _min_distance_to_spans(pos: float, spans: list[Span]) -> int:
    """
    Return the minimum integer distance from a position to the centers of spans.

    For each (start, end) span, the center is computed as `_center(start, end)`.
    The function returns the smallest floor distance `int(abs(center - pos))`
    over all spans. If `spans` is empty, returns a large sentinel (1_000_000_000).

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
    best = 10 ** 9
    for s, e in spans:
        c = _center(s, e)
        d = abs(c - pos)
        if d < best:
            best = int(d)
    return best


# Require a clear spatial advantage (≥3 chars) to break near-score ties
DIST_TIEBREAK_MIN_ADVANTAGE = 3


def choose_with_tiebreak(
    occ, cand_scores, senses_by_id, *, margin_threshold: float = 0.10, near_tie_margin: float = 0.06
) -> tuple[str | None, float]:
    """
    Pick a winning sense from candidate probabilities, using a relative-margin
    rule first and (if needed) a spatial tiebreak based on definition-span
    proximity.

    Process:
      1) Sort candidates by score desc (from `cand_scores`).
      2) Compute relative margin: (p1 - p2) / max(p1, 1e-9). If ≥ margin_threshold,
         return top sense.
      3) Near tie: if (p1 - p2) ≤ near_tie_margin and there are ≥2 items, compare
         the occurrence center to each sense’s definition spans via
         `_min_distance_to_spans`. If one sense is ≥3 units closer (i.e., other + 2 < self),
         return that sense.
      4) Otherwise, return (None, margin).

    Args:
        occ (OccurrenceLite):
            The occurrence to resolve; must expose `start` and `end` (ints).
        cand_scores (dict[str, float]):
            Candidate scores in [0,0.9...] keyed by `sense_id`. Typically derived from
            distance/overlap scoring.
        senses_by_id (dict[str, AcronymSense]):
            Lookup for senses by `sense_id`. Each sense may provide `def_spans`
            (list[Span]) used for distance tiebreaking.
        margin_threshold (float, optional):
            Minimum relative margin to accept the probabilistic winner. Default 0.10.
        near_tie_margin (float, optional):
            Absolute score gap at which to engage the distance tiebreak when the
            relative margin test fails. Default 0.06.

    Returns:
        tuple[str | None, float]:
            `(sense_id_or_none, margin)` where `margin` is the relative margin between
            the top two candidates. If no candidates, returns `(None, 0.0)`.

    Notes:
        - Occurrence center is `(start + end) / 2.0`.
        - Distance tiebreak uses centers of `def_spans`; missing/empty spans yield a
          large sentinel distance, biasing toward senses with real spans.
        - The ±2 bias (≥3 units closer) avoids flapping when distances are almost equal.
    """
    items = sorted(
        cand_scores.items(),
        key=lambda kv: (kv[1], getattr(senses_by_id.get(kv[0]), "sense_confidence", 0.0), kv[0]),
        reverse=True,
    )

    if not items:
        return None, 0.0
    (sid1, p1) = items[0]
    p2 = items[1][1] if len(items) > 1 else 0.0
    margin = (p1 - p2) / max(p1, 1e-9)
    if margin >= margin_threshold:
        return sid1, margin

    # near tie → distance tiebreak
    pos = _center(occ.start, occ.end)

    def dist_for(sid):
        spans = getattr(senses_by_id[sid], "def_spans", []) or []
        return _min_distance_to_spans(pos, spans)

    if (p1 - p2) <= near_tie_margin and len(items) > 1:
        sid2 = items[1][0]
        d1, d2 = dist_for(sid1), dist_for(sid2)
        if d1 - d2 >= DIST_TIEBREAK_MIN_ADVANTAGE:
            return sid2, margin
        if d2 - d1 >= DIST_TIEBREAK_MIN_ADVANTAGE:
            return sid1, margin

    return None, margin


def prior_weight_for_gap(gap: float, *, max_w: float = 0.08, engage_gap: float = 0.06) -> float:
    """Return dynamic prior weight based on top-two score gap.

    Engages only for near-ties: returns 0 when gap >= engage_gap, else ramps up to
    max_w as gap approaches 0.

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


def sense_prior_term(*, sense_confidence: float, weight: float) -> float:
    """Return additive prior term `weight * sense_confidence` (clamped).

    Args:
        sense_confidence: Sense confidence in [0, 1].
        weight: Prior weight in [0, 1] (typically small).

    Returns:
        Additive score contribution.
    """
    if weight <= 0.0:
        return 0.0
    c = 0.0 if sense_confidence < 0.0 else (1.0 if sense_confidence > 1.0 else sense_confidence)
    return weight * c


def dynamic_prior_weight(
    base_scores: dict[str, float],
    *,
    max_w: float,
    engage_gap: float,
) -> float:
    """
    Compute a dynamic prior weight for near-ties based on the top-two score gap.

    Engages only when there are at least two candidates and the top-two gap is below
    `engage_gap`; otherwise returns 0.0.

    Args:
        base_scores: Mapping of sense_id -> base score (no prior).
        max_w: Maximum prior weight applied at exact tie; set 0.0 to disable.
        engage_gap: Absolute gap threshold above which prior is disabled.

    Returns:
        A weight in [0.0, max_w] used to scale `sense_confidence` as an additive prior.
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
    sense_list: list[AcronymSense],
    ctx_tokens: set[str],
    dist_weight: float,
    overlap_weight: float,
) -> dict[str, float]:
    """
    Compute per-sense base scores for a single occurrence.

    Scores combine distance-to-definition-span proximity and local token overlap, with
    no confidence prior applied.

    Args:
        occ: Occurrence being resolved.
        sense_list: Candidate senses for the occurrence acronym.
        ctx_tokens: Tokens from the occurrence context window.
        dist_weight: Weight for proximity score.
        overlap_weight: Weight for overlap score.

    Returns:
        Mapping of sense_id -> base score.
    """
    out: dict[str, float] = {}

    for s in sense_list:
        # distance score to nearest def span centre
        if s.def_spans:
            d = min(abs(occ.start - ((a + b) // 2)) for (a, b) in s.def_spans)
            dist_score = 1.0 / (1.0 + d)
        else:
            dist_score = 0.0

        # label overlap scores
        label_tokens = set(_ascii_tokens(s.definition))
        overlap = len(label_tokens & ctx_tokens) / max(1, len(label_tokens)) if label_tokens else 0.0

        out[s.sense_id] = dist_weight * dist_score + overlap_weight * overlap

    return out


def disambiguate_occurrences(
    text: str,
    occurrences: list[OccurrenceLite],
    senses: dict[str, list["AcronymSense"]],
    *,
    window_chars: int = 300,
    sense_prior_weight: float = 0.02,
    margin_threshold: float = 0.10,
    dist_weight: float = 0.75,
    overlap_weight: float = 0.25,
    senses_by_id: dict[str, "AcronymSense"] | None = None,
) -> list[OccurrenceResolution]:
    """
    Resolve acronym occurrences to the most likely sense using two signals:
    (1) proximity to the nearest definition span center and (2) label–context
    token overlap. A probabilistic margin and spatial tiebreak decide close calls.

    Scoring:
        distance = 1 / (1 + d), where d = min distance from occurrence start
                   to any definition-span center for that sense (0 if no spans).
        overlap  = |label_tokens ∩ ctx_tokens| / max(1, |label_tokens|).
        score    = dist_weight * distance + overlap_weight * overlap.

    Tiebreak:
        If the relative margin between top two scores
        ((p1 - p2) / max(p1, 1e-9)) ≥ margin_threshold → choose top.
        Otherwise, compare occurrence center to each sense’s def-spans using
        _min_distance_to_spans; if one is ≥3 units closer, choose it; else None.

    Args:
        text: Full source text.
        occurrences: OccurrenceLite list (acronym, start, end).
        senses: Mapping from UPPER(acronym) to candidate AcronymSense list.
        window_chars: Half-window (chars) around each occurrence to form ctx_tokens.
        sense_prior_weight: Maximum weight for an optional confidence prior.
            The prior is applied only for near-ties: a dynamic weight `w` is derived from
            the top-two base-score gap, then each sense score is nudged by
            `score += w * sense_confidence`. Set to 0.0 to disable.
        margin_threshold: Relative margin needed to accept the probabilistic winner.
        dist_weight: Weight for distance score.
        overlap_weight: Weight for overlap score.
        senses_by_id: Optional {sense_id → AcronymSense}; if None, built from `senses`.

    Returns:
        List of OccurrenceResolution in the same order as `occurrences`. Each item
        includes chosen_sense_id (or None), per-sense scores, and the top-two margin.

    Notes:
        - Tokenization uses `_tokens` (ASCII letters/digits leading; allows `'` and `-`).
        - Distance uses span centers; near-tie uses occurrence center vs span centers.
        - If an occurrence’s acronym is absent from `senses`, returns an ambiguous result.
    """
    results: list[OccurrenceResolution] = []
    senses_by_id = senses_by_id or {s.sense_id: s for lst in senses.values() for s in lst}

    for occ in occurrences:
        sense_list = senses.get(occ.acronym.upper(), [])
        if not sense_list:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0))
            continue

        L = max(0, occ.start - window_chars)
        R = min(len(text), occ.end + window_chars)
        ctx_tokens = set(_ascii_tokens(text[L:R]))

        base_scores = base_scores_for_occurrence(
            occ=occ,
            sense_list=sense_list,
            ctx_tokens=ctx_tokens,
            dist_weight=dist_weight,
            overlap_weight=overlap_weight,
        )

        if not base_scores:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0))
            continue

        w = dynamic_prior_weight(base_scores, max_w=sense_prior_weight, engage_gap=NEAR_TIE_GAP)

        cand_scores: dict[str, float] = dict(base_scores)
        if w:
            for sid in cand_scores:
                sc = getattr(senses_by_id.get(sid), "sense_confidence", 0.0)
                cand_scores[sid] += sense_prior_term(sense_confidence=sc, weight=w)

        chosen, margin = choose_with_tiebreak(
            occ, cand_scores, senses_by_id,
            margin_threshold=margin_threshold,
            near_tie_margin=NEAR_TIE_GAP
        )
        results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, chosen, cand_scores, margin))

    return results
