import re

from ..common.types import OccurrenceLite, OccurrenceResolution, AcronymSense


def _tokens(s: str) -> list[str]:
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
    """
    Returns the center point of the given interval.
    Args:
        s (int): the interval start
        e (int): the interval end
    Returns:
        float: the center point
    """
    return (s + e) / 2.0


def _min_distance_to_spans(pos: float, spans: list[tuple[int, int]]) -> int:
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
    best = 10**9
    for s, e in spans:
        c = _center(s, e)
        d = abs(c - pos)
        if d < best:
            best = int(d)
    return best


def choose_with_tiebreak(
    occ, cand_probs, senses_by_id, *, margin_threshold: float = 0.10, near_tie_margin: float = 0.06
) -> tuple[str | None, float]:
    """
        Pick a winning sense from candidate probabilities, using a relative-margin
        rule first and (if needed) a spatial tiebreak based on definition-span
        proximity.

        Process:
          1) Sort candidates by score desc (from `cand_probs`).
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
            cand_probs (dict[str, float]):
                Candidate scores in [0,1] keyed by `sense_id`. Typically derived from
                distance/overlap scoring.
            senses_by_id (dict[str, AcronymSense]):
                Lookup for senses by `sense_id`. Each sense may provide `def_spans`
                (list[tuple[int, int]]) used for distance tiebreaking.
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
    items = sorted(cand_probs.items(), key=lambda kv: kv[1], reverse=True)
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
        if d2 + 2 < d1:
            return sid2, margin
        if d1 + 2 < d2:
            return sid1, margin

    return None, margin


def disambiguate_occurrences(
    text: str,
    occurrences: list[OccurrenceLite],
    senses: dict[str, list["AcronymSense"]],
    *,
    window_chars: int = 300,
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
        cand_scores: dict[str, float] = {}
        sense_list = senses.get(occ.acronym.upper(), [])
        if not sense_list:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0))
            continue

        L = max(0, occ.start - window_chars)
        R = min(len(text), occ.end + window_chars)
        ctx_tokens = set(_tokens(text[L:R]))

        for s in sense_list:
            # 1) distance score to nearest def span
            if s.def_spans:
                # use center of span
                dists = [abs(occ.start - ((a + b) // 2)) for (a, b) in s.def_spans]
                d = min(dists)
                dist_score = 1.0 / (1.0 + d)  # 0..1, sharply favors nearby
            else:
                dist_score = 0.0

            # 2) label overlap
            label_tokens = set(_tokens(s.definition))
            overlap = len(label_tokens & ctx_tokens) / max(1, len(label_tokens)) if label_tokens else 0.0

            score = dist_weight * dist_score + overlap_weight * overlap
            cand_scores[s.sense_id] = score

        if not cand_scores:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0))
            continue

        chosen, margin = choose_with_tiebreak(
            occ, cand_scores, senses_by_id, margin_threshold=margin_threshold, near_tie_margin=0.06
        )
        results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, chosen, cand_scores, margin))

    return results
