import re

from ..common.types import OccurrenceLite, OccurrenceResolution


def _tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", s.lower())


def _center(s: int, e: int) -> float:
    return (s + e) / 2.0


def _min_distance_to_spans(pos: float, spans: list[tuple[int, int]]) -> int:
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
            if label_tokens:
                overlap = len(label_tokens & ctx_tokens) / max(1, len(label_tokens))
            else:
                overlap = 0.0

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
