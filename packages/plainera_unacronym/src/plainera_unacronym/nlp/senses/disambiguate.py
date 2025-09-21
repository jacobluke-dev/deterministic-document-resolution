import re
from typing import Dict, List

from plainera_unacronym.nlp.common.types import OccurrenceLite, AcronymSense, OccurrenceResolution


def _tok(s: str) -> set[str]:
    return set(t for t in re.findall(r"[A-Za-z0-9']+", s.lower()) if len(t) > 1)

def _sim(context: str, label: str) -> float:
    # token overlap + substring bonus
    a, b = _tok(context), _tok(label)
    if not a or not b:
        base = 0.0
    else:
        base = len(a & b) / len(b)
    bonus = 0.2 if label.lower() in context.lower() else 0.0
    return min(1.0, base + bonus)


_word = re.compile(r"[A-Za-z0-9'’]+")

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _word.findall(s)]

def disambiguate_occurrences(
    text: str,
    occurrences: List[OccurrenceLite],
    senses: Dict[str, List["AcronymSense"]],  # your class with .sense_id, .definition, .def_spans
    *,
    window_chars: int = 300,
    margin_threshold: float = 0.2,
    dist_weight: float = 0.75,
    overlap_weight: float = 0.25,
) -> List[OccurrenceResolution]:
    results: List[OccurrenceResolution] = []

    for occ in occurrences:
        cand_scores: Dict[str, float] = {}
        sense_list = senses.get(occ.acronym.upper(), [])
        if not sense_list:
            results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, None, {}, 0.0))
            continue

        # Context window
        L = max(0, occ.start - window_chars)
        R = min(len(text), occ.end + window_chars)
        ctx_tokens = set(_tokens(text[L:R]))

        for s in sense_list:
            # 1) distance score to nearest def span
            if s.def_spans:
                # use center of span
                dists = [abs(occ.start - ((a + b) // 2)) for (a, b) in s.def_spans]
                d = min(dists)
                dist_score = 1.0 / (1.0 + d)         # 0..1, sharply favors nearby
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

        # pick + margin
        best_id = max(cand_scores, key=cand_scores.get)
        sorted_vals = sorted(cand_scores.values(), reverse=True)
        if len(sorted_vals) == 1:
            margin = 1.0
        else:
            margin = (sorted_vals[0] - sorted_vals[1]) / (sorted_vals[0] + 1e-9)

        chosen = best_id if margin >= margin_threshold else None
        results.append(OccurrenceResolution(occ.acronym, occ.start, occ.end, chosen, cand_scores, margin))

    return results
