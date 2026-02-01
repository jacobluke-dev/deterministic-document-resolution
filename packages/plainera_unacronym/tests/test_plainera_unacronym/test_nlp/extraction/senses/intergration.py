import pytest

from plainera_unacronym.nlp.common.types import OccurrenceLite, AcronymSense
from plainera_unacronym.nlp.extraction.senses.disambiguate import choose_with_tiebreak, disambiguate_occurrences


def _get_attr_any(obj, names: list[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AssertionError(f"Could not find any of {names} on {type(obj)}. attrs={dir(obj)}")


def _chosen_id(res) -> str | None:
    # Try common attribute names. If your class uses another name, add it here once.
    return _get_attr_any(res, ["chosen_sense_id", "sense_id", "chosen", "resolved_sense_id"])


def _scores(res) -> dict[str, float]:
    return _get_attr_any(res, ["scores", "cand_scores", "per_sense_scores"])


def _margin(res) -> float:
    return _get_attr_any(res, ["margin", "top2_margin", "relative_margin"])


def S(acr: str, sid: str, definition: str, spans: list[tuple[int, int]]):
    return AcronymSense(acronym=acr, definition=definition, sense_id=sid, def_spans=spans, support=1)


# -------------------------------------------------------------------
# choose_with_tiebreak integration tests
# -------------------------------------------------------------------

class TestChooseWithTiebreak:
    def test_returns_none_when_no_candidates(self):
        occ = OccurrenceLite("NLP", 10, 13)
        chosen, margin = choose_with_tiebreak(occ, {}, {}, margin_threshold=0.10, near_tie_margin=0.06)
        assert chosen is None
        assert margin == 0.0

    def test_accepts_probabilistic_winner_when_margin_exceeds_threshold(self):
        occ = OccurrenceLite("PDF", 10, 13)
        senses_by_id = {"s1": AcronymSense("s1", "Portable Document Format", def_spans=[(0, 1)])}
        cand = {"s1": 0.80, "s2": 0.60}
        chosen, margin = choose_with_tiebreak(occ, cand, senses_by_id | {"s2": AcronymSense("s2", "Other", [])}, margin_threshold=0.10)
        # margin = (0.8 - 0.6) / 0.8 = 0.25
        assert chosen == "s1"
        assert margin == pytest.approx(0.25, rel=0, abs=1e-9)

    def test_returns_none_when_margin_low_and_not_near_tie(self):
        occ = OccurrenceLite("PDF", 10, 13)
        senses_by_id = {
            "s1": AcronymSense("s1", "Portable Document Format", def_spans=[(0, 1)]),
            "s2": AcronymSense("s2", "Personal Data File", def_spans=[(0, 1)]),
        }
        # margin below threshold, but (p1 - p2) > near_tie_margin so no distance tiebreak
        cand = {"s1": 0.50, "s2": 0.43}  # diff=0.07 > 0.06 near-tie margin
        chosen, margin = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.20, near_tie_margin=0.06)
        assert chosen is None
        assert margin == pytest.approx((0.50 - 0.43) / 0.50, rel=0, abs=1e-9)

    def test_near_tie_distance_tiebreak_picks_closer_when_advantage_ge_3(self):
        # Make margin low and near-tie engaged; then force distance advantage >= 3
        occ = OccurrenceLite("NLP", 100, 103)  # center ~101.5 used by tiebreak
        senses_by_id = {
            "near": AcronymSense("near", "Near Sense", def_spans=[(100, 102)]),   # center ~101 -> distance tiny
            "far":  AcronymSense("far",  "Far Sense",  def_spans=[(50, 52)]),     # center ~51  -> distance ~50
        }
        cand = {"near": 0.50, "far": 0.47}  # diff=0.03 <= near_tie_margin, margin=0.06 < 0.10
        chosen, _ = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.10, near_tie_margin=0.06)
        assert chosen == "near"

    def test_near_tie_distance_tiebreak_returns_none_when_distances_too_close(self):
        occ = OccurrenceLite("NLP", 100, 103)
        senses_by_id = {
            "a": AcronymSense("a", "A", def_spans=[(100, 102)]),  # center ~101
            "b": AcronymSense("b", "B", def_spans=[(98, 100)]),   # center ~99
        }
        cand = {"a": 0.50, "b": 0.48}  # diff=0.02 near-tie
        chosen, _ = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.10, near_tie_margin=0.06)
        # distance advantage should be < 3, so ambiguous
        assert chosen is None


# -------------------------------------------------------------------
# disambiguate_occurrences integration tests
# -------------------------------------------------------------------

class TestDisambiguateOccurrences:
    def test_unknown_acronym_returns_ambiguous_resolution(self):
        text = "Nothing to see here."
        occs = [OccurrenceLite("XYZ", 0, 3)]
        senses = {}  # no senses for XYZ

        out = disambiguate_occurrences(text, occs, senses)
        assert len(out) == 1
        assert _chosen_id(out[0]) is None
        assert _scores(out[0]) == {}
        assert _margin(out[0]) == 0.0

    def test_distance_signal_dominates_and_selects_nearest_span(self):
        text = "Portable Document Format (PDF) ... later mention PDF"
        # occurrence at the later "PDF" (use a position that is much closer to one span)
        occs = [OccurrenceLite("PDF", 40, 43)]

        s_near = AcronymSense("near", "Portable Document Format", def_spans=[(40, 43)])  # center ~41
        s_far = AcronymSense("far", "Personal Data File", def_spans=[(0, 3)])           # center ~1

        senses = {"PDF": [s_near, s_far]}

        out = disambiguate_occurrences(text, occs, senses, margin_threshold=0.10)
        assert len(out) == 1
        assert _chosen_id(out[0]) == "near"
        assert set(_scores(out[0]).keys()) == {"near", "far"}
        assert _margin(out[0]) >= 0.10

    def test_overlap_signal_selects_best_label_when_no_spans(self):
        text = "We discuss natural language processing and later mention NLP again."
        occs = [OccurrenceLite("NLP", text.index("NLP"), text.index("NLP") + 3)]

        # No def_spans -> dist_score=0 for both; overlap decides.
        s_good = AcronymSense("s1", "natural language processing", def_spans=[])
        s_bad = AcronymSense("s2", "nice lovely plants", def_spans=[])

        senses = {"NLP": [s_good, s_bad]}

        out = disambiguate_occurrences(text, occs, senses, window_chars=50, margin_threshold=0.10)
        assert len(out) == 1
        assert _chosen_id(out[0]) == "s1"

    def test_returns_none_when_scores_close_and_distance_tiebreak_not_decisive(self):
        text = "Alpha beta gamma NLP delta epsilon."
        occ_pos = text.index("NLP")
        occs = [OccurrenceLite("NLP", occ_pos, occ_pos + 3)]

        # Spans extremely close so distance tiebreak can't get >=3 advantage
        s1 = AcronymSense("a", "alpha beta", def_spans=[(occ_pos - 2, occ_pos + 2)])
        s2 = AcronymSense("b", "gamma delta", def_spans=[(occ_pos - 3, occ_pos + 1)])

        senses = {"NLP": [s1, s2]}

        out = disambiguate_occurrences(
            text,
            occs,
            senses,
            window_chars=30,
            margin_threshold=0.50,  # make it hard to accept probabilistic winner
        )
        assert len(out) == 1
        assert _chosen_id(out[0]) is None
        assert set(_scores(out[0]).keys()) == {"a", "b"}
        assert 0.0 <= _margin(out[0]) < 0.50
