"""
Integration tests for acronym sense disambiguation (distance + token overlap + near-tie tiebreak).

Example: senses for "NLP" = Natural Language Processing vs Nice Lovely Plants; pick the sense whose label
overlaps the local context more, and/or whose definition span is nearer to the occurrence.
Numbers: occurrence NLP at [100,103]; Sense A spans [(98,110)] (centre≈104) vs Sense B [(10,20)] (centre≈15).
If scores are close, the near-tie tiebreak chooses the sense ≥3 chars closer; otherwise return None.
"""

import pytest
from plainera_unacronym.nlp.common.types import AcronymSense, OccurrenceLite, Span
from plainera_unacronym.nlp.extraction.senses.disambiguate import choose_with_tiebreak, disambiguate_occurrences


def S(acr: str, sid: str, definition: str, spans: list[Span]):
    return AcronymSense(acronym=acr, definition=definition, sense_id=sid, def_spans=spans, support=1)


def _get_attr_any(obj, names: list[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AssertionError(f"Could not find any of {names} on {type(obj)}. attrs={dir(obj)}")


def _chosen_id(res) -> str | None:
    return _get_attr_any(res, ["chosen_sense_id", "sense_id", "chosen", "resolved_sense_id"])


def _scores(res) -> dict[str, float]:
    return _get_attr_any(res, ["candidates", "scores", "cand_scores", "per_sense_scores"])


def _margin(res) -> float:
    return _get_attr_any(res, ["margin", "top2_margin", "relative_margin"])


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

        senses_by_id = {
            "s1": S("PDF", "s1", "Portable Document Format", [(0, 1)]),
            "s2": S("PDF", "s2", "Other", [(0, 1)]),
        }
        cand = {"s1": 0.80, "s2": 0.60}

        chosen, margin = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.10)
        assert chosen == "s1"
        assert margin == pytest.approx((0.80 - 0.60) / 0.80, rel=0, abs=1e-9)

    def test_returns_none_when_margin_low_and_not_near_tie(self):
        occ = OccurrenceLite("PDF", 10, 13)
        senses_by_id = {
            "s1": S("PDF", "s1", "Portable Document Format", [(0, 1)]),
            "s2": S("PDF", "s2", "Personal Data File", [(0, 1)]),
        }
        cand = {"s1": 0.50, "s2": 0.43}  # diff=0.07 > 0.06 => no distance tiebreak

        chosen, margin = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.20, near_tie_margin=0.06)
        assert chosen is None
        assert margin == pytest.approx((0.50 - 0.43) / 0.50, rel=0, abs=1e-9)

    def test_near_tie_distance_tiebreak_picks_closer_when_advantage_ge_3(self):
        occ = OccurrenceLite("NLP", 100, 103)
        senses_by_id = {
            "near": S("NLP", "near", "Near Sense", [(100, 102)]),
            "far":  S("NLP", "far", "Far Sense",  [(50, 52)]),
        }
        cand = {"near": 0.50, "far": 0.47}  # diff=0.03 => engage distance tiebreak

        chosen, _ = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.10, near_tie_margin=0.06)
        assert chosen == "near"

    def test_near_tie_distance_tiebreak_returns_none_when_distances_too_close(self):
        occ = OccurrenceLite("NLP", 100, 103)
        senses_by_id = {
            "a": S("NLP", "a", "A", [(100, 102)]),
            "b": S("NLP", "b", "B", [(98, 100)]),
        }
        cand = {"a": 0.50, "b": 0.48}

        chosen, _ = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.10, near_tie_margin=0.06)
        assert chosen is None


# -------------------------------------------------------------------
# disambiguate_occurrences integration tests
# -------------------------------------------------------------------

class TestDisambiguateOccurrences:
    def test_unknown_acronym_returns_ambiguous_resolution(self):
        text = "Nothing to see here."
        occs = [OccurrenceLite("XYZ", 0, 3)]
        senses = {}

        out = disambiguate_occurrences(text, occs, senses)
        assert len(out) == 1
        assert _chosen_id(out[0]) is None
        assert _scores(out[0]) == {}
        assert _margin(out[0]) == 0.0

    def test_distance_signal_dominates_and_selects_nearest_span(self):
        text = "Portable Document Format (PDF) ... later mention PDF"
        occs = [OccurrenceLite("PDF", 40, 43)]

        s_near = S("PDF", "near", "Portable Document Format", [(40, 43)])
        s_far  = S("PDF", "far",  "Personal Data File",      [(0, 3)])
        senses = {"PDF": [s_near, s_far]}

        out = disambiguate_occurrences(text, occs, senses, margin_threshold=0.10)
        assert len(out) == 1
        assert _chosen_id(out[0]) == "near"
        assert set(_scores(out[0]).keys()) == {"near", "far"}

    def test_overlap_signal_selects_best_label_when_no_spans(self):
        text = "We discuss natural language processing and later mention NLP again."
        occs = [OccurrenceLite("NLP", text.index("NLP"), text.index("NLP") + 3)]

        s_good = S("NLP", "s1", "natural language processing", [])
        s_bad  = S("NLP", "s2", "nice lovely plants", [])
        senses = {"NLP": [s_good, s_bad]}

        out = disambiguate_occurrences(text, occs, senses, window_chars=50, margin_threshold=0.10)
        assert len(out) == 1
        assert _chosen_id(out[0]) == "s1"

    def test_returns_none_when_scores_close_and_distance_tiebreak_not_decisive(self):
        text = "Alpha beta gamma NLP delta epsilon."
        occ_pos = text.index("NLP")
        occs = [OccurrenceLite("NLP", occ_pos, occ_pos + 3)]

        s1 = S("NLP", "a", "alpha beta",  [(occ_pos - 2, occ_pos + 2)])
        s2 = S("NLP", "b", "gamma delta", [(occ_pos - 3, occ_pos + 1)])
        senses = {"NLP": [s1, s2]}

        out = disambiguate_occurrences(text, occs, senses, window_chars=30, margin_threshold=0.50)
        assert len(out) == 1
        assert _chosen_id(out[0]) is None
        assert set(_scores(out[0]).keys()) == {"a", "b"}


    def test_multiple_occurrences_switch_between_senses_and_can_be_ambiguous(self):
        s1 = "The European Medicines Agency (EMA) set guidance. "
        s2 = "Later, Emergency Management Australia (EMA) responded. "
        text = s1 + s2

        # Definition spans
        span_med = (text.index("European"), text.index("European") + len("European Medicines Agency"))
        span_emg = (text.index("Emergency"), text.index("Australia") + len("Australia"))

        c1 = (span_med[0] + span_med[1]) // 2
        c2 = (span_emg[0] + span_emg[1]) // 2
        mid = (c1 + c2) // 2

        # Insert a neutral " EMA " exactly at midpoint to force equal-distance tie.
        text = text[:mid] + " EMA " + text[mid:]

        # Recompute spans after insertion (span_med is before mid; span_emg is after mid)
        if span_emg[0] >= mid:
            shift = len(" EMA ")
            span_emg = (span_emg[0] + shift, span_emg[1] + shift)

        span_med = span_med  # unchanged (insertion at/after midpoint)

        # Occurrences: near first def, near second def, and the midpoint EMA (ambiguous)
        occ1_i = text.index("(EMA)") + 1
        occ2_i = text.index("Australia (EMA)") + len("Australia (")
        occ3_i = text.index(" EMA ") + 1  # points at 'E' of inserted EMA

        occs = [
            OccurrenceLite("EMA", occ1_i, occ1_i + 3),
            OccurrenceLite("EMA", occ2_i, occ2_i + 3),
            OccurrenceLite("EMA", occ3_i, occ3_i + 3),
        ]

        senses = {
            "EMA": [
                AcronymSense("EMA", "European Medicines Agency", "ema|medicines", [span_med], 2),
                AcronymSense("EMA", "Emergency Management Australia", "ema|emergency", [span_emg], 2),
            ]
        }

        out = disambiguate_occurrences(text, occs, senses, dist_weight=1.0, overlap_weight=0.0,
                                       margin_threshold=0.10)

        assert len(out) == 3
        assert out[0].chosen_sense_id == "ema|medicines"
        assert out[1].chosen_sense_id == "ema|emergency"

        # The *point*: midpoint occurrence is equidistant => no decisive advantage => None.
        assert out[2].chosen_sense_id is None
        assert set(out[2].candidates.keys()) == {"ema|medicines", "ema|emergency"}
        assert 0.0 <= out[2].margin < 0.10

    def test_order_independence_of_senses_and_robust_scoring(self):
        text = "Org A (ABC) meets Org B (ABC). Later ABC again."
        i1 = text.index("(ABC)") + 1
        i2 = text.index("B (ABC)") + 3
        i3 = text.rindex("ABC")
        occs = [OccurrenceLite("ABC", i1, i1 + 3), OccurrenceLite("ABC", i2, i2 + 3), OccurrenceLite("ABC", i3, i3 + 3)]
        senses_A_first = {
            "ABC": [
                AcronymSense("ABC",
                             "Alpha Beta Council",
                             "abc|alpha_beta_council",
                             [(text.index("Org A"),
                               text.index("Org A") + 5)],
                             1),
                AcronymSense("ABC",
                             "Applied Business Consortium",
                             "abc|applied_business_consortium",
                             [(text.index("Org B"),
                               text.index("Org B") + 5)],
                             1),
            ]
        }
        senses_B_first = {"ABC": list(reversed(senses_A_first["ABC"]))}

        out1 = disambiguate_occurrences(text, occs, senses_A_first)
        out2 = disambiguate_occurrences(text, occs, senses_B_first)

        # Results should be identical even if senses list order is reversed
        assert [r.chosen_sense_id for r in out1] == [r.chosen_sense_id for r in out2]

    def test_near_tie_distance_tiebreak_with_bias_window(self):
        text = "ACR ... many dots ..."
        # Put the occurrence far from both spans to shrink scores and margin.
        occs = [OccurrenceLite("ACR", start=0, end=2)]  # pos ~0

        # Centers at ~50 and ~53 → d1=50, d2=53 (diff=3 > 2 bias)
        senses = {
            "ACR": [
                AcronymSense("ACR", "Alpha Core Reader", "acr|alpha", [(49, 51)], 1),  # center ~50
                AcronymSense("ACR", "Advanced Cardiac Rehab", "acr|cardiac", [(52, 54)], 1),  # center ~53
            ]
        }

        out = disambiguate_occurrences(
            text, occs, senses, dist_weight=1.0, overlap_weight=0.0, margin_threshold=0.10
        )
        # Near tie on probs, distance decides; alpha is nearer (50 vs 53)
        assert out[0].chosen_sense_id == "acr|alpha"

        # And now the margin should indeed be small (< 0.10)
        assert 0.0 <= out[0].margin < 0.10

    #TODO coming later span normalisation isn't currently supported
    # def test_handles_reversed_span_endpoints(self):
    #     text = "Zeta Corp (ZC) operates globally. Later, ZC is referenced again."
    #     occs = [
    #         OccurrenceLite("ZC", text.index("(ZC)") + 1, text.index("(ZC)") + 3),
    #         OccurrenceLite("ZC", text.rindex("ZC"), text.rindex("ZC") + 2),
    #     ]
    #     senses = {
    #         "ZC": [
    #             AcronymSense("ZC", "Zeta Corporation",
    #             "zc|zeta_corporation", [(text.index("Zeta"), text.index("Zeta") + 4)], 1),
    #             AcronymSense("ZC", "Zero Cool",
    #             "zc|zero_cool",
    #             [(text.index("globally") + 2, text.index("globally") - 2)], 1),  # reversed endpoints on purpose
    #         ]
    #     }
    #     out = disambiguate_occurrences(text, occs, senses, dist_weight=1.0, overlap_weight=0.0)
    #     # Should not crash; should still choose the appropriate nearest
    #     assert len(out) == 2
    #     assert out[0].chosen_sense_id == "zc|zeta_corporation"
    #     assert out[1].chosen_sense_id in {"zc|zeta_corporation", "zc|zero_cool"}
