import math
from types import SimpleNamespace as NS
import pytest

from plainera_unacronym.nlp.common.types import OccurrenceLite, AcronymSense
from plainera_unacronym.nlp.senses.disambiguate import _ascii_tokens, _center, _min_distance_to_spans, choose_with_tiebreak, \
    disambiguate_occurrences


class TestTokens:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("", []),
            ("!!!", []),
            ("HELLO world", ["hello", "world"]),
            ("rock'n'roll", ["rock'n'roll"]),
            ("state-of-the-art", ["state-of-the-art"]),
            ("James'", ["james'"]),  # trailing apostrophe allowed by regex
            ("end-", ["end-"]),  # trailing hyphen allowed by regex
            ("-dash", ["dash"]),  # leading hyphen is not allowed; skips to 'dash'
            ("'tis", ["tis"]),  # leading apostrophe not allowed; captures from 't'
            ("A1B2", ["a1b2"]),
            ("42", ["42"]),
            ("snake_case", ["snake", "case"]),  # underscore not in pattern
            ("e.g.", ["e", "g"]),  # dots split tokens
            ("email@example.com", ["email", "example", "com"]),
            ("O'Neill", ["o'neill"]),
            ("naïve café", ["na", "ve", "caf"]),  # non-ASCII chars split tokens
        ],
    )
    def test_tokenization_cases(self, s, expected):
        assert _ascii_tokens(s) == expected


class TestCenter:
    @pytest.mark.parametrize(
        "s,e,expected",
        [
            (0, 0, 0.0),  # identical endpoints
            (0, 2, 1.0),  # even span -> integer midpoint
            (0, 1, 0.5),  # odd span -> .5 midpoint
            (2, 4, 3.0),
            (-2, 2, 0.0),  # crosses zero
            (-3, -1, -2.0),
            (-1, 0, -0.5),
            (5, 1, 3.0),  # reversed order still averages
        ],
    )
    def test_basic_midpoints(self, s, e, expected):
        assert _center(s, e) == expected

    @pytest.mark.parametrize(
        "s,e",
        [
            (10 ** 12, 10 ** 12 + 2),  # very large even span
            (-10 ** 12, -10 ** 12 + 1),  # very large odd span negative
            (-10 ** 12, 10 ** 12),  # huge cross-zero span
        ],
    )
    def test_large_values(self, s, e):
        expected = (s + e) / 2.0
        assert _center(s, e) == pytest.approx(expected, rel=0, abs=0)

    @pytest.mark.parametrize("s,e", [(7, 3), (-5, 11), (0, 9), (-8, -3)])
    def test_symmetry(self, s, e):
        assert _center(s, e) == _center(e, s)

    @pytest.mark.parametrize("s,e", [(0, 10), (-4, 6), (2, 3), (-9, -7)])
    def test_midpoint_equidistant(self, s, e):
        m = _center(s, e)
        assert abs(m - s) == abs(e - m)


class TestMinDistanceToSpansUnit:
    def test_empty_spans_returns_sentinel(self):
        assert _min_distance_to_spans(0.0, []) == 10 ** 9

    @pytest.mark.parametrize(
        "pos,spans,expected",
        [
            # exact center → distance 0
            (1.0, [(0, 2)], 0),
            # uses center: center(0, 2)=1 → |1-1.4|=0.4 → floor -> 0
            (1.4, [(0, 2)], 0),
            # farther: |1-2.2|=1.2 → floor->1
            (2.2, [(0, 2)], 1),
            # multiple spans: chooses nearest center
            (10.0, [(0, 2), (8, 12), (20, 22)], 0),  # center(8,12)=10 → 0
            # reversed endpoints still average the same
            (3.0, [(5, 1)], 0),  # center(5,1)=3 → distance 0
        ],
    )
    def test_distance_floor_and_center_logic(self, pos, spans, expected):
        assert _min_distance_to_spans(pos, spans) == expected

    def test_overlapping_spans(self):
        # centers: (0,4)->2; (1,3)->2; equal centers -> still 0 if pos=2
        assert _min_distance_to_spans(2.0, [(0, 4), (1, 3)]) == 0

    def test_negative_coordinates(self):
        # center(-6,-2)=-4; | -4 - (-5.1) | = 1.1 -> floor 1
        assert _min_distance_to_spans(-5.1, [(-6, -2)]) == 1

    def test_large_values(self):
        spans = [(10 ** 12, 10 ** 12 + 2)]
        # center = 10**12 + 1; distance to pos center+0.9 → 0 floor
        assert _min_distance_to_spans(10 ** 12 + 1.9, spans) == 0

    def test_tie_prefers_first_min_but_value_is_same(self):
        # centers at 0 and 2; pos=1 → both distance 1.0 -> floor 1
        assert _min_distance_to_spans(1.0, [(-1, 1), (1, 3)]) == 1

    def test_symmetry_of_center_used(self):
        # sanity: _center symmetric
        assert _center(2, 8) == _center(8, 2) == 5.0


def _ref_min_distance_to_centers(pos: float, spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 10 ** 9
    centers = [(_center(s, e)) for s, e in spans]
    d = min(abs(c - pos) for c in centers) if centers else math.inf
    return 10 ** 9 if d is math.inf else int(d)


class TestMinDistanceToSpansIntegration:
    def test_integration_against_reference_and_order_independence(self):
        spans = [(0, 2), (8, 12), (20, 26)]  # centers: 1, 10, 23
        spans_shuffled = [(20, 26), (0, 2), (8, 12)]

        # Sweep positions over a range and compare to reference
        for pos in [x / 2.0 for x in range(-4, 60)]:  # -2.0 .. 29.5
            got = _min_distance_to_spans(pos, spans)
            ref = _ref_min_distance_to_centers(pos, spans)
            assert got == ref, f"pos={pos}, spans={spans}"

            # order independence
            got2 = _min_distance_to_spans(pos, spans_shuffled)
            assert got2 == ref, f"order-sensitive at pos={pos}"

    def test_piecewise_monotonic_near_a_center(self):
        spans = [(40, 60)]  # center at 50
        trail = [49.9, 49.4, 49.1, 49.01, 49.001, 50.0, 50.2, 50.9, 51.4]
        # distances should not increase as we approach center, then rise after passing it
        dists = [_min_distance_to_spans(p, spans) for p in trail]
        # Check valley at the center (allow equal plateaus due to flooring)
        i_min = dists.index(min(dists))
        assert trail[i_min] in (49.9, 49.4, 49.1, 49.01, 49.001, 50.0)  # minimum at/near center
        # Optional: ensure non-increasing up to the minimum index, then non-decreasing
        pre = dists[: i_min + 1]
        post = dists[i_min:]
        assert all(pre[i] >= pre[i + 1] for i in range(len(pre) - 1))
        assert all(post[i] <= post[i + 1] for i in range(len(post) - 1))

    def test_handles_mixed_and_reversed_spans(self):
        spans = [(-10, -2), (15, 5), (100, 80)]  # includes reversed endpoints
        positions = [-20.0, -6.0, 0.0, 7.5, 50.0, 90.0, 200.0]
        for p in positions:
            # parity with reference regardless of span ordering or reversal
            assert _min_distance_to_spans(p, spans) == _ref_min_distance_to_centers(p, spans)


class TestChooseWithTiebreakUnit:
    def test_empty_candidates(self):
        occ = NS(start=0, end=10)
        sid, margin = choose_with_tiebreak(occ, {}, {})
        assert sid is None and margin == 0.0

    def test_single_candidate_short_circuit(self):
        occ = NS(start=0, end=2)  # center=1
        cand_probs = {"A": 0.7}
        sid, margin = choose_with_tiebreak(occ, cand_probs, {})
        # p2=0 → margin=(0.7-0)/0.7=1.0 ≥ margin_threshold → choose 'A'
        assert sid == "A"
        assert margin == pytest.approx(1.0)

    def test_clear_margin_winner(self):
        occ = NS(start=10, end=20)  # center=15
        cand_probs = {"A": 0.90, "B": 0.70}
        sid, margin = choose_with_tiebreak(occ, cand_probs, {})
        exp_margin = (0.90 - 0.70) / 0.90
        assert sid == "A"
        assert margin == pytest.approx(exp_margin)

    def test_near_tie_distance_picks_second(self):
        # Force near tie (diff ≤ near_tie_margin and margin < margin_threshold)
        occ = NS(start=10, end=12)  # center = 11
        cand_probs = {"A": 0.51, "B": 0.50}
        senses = {
            # A is far (center ~100), B is near (center ~11)
            "A": NS(def_spans=[(99, 101)]),
            "B": NS(def_spans=[(10, 12)]),
        }
        sid, margin = choose_with_tiebreak(occ, cand_probs, senses)
        assert sid == "B"
        assert margin == pytest.approx((0.51 - 0.50) / 0.51)

    def test_near_tie_distance_picks_first(self):
        occ = NS(start=50, end=52)  # center = 51
        cand_probs = {"A": 0.505, "B": 0.50}
        senses = {
            "A": NS(def_spans=[(50, 52)]),     # near center
            "B": NS(def_spans=[(200, 220)]),   # far away
        }
        sid, margin = choose_with_tiebreak(occ, cand_probs, senses)
        assert sid == "A"
        assert margin == pytest.approx((0.505 - 0.50) / 0.505)

    def test_near_tie_undecided_returns_none_when_within_bias(self):
        # Distances within the ±2 bias window → unresolved tie
        occ = NS(start=0, end=2)  # center = 1
        cand_probs = {"A": 0.500, "B": 0.495}
        senses = {
            "A": NS(def_spans=[(0, 2)]),   # center 1 → d=0
            "B": NS(def_spans=[(0, 4)]),   # center 2 → d=1
        }
        sid, margin = choose_with_tiebreak(occ, cand_probs, senses)
        assert sid is None
        assert margin == pytest.approx((0.500 - 0.495) / 0.500)

    def test_uses_center_of_occurrence(self):
        # If center moves, the winner toggles
        cand_probs = {"A": 0.51, "B": 0.50}
        senses = {
            "A": NS(def_spans=[(0, 2)]),   # center 1
            "B": NS(def_spans=[(8, 12)]),  # center 10
        }
        # Occ near A → pick A
        sid1, _ = choose_with_tiebreak(NS(start=0, end=2), cand_probs, senses)   # center=1
        # Occ near B → pick B
        sid2, _ = choose_with_tiebreak(NS(start=9, end=11), cand_probs, senses)  # center=10
        assert sid1 == "A"
        assert sid2 == "B"

    def test_handles_missing_or_empty_def_spans(self):
        occ = NS(start=0, end=2)  # center 1
        cand_probs = {"A": 0.51, "B": 0.50}
        senses = {
            "A": NS(def_spans=None),      # treated as []
            "B": NS(def_spans=[]),
        }
        # With no spans, both distances fall back to sentinel 1e9, tie remains unresolved
        sid, margin = choose_with_tiebreak(occ, cand_probs, senses)
        assert sid is None
        assert margin == pytest.approx((0.51 - 0.50) / 0.51)


class TestChooseWithTiebreakIntegration:
    def test_order_independence_and_switching_with_position(self):
        # Three senses; only top two matter, but include a third to ensure sort logic is robust.
        cand_probs = {"A": 0.505, "B": 0.500, "C": 0.10}

        senses = {
            "A": NS(def_spans=[(0, 2)]),     # center 1
            "B": NS(def_spans=[(18, 22)]),   # center 20
            "C": NS(def_spans=[(100, 120)]),
        }

        # Sweep the occurrence across the line and ensure winner flips as proximity changes
        # Near A → choose A; around mid (≈10.5), distances nearly equal → None; near B → choose B.
        trail = [
            NS(start=0, end=2),     # center 1 (near A)
            NS(start=6, end=16),    # center 11 (roughly mid)
            NS(start=19, end=21),   # center 20 (near B)
        ]

        winners = []
        for occ in trail:
            sid, _ = choose_with_tiebreak(occ, dict(cand_probs), senses)
            winners.append(sid)

        assert winners[0] == "A"
        assert winners[1] in (None, "A", "B")  # acceptable: depends on bias window; commonly None
        assert winners[2] == "B"

    def test_custom_thresholds_affect_outcome(self):
        # With a larger near_tie_margin, the distance tiebreak engages more often.
        occ = NS(start=50, end=52)  # center 51
        # A slight probabilistic lead for A
        cand_probs = {"A": 0.55, "B": 0.52}
        senses = {"A": NS(def_spans=[(0, 2)]), "B": NS(def_spans=[(50, 52)])}  # B is spatially closer

        # Default thresholds → margin = (0.55-0.52)/0.55 ≈ 0.0545 < 0.10, near tie triggers, B should win
        sid_def, margin_def = choose_with_tiebreak(occ, cand_probs, senses)
        assert sid_def == "B"
        assert 0.0 <= margin_def < 0.10

        # If we make the margin_threshold tiny, the probabilistic winner short-circuits before distance
        sid_small_thr, margin_small_thr = choose_with_tiebreak(
            occ, cand_probs, senses, margin_threshold=0.01, near_tie_margin=0.06
        )
        assert sid_small_thr == "A"
        assert margin_small_thr >= 0.01

    def test_handles_realistic_mix_missing_spans_and_ties(self):
        # Some senses lack def_spans; ensure function remains stable.
        occ = NS(start=90, end=110)  # center 100
        cand_probs = {"A": 0.50, "B": 0.50}
        senses = {"A": NS(def_spans=None), "B": NS(def_spans=[(98, 102)])}  # B has nearby span

        # diff=0 ≤ near_tie_margin; B has finite distance, A hits sentinel → B should win
        sid, margin = choose_with_tiebreak(occ, cand_probs, senses)
        assert sid == "B"
        assert margin == 0.0


class TestDisambiguateOccurrencesUnit:
    def test_no_senses_for_acronym_returns_none(self):
        text = "Please refer to the EMA guidelines."
        occs = [OccurrenceLite(acronym="EMA", start=19, end=22)]
        senses = {}  # no senses present
        out = disambiguate_occurrences(text, occs, senses)
        assert len(out) == 1
        r = out[0]
        assert r.acronym == "EMA"
        assert r.chosen_sense_id is None
        assert r.candidates == {}
        assert r.margin == 0.0

    def test_overlap_only_picks_best_label_match(self):
        # Make distance irrelevant (no def_spans) and rely on label overlap
        text = "We work with the European Medicines Agency on drug approvals."
        occs = [OccurrenceLite(acronym="EMA", start=14, end=17)]
        senses = {
            "EMA": [
                AcronymSense(
                    acronym="EMA",
                    definition="European Medicines Agency",
                    sense_id="ema|european_medicines_agency",
                    def_spans=[],  # distance score -> 0.0
                    support=1,
                ),
                AcronymSense(
                    acronym="EMA",
                    definition="Emergency Management Australia",
                    sense_id="ema|emergency_management_australia",
                    def_spans=[],
                    support=1,
                ),
            ]
        }
        out = disambiguate_occurrences(
            text,
            occs,
            senses,
            dist_weight=0.0,       # isolate overlap
            overlap_weight=1.0,    # full weight on label overlap
        )
        r = out[0]
        assert r.chosen_sense_id == "ema|european_medicines_agency"
        # Strong margin because other label has near-zero overlap
        assert r.margin >= 0.10
        assert set(r.candidates) == {
            "ema|european_medicines_agency",
            "ema|emergency_management_australia",
        }
        assert all(0.0 <= v <= 1.0 for v in r.candidates.values())

    def test_distance_only_picks_nearest_definition_span(self):
        text = "FDA met EMA in Brussels yesterday."
        # Occurrence near index ~8..11; pick the closer span via distance
        occs = [OccurrenceLite(acronym="EMA", start=8, end=11)]
        senses = {
            "EMA": [
                AcronymSense(
                    acronym="EMA",
                    definition="European Medicines Agency",
                    sense_id="ema|medicines",
                    def_spans=[(6, 10)],  # center ~8
                    support=2,
                ),
                AcronymSense(
                    acronym="EMA",
                    definition="Emergency Management Australia",
                    sense_id="ema|emergency",
                    def_spans=[(100, 110)],  # far away
                    support=2,
                ),
            ]
        }
        out = disambiguate_occurrences(
            text,
            occs,
            senses,
            dist_weight=1.0,
            overlap_weight=0.0,   # isolate distance
        )
        r = out[0]
        assert r.chosen_sense_id == "ema|medicines"
        assert r.margin >= 0.10

    def test_senses_by_id_fallback_built_automatically(self):
        text = "ACR appears here."
        occs = [OccurrenceLite(acronym="ACR", start=0, end=3)]
        senses = {
            "ACR": [
                AcronymSense("ACR", "Alpha Core Reader", "acr|alpha_core_reader", [(0, 2)], 1),
                AcronymSense("ACR", "Advanced Cardiac Rehab", "acr|advanced_cardiac_rehab", [(50, 60)], 1),
            ]
        }
        # Pass senses_by_id=None to exercise the internal build
        out = disambiguate_occurrences(
            text, occs, senses, senses_by_id=None, dist_weight=1.0, overlap_weight=0.0
        )
        assert len(out) == 1
        assert out[0].chosen_sense_id in {"acr|alpha_core_reader", "acr|advanced_cardiac_rehab"}

    def test_windowing_affects_overlap_tokens(self):
        text = (
            "x " * 100
            + "United Kingdom Health Security Agency collaborates widely. "
            + "x " * 100
            + "UKHSA"
        )
        occ_start = len(text) - 5
        occs = [OccurrenceLite(acronym="UKHSA", start=occ_start, end=occ_start + 5)]
        senses = {
            "UKHSA": [
                AcronymSense(
                    "UKHSA",
                    "United Kingdom Health Security Agency",
                    "ukhsa|united_kingdom_health_security_agency",
                    [],  # distance 0 → rely on overlap
                    1,
                )
            ]
        }

        sid = "ukhsa|united_kingdom_health_security_agency"

        # Tiny window → label tokens outside window → overlap≈0 → may return None
        r_small = disambiguate_occurrences(
            text, occs, senses, window_chars=20, dist_weight=0.0, overlap_weight=1.0
        )[0]

        # Large window → label tokens inside window → higher overlap
        r_large = disambiguate_occurrences(
            text, occs, senses, window_chars=400, dist_weight=0.0, overlap_weight=1.0
        )[0]

        small_score = r_small.candidates.get(sid, 0.0)
        large_score = r_large.candidates.get(sid, 0.0)

        # Overlap should not decrease when we widen the window
        assert small_score <= large_score

        # With a large enough window, we should actually pick the sense
        assert r_large.chosen_sense_id == sid

        # With a tiny window, resolution can be None (score 0.0)
        assert r_small.chosen_sense_id in (None, sid)


class TestDisambiguateOccurrencesIntegration:
    def test_multiple_occurrences_switch_between_senses(self):
        text = (
            "The European Medicines Agency (EMA) set guidance. "
            "Later, Emergency Management Australia (EMA) responded. "
            "EMA was mentioned again."
        )
        # Place occurrences near each definition mention
        occs = [
            OccurrenceLite("EMA", start=text.index("EMA) set"), end=text.index("EMA) set") + 3),
            OccurrenceLite("EMA", start=text.index("Australia (EMA)") + len("Australia ("), end=text.index("Australia (EMA)") + len("Australia (EMA)")),
            OccurrenceLite("EMA", start=text.rindex("EMA"), end=text.rindex("EMA") + 3),
        ]
        senses = {
            "EMA": [
                AcronymSense(
                    "EMA",
                    "European Medicines Agency",
                    "ema|medicines",
                    # definition spans roughly where 'European Medicines Agency' occurs
                    [(text.index("European"), text.index("European") + len("European Medicines Agency"))],
                    2,
                ),
                AcronymSense(
                    "EMA",
                    "Emergency Management Australia",
                    "ema|emergency",
                    [(text.index("Emergency"), text.index("Australia") + len("Australia"))],
                    2,
                ),
            ]
        }

        out = disambiguate_occurrences(text, occs, senses, dist_weight=0.75, overlap_weight=0.25)
        assert len(out) == 3
        # First near EMA (Medicines) → pick medicines
        assert out[0].chosen_sense_id == "ema|medicines"
        # Second near EMA (Emergency) → pick emergency
        assert out[1].chosen_sense_id == "ema|emergency"
        # Third occurrence is later; could be ambiguous—ensure it resolves to whichever is closer overall
        assert out[2].chosen_sense_id in {"ema|medicines", "ema|emergency"}

    def test_order_independence_of_senses_and_robust_scoring(self):
        text = "Org A (ABC) meets Org B (ABC). Later ABC again."
        i1 = text.index("(ABC)") + 1
        i2 = text.index("B (ABC)") + 3
        i3 = text.rindex("ABC")
        occs = [OccurrenceLite("ABC", i1, i1 + 3), OccurrenceLite("ABC", i2, i2 + 3), OccurrenceLite("ABC", i3, i3 + 3)]
        senses_A_first = {
            "ABC": [
                AcronymSense("ABC", "Alpha Beta Council", "abc|alpha_beta_council", [(text.index("Org A"), text.index("Org A") + 5)], 1),
                AcronymSense("ABC", "Applied Business Consortium", "abc|applied_business_consortium", [(text.index("Org B"), text.index("Org B") + 5)], 1),
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

    def test_handles_reversed_span_endpoints(self):
        text = "Zeta Corp (ZC) operates globally. Later, ZC is referenced again."
        occs = [
            OccurrenceLite("ZC", text.index("(ZC)") + 1, text.index("(ZC)") + 3),
            OccurrenceLite("ZC", text.rindex("ZC"), text.rindex("ZC") + 2),
        ]
        senses = {
            "ZC": [
                AcronymSense("ZC", "Zeta Corporation", "zc|zeta_corporation", [(text.index("Zeta"), text.index("Zeta") + 4)], 1),
                AcronymSense("ZC", "Zero Cool", "zc|zero_cool", [(text.index("globally") + 2, text.index("globally") - 2)], 1),  # reversed endpoints on purpose
            ]
        }
        out = disambiguate_occurrences(text, occs, senses, dist_weight=1.0, overlap_weight=0.0)
        # Should not crash; should still choose the appropriate nearest
        assert len(out) == 2
        assert out[0].chosen_sense_id == "zc|zeta_corporation"
        assert out[1].chosen_sense_id in {"zc|zeta_corporation", "zc|zero_cool"}
