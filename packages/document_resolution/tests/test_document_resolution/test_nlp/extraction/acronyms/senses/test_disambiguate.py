import math
from types import SimpleNamespace as NS

import pytest
from document_resolution.nlp.common.types import AcronymMeaning, OccurrenceLite, Span
from document_resolution.nlp.extraction.acronyms.meanings.disambiguate import (
    _ascii_tokens,
    _center,
    _min_distance_to_spans,
    choose_with_tiebreak,
    disambiguate_occurrences,
)


class TestAsciiTokens:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("", []),
            ("!!!", []),
            ("HELLO world", ["hello", "world"]),
            ("rock'n'roll", ["rock'n'roll"]),
            ("state-of-the-art", ["state-of-the-art"]),
            ("James'", ["james'"]),  # trailing apostrophe kept
            ("end-", ["end-"]),  # trailing hyphen kept
            ("-dash", ["dash"]),  # leading hyphen not allowed
            ("'tis", ["tis"]),  # leading apostrophe not allowed
            ("A1B2", ["a1b2"]),
            ("42", ["42"]),
            ("snake_case", ["snake", "case"]),  # underscore splits
            ("e.g.", ["e", "g"]),  # dot splits
            ("email@example.com", ["email", "example", "com"]),
            ("O'Neill", ["o'neill"]),
            ("naïve café", ["na", "ve", "caf"]),  # non-ASCII chars split tokens
        ],
    )
    def test_tokenization_cases(self, s, expected):
        assert _ascii_tokens(s) == expected

    def test_does_not_merge_across_whitespace_or_punct(self):
        assert _ascii_tokens("a--b") == ["a--b"]  # hyphens allowed internally
        assert _ascii_tokens("a - b") == ["a", "b"]  # spaces split
        assert _ascii_tokens("a,b;c") == ["a", "b", "c"]  # punctuation splits


class TestCenter:
    @pytest.mark.parametrize(
        "s, e, expected",
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
            (10**12, 10**12 + 2),  # very large even span
            (-(10**12), -(10**12) + 1),  # very large odd span negative
            (-(10**12), 10**12),  # huge cross-zero span
        ],
    )
    def test_large_values(self, s, e):
        expected = (s + e) / 2.0
        assert _center(s, e) == pytest.approx(expected, rel=0, abs=0)

    @pytest.mark.parametrize("s,e", [(7, 3), (-5, 11), (0, 9), (-8, -3)])
    def test_symmetry(self, s, e):
        assert _center(s, e) == _center(e, s)

    def test_large_values_stable(self):
        s, e = 10**12, 10**12 + 2
        assert _center(s, e) == pytest.approx((s + e) / 2.0, rel=0, abs=0)

    @pytest.mark.parametrize("s,e", [(0, 10), (-4, 6), (2, 3), (-9, -7)])
    def test_midpoint_equidistant(self, s, e):
        m = _center(s, e)
        assert abs(m - s) == abs(e - m)


class TestMinDistanceToSpansUnit:
    def test_empty_spans_returns_sentinel(self):
        assert _min_distance_to_spans(0.0, []) == 10**9

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
        spans = [(10**12, 10**12 + 2)]
        # center = 10**12 + 1; distance to pos center+0.9 → 0 floor
        assert _min_distance_to_spans(10**12 + 1.9, spans) == 0

    def test_tie_prefers_first_min_but_value_is_same(self):
        # centers at 0 and 2; pos=1 → both distance 1.0 -> floor 1
        assert _min_distance_to_spans(1.0, [(-1, 1), (1, 3)]) == 1

    def test_symmetry_of_center_used(self):
        # sanity: _center symmetric
        assert _center(2, 8) == _center(8, 2) == 5.0


def _ref_min_distance_to_centers(pos: float, spans: list[Span]) -> int:
    if not spans:
        return 10**9
    centers = [(_center(s, e)) for s, e in spans]
    d = min(abs(c - pos) for c in centers) if centers else math.inf
    return 10**9 if d is math.inf else int(d)


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
        sid, gap, relative = choose_with_tiebreak(occ, {}, {})
        assert sid is None
        assert gap == 0.0
        assert relative == 0.0

    def test_single_candidate_short_circuit(self):
        occ = NS(start=0, end=2)
        cand_scores = {"A": 0.7}
        sid, relative, gap = choose_with_tiebreak(occ, cand_scores, {})
        assert sid == "A"
        assert gap == pytest.approx(0.7)  # p2=0 -> gap=0.7

    def test_clear_margin_winner(self):
        occ = NS(start=10, end=20)
        cand_scores = {"A": 0.90, "B": 0.70}
        sid, relative, gap = choose_with_tiebreak(occ, cand_scores, {})
        assert sid == "A"
        assert gap == pytest.approx(0.20)
        assert relative == pytest.approx(0.2222, abs=1e-4)

    def test_near_tie_distance_picks_second(self):
        # Force near tie (diff ≤ near_tie_margin and margin < margin_threshold)
        occ = NS(start=10, end=12)  # center = 11
        cand_scores = {"A": 0.51, "B": 0.50}
        meanings = {
            "A": NS(def_spans=[(99, 101)]),  # far (center ~100)
            "B": NS(def_spans=[(10, 12)]),  # near (center ~11)
        }
        sid, relative, gap = choose_with_tiebreak(occ, cand_scores, meanings)
        assert sid == "B"
        assert gap == pytest.approx(0.01)
        assert relative == pytest.approx(0.0196078, abs=1e-7)

    def test_near_tie_distance_picks_first(self):
        occ = NS(start=50, end=52)  # center = 51
        cand_scores = {"A": 0.505, "B": 0.50}
        meanings = {
            "A": NS(def_spans=[(50, 52)]),  # near
            "B": NS(def_spans=[(200, 220)]),  # far
        }
        sid, relative, gap = choose_with_tiebreak(occ, cand_scores, meanings)
        assert sid == "A"
        assert gap == pytest.approx(0.005)
        assert relative == pytest.approx(0.0099009900, abs=1e-7)

    def test_near_tie_undecided_returns_none_when_within_bias(self):
        # Distances within the ±2 bias window → unresolved tie
        occ = NS(start=0, end=2)  # center = 1
        cand_scores = {"A": 0.500, "B": 0.495}
        meanings = {
            "A": NS(def_spans=[(0, 2)]),  # center 1 -> d=0
            "B": NS(def_spans=[(0, 4)]),  # center 2 -> d=1 (within bias window)
        }
        sid, relative, gap = choose_with_tiebreak(occ, cand_scores, meanings)
        assert sid is None
        assert gap == pytest.approx(0.005)
        assert relative == pytest.approx(0.01, abs=1e-2)

    def test_uses_center_of_occurrence(self):
        cand_scores = {"A": 0.51, "B": 0.50}
        meanings = {
            "A": NS(def_spans=[(0, 2)]),  # center 1
            "B": NS(def_spans=[(8, 12)]),  # center 10
        }
        sid1, _, _ = choose_with_tiebreak(NS(start=0, end=2), cand_scores, meanings)  # center=1
        sid2, _, _ = choose_with_tiebreak(NS(start=9, end=11), cand_scores, meanings)  # center=10
        assert sid1 == "A"
        assert sid2 == "B"

    def test_handles_missing_or_empty_def_spans(self):
        occ = NS(start=0, end=2)
        cand_scores = {"A": 0.51, "B": 0.50}
        meanings = {
            "A": NS(def_spans=None),  # should be treated as []
            "B": NS(def_spans=[]),
        }
        sid, relative, gap = choose_with_tiebreak(occ, cand_scores, meanings)
        assert sid is None
        assert gap == pytest.approx(0.01)
        assert relative == pytest.approx(0.019607843, abs=1e-9)


class TestChooseWithTiebreakIntegration:
    def test_order_independence_and_switching_with_position(self):
        # Three meanings; only top two matter, but include a third to ensure sort logic is robust.
        cand_probs = {"A": 0.505, "B": 0.500, "C": 0.10}

        meanings = {
            "A": NS(def_spans=[(0, 2)]),  # center 1
            "B": NS(def_spans=[(18, 22)]),  # center 20
            "C": NS(def_spans=[(100, 120)]),
        }

        # Sweep the occurrence across the line and ensure winner flips as proximity changes
        # Near A → choose A; around mid (≈10.5), distances nearly equal → None; near B → choose B.
        trail = [
            NS(start=0, end=2),  # center 1 (near A)
            NS(start=6, end=16),  # center 11 (roughly mid)
            NS(start=19, end=21),  # center 20 (near B)
        ]

        winners = []
        for occ in trail:
            sid, _, _ = choose_with_tiebreak(occ, dict(cand_probs), meanings)
            winners.append(sid)

        assert winners[0] == "A"
        assert winners[1] in (None, "A", "B")  # acceptable: depends on bias window; commonly None
        assert winners[2] == "B"

    def test_custom_thresholds_affect_outcome(self):
        # With a larger near_tie_margin, the distance tiebreak engages more often.
        occ = NS(start=50, end=52)  # center 51
        # A slight probabilistic lead for A
        cand_probs = {"A": 0.55, "B": 0.52}
        meanings = {"A": NS(def_spans=[(0, 2)]), "B": NS(def_spans=[(50, 52)])}  # B is spatially closer

        # Default thresholds → margin = (0.55-0.52)/0.55 ≈ 0.0545 < 0.10, near tie triggers, B should win
        sid_def, rel_margin_def, abs_def = choose_with_tiebreak(occ, cand_probs, meanings)
        assert sid_def == "B"
        assert 0.0 <= abs_def < 0.10
        assert 0.0 <= rel_margin_def <= 0.05454545454545459

        # If we make the margin_threshold tiny, the probabilistic winner short-circuits before distance
        sid_small_thr, rel_margin_small_thr, abs_margin_small_thr = choose_with_tiebreak(
            occ, cand_probs, meanings, margin_threshold=0.01, near_tie_margin=0.06
        )
        assert sid_small_thr == "A"
        assert abs_margin_small_thr >= 0.01
        assert rel_margin_small_thr >= 0.05454545454545459

    def test_handles_realistic_mix_missing_spans_and_ties(self):
        # Some meanings lack def_spans; ensure function remains stable.
        occ = NS(start=90, end=110)  # center 100
        cand_probs = {"A": 0.50, "B": 0.50}
        meanings = {"A": NS(def_spans=None), "B": NS(def_spans=[(98, 102)])}  # B has nearby span

        # diff=0 ≤ near_tie_margin; B has finite distance, A hits sentinel → B should win
        sid, rel_margin, abs_margin = choose_with_tiebreak(occ, cand_probs, meanings)
        assert sid == "B"
        assert abs_margin == 0.0
        assert rel_margin == 0.0


class TestDisambiguateOccurrencesUnit:
    def test_no_meanings_for_acronym_returns_none(self):
        text = "Please refer to the EMA guidelines."
        occs = [OccurrenceLite(acronym="EMA", start=19, end=22)]
        meanings = {}  # no meanings present
        out = disambiguate_occurrences(text, occs, meanings)
        assert len(out) == 1
        r = out[0]
        assert r.acronym == "EMA"
        assert r.chosen_meaning_id is None
        assert r.candidate_scores == {}
        assert r.margin == 0.0

    def test_overlap_only_picks_best_label_match(self):
        # Make distance irrelevant (no def_spans) and rely on label overlap
        text = "We work with the European Medicines Agency on drug approvals."
        occs = [OccurrenceLite(acronym="EMA", start=14, end=17)]
        meanings = {
            "EMA": [
                AcronymMeaning(
                    acronym="EMA",
                    definition="European Medicines Agency",
                    meaning_id="ema|european_medicines_agency",
                    def_spans=[],  # distance score -> 0.0
                    support=1,
                    meaning_confidence=0.9,
                ),
                AcronymMeaning(
                    acronym="EMA",
                    definition="Emergency Management Australia",
                    meaning_id="ema|emergency_management_australia",
                    def_spans=[],
                    support=1,
                    meaning_confidence=0.9,
                ),
            ]
        }
        out = disambiguate_occurrences(
            text,
            occs,
            meanings,
            dist_weight=0.0,  # isolate overlap
            overlap_weight=1.0,  # full weight on label overlap
        )
        r = out[0]
        assert r.chosen_meaning_id == "ema|european_medicines_agency"
        # Strong margin because other label has near-zero overlap
        assert r.margin >= 0.10
        assert set(r.candidate_scores) == {
            "ema|european_medicines_agency",
            "ema|emergency_management_australia",
        }
        assert all(0.0 <= v <= 1.0 for v in r.candidate_scores.values())

    def test_distance_only_picks_nearest_definition_span(self):
        text = "FDA met EMA in Brussels yesterday."
        # Occurrence near index ~8..11; pick the closer span via distance
        occs = [OccurrenceLite(acronym="EMA", start=8, end=11)]
        meanings = {
            "EMA": [
                AcronymMeaning(
                    acronym="EMA",
                    definition="European Medicines Agency",
                    meaning_id="ema|medicines",
                    def_spans=[(6, 10)],  # center ~8
                    support=2,
                    meaning_confidence=0.9,
                ),
                AcronymMeaning(
                    acronym="EMA",
                    definition="Emergency Management Australia",
                    meaning_id="ema|emergency",
                    def_spans=[(100, 110)],  # far away
                    support=2,
                    meaning_confidence=0.9,
                ),
            ]
        }
        out = disambiguate_occurrences(
            text,
            occs,
            meanings,
            dist_weight=1.0,
            overlap_weight=0.0,  # isolate distance
        )
        r = out[0]
        assert r.chosen_meaning_id == "ema|medicines"
        assert r.margin >= 0.10

    def test_meanings_by_id_fallback_built_automatically(self):
        text = "ACR appears here."
        occs = [OccurrenceLite(acronym="ACR", start=0, end=3)]
        meanings = {
            "ACR": [
                AcronymMeaning("ACR", "Alpha Core Reader", "acr|alpha_core_reader", 0.9, [(0, 2)], 1),
                AcronymMeaning("ACR", "Advanced Cardiac Rehab", "acr|advanced_cardiac_rehab", 0.9, [(50, 60)], 1),
            ]
        }
        # Pass meanings_by_id=None to exercise the internal build
        out = disambiguate_occurrences(text, occs, meanings, meanings_by_id=None, dist_weight=1.0, overlap_weight=0.0)
        assert len(out) == 1
        assert out[0].chosen_meaning_id in {"acr|alpha_core_reader", "acr|advanced_cardiac_rehab"}

    def test_windowing_affects_overlap_tokens(self):
        text = "x " * 100 + "United Kingdom Health Security Agency collaborates widely. " + "x " * 100 + "UKHSA"
        occ_start = len(text) - 5
        occs = [OccurrenceLite(acronym="UKHSA", start=occ_start, end=occ_start + 5)]
        meanings = {
            "UKHSA": [
                AcronymMeaning(
                    "UKHSA",
                    "United Kingdom Health Security Agency",
                    "ukhsa|united_kingdom_health_security_agency",
                    0.9,
                    [],  # distance 0 → rely on overlap
                    1,
                )
            ]
        }

        sid = "ukhsa|united_kingdom_health_security_agency"

        # Tiny window → label tokens outside window → overlap≈0 → may return None
        r_small = disambiguate_occurrences(text, occs, meanings,
                                           window_chars=20,
                                           dist_weight=0.0,
                                           overlap_weight=1.0)[0]

        # Large window → label tokens inside window → higher overlap
        r_large = disambiguate_occurrences(text, occs, meanings,
                                           window_chars=400,
                                           dist_weight=0.0,
                                           overlap_weight=1.0)[0]

        small_score = r_small.candidate_scores.get(sid, 0.0)
        large_score = r_large.candidate_scores.get(sid, 0.0)

        # Overlap should not decrease when we widen the window
        assert small_score <= large_score

        # With a large enough window, we should actually pick the meaning
        assert r_large.chosen_meaning_id == sid

        # With a tiny window, resolution can be None (score 0.0)
        assert r_small.chosen_meaning_id in (None, sid)

    def test_dynamic_prior_disabled_keeps_near_tie_unresolved(self, _patch):
        """
        With forced near-tie base scores, disabling the prior should leave the
        occurrence undecided *when distance tiebreak cannot distinguish meanings*.
        """
        from document_resolution.nlp.extraction.acronyms.meanings import disambiguate as mod

        def fake_base_scores_for_occurrence(*_, **__):
            return {
                "nlp|natural_language_processing": 0.50,
                "nlp|nice_lovely_plants": 0.49,
            }

        # Make distance tiebreak non-informative (same distance for every meaning).
        def fake_min_distance_to_spans(*_, **__):
            return 0

        _patch(
            mod.disambiguate_occurrences,
            base_scores_for_occurrence=fake_base_scores_for_occurrence,
            _min_distance_to_spans=fake_min_distance_to_spans,
        )

        occs = [OccurrenceLite("NLP", 0, 3)]
        out = mod.disambiguate_occurrences(
            text="x" * 50,
            occurrences=occs,
            meanings={
                "NLP": [
                    # Only ids matter because base_scores is patched; spans won’t help now anyway.
                    mod.AcronymMeaning(
                        "NLP", "Natural language processing", "nlp|natural_language_processing", 0.9, [], 1
                    ),
                    mod.AcronymMeaning("NLP", "Nice Lovely Plants", "nlp|nice_lovely_plants", 0.1, [], 1),
                ]
            },
            meanings_prior_weight=0.0,  # disable prior
            meanings_by_id={
                "nlp|natural_language_processing": mod.AcronymMeaning(
                    "NLP", "Natural language processing", "nlp|natural_language_processing", 0.9, [], 1
                ),
                "nlp|nice_lovely_plants": mod.AcronymMeaning(
                    "NLP", "Nice Lovely Plants", "nlp|nice_lovely_plants", 0.1, [], 1
                ),
            },
            window_chars=10,
            margin_threshold=0.10,
        )

        assert out and out[0].chosen_meaning_id is None

    def test_dynamic_prior_breaks_near_tie_in_favour_of_higher_confidence_meaning(self, _patch):
        def fake_base_scores_for_occurrence(*_, **__):
            return {
                "nlp|natural_language_processing": 0.50,
                "nlp|nice_lovely_plants": 0.49,
            }

        _patch(disambiguate_occurrences, base_scores_for_occurrence=fake_base_scores_for_occurrence)

        # must be non-empty so disambiguate_occurrences doesn't short-circuit
        dummy_meanings = [
            NS(
                meaning_id="nlp|natural_language_processing",
                definition="Natural language processing",
                def_spans=[(0, 1)]
            ),
            NS(meaning_id="nlp|nice_lovely_plants", definition="Nice Lovely Plants", def_spans=[(0, 1)]),
        ]

        meanings_by_id = {
            "nlp|natural_language_processing": NS(meaning_confidence=1.0, def_spans=[(0, 1)]),
            "nlp|nice_lovely_plants": NS(meaning_confidence=0.0, def_spans=[(0, 1)]),
        }

        out = disambiguate_occurrences(
            text="x" * 50,
            occurrences=[OccurrenceLite("NLP", 0, 3)],
            meanings={"NLP": dummy_meanings},
            meanings_by_id=meanings_by_id,
            window_chars=10,
            meanings_prior_weight=0.08,
            margin_threshold=0.10,
        )

        assert out
        assert out[0].chosen_meaning_id == "nlp|natural_language_processing"
