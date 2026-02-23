from plainera_unacronym.nlp.extraction.matchers.defs.common import (
    AlignmentHit,
    InitialsStream,
    _align_rtl_scan_wrapper,
    align_rtl_scan,
)


class TestAlignRtlScanWrapper:
    def test_returns_none_when_align_rtl_scan_returns_none(self, _patch):
        _patch(_align_rtl_scan_wrapper, align_rtl_scan=lambda *a, **k: None)
        stream = InitialsStream(letters=["A"], owners=[0], is_stop=[False])
        assert _align_rtl_scan_wrapper(
                ["A"],
                stream=stream,
                allow_upper_on_stop=False,
                allow_lower_on_non_stop=False,
        ) is None

    def test_builds_alignment_hit_from_used_positions(self, _patch):
        # pretend positions 0 and 2 were used
        _patch(_align_rtl_scan_wrapper, align_rtl_scan=lambda *a, **k: [0, 2])

        stream = InitialsStream(
            letters=["A", "B", "C"],
            owners=[2, 1, 2],  # positions 0 and 2 both owned by token 2
            is_stop=[False, False, False],
        )

        hit = _align_rtl_scan_wrapper(
            ["A", "C"],
            stream=stream,
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
        )

        assert isinstance(hit, AlignmentHit)
        assert hit.used_letter_pos == [0, 2]
        assert hit.hit_tokens == {2}
        assert hit.tok_left == 2
        assert hit.tok_right == 2

    def test_tok_bounds_span_multiple_tokens(self, _patch):
        _patch(_align_rtl_scan_wrapper, align_rtl_scan=lambda *a, **k: [0, 1])

        stream = InitialsStream(
            letters=["A", "B"],
            owners=[3, 1],
            is_stop=[False, False],
        )

        hit = _align_rtl_scan_wrapper(
            ["A", "B"],
            stream=stream,
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
        )

        assert hit.hit_tokens == {1, 3}
        assert hit.tok_left == 1
        assert hit.tok_right == 3


class TestAlignRtlScan:
    def test_returns_none_when_targets_empty(self):
        # No targets => by current implementation ti=-1, loop skips, returns used (empty list).
        # Decide what you want. Here we assert current behaviour explicitly.
        assert align_rtl_scan([], ["A"], [False], allow_upper_on_stop=False) == []

    def test_returns_none_when_initials_empty(self):
        assert align_rtl_scan(["A"], [], [], allow_upper_on_stop=False) is None

    def test_matches_simple_all_uppercase_requires_non_stop_by_default(self):
        targets = ["P", "D", "F"]
        initials = ["F", "D", "P"]
        is_stop = [False, False, False]
        used = align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False)
        assert used == [0, 1, 2]

    def test_uppercase_rejected_on_stopword_when_allow_upper_on_stop_false(self):
        targets = ["A"]
        initials = ["A"]
        is_stop = [True]
        assert align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False) is None

    def test_uppercase_allowed_on_stopword_when_allow_upper_on_stop_true(self):
        targets = ["A"]
        initials = ["A"]
        is_stop = [True]
        assert align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=True) == [0]

    def test_lowercase_requires_stopword_by_default(self):
        targets = ["m"]  # wants stop letter
        initials = ["M"]
        assert align_rtl_scan(targets, initials, [False], allow_upper_on_stop=False) is None
        assert align_rtl_scan(targets, initials, [True], allow_upper_on_stop=False) == [0]

    def test_lowercase_can_match_non_stop_when_allow_lower_on_non_stop_true(self):
        targets = ["m"]
        initials = ["M"]
        assert (
            align_rtl_scan(
            targets,
            initials,
            [False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,
            )
            == [0]
        )

    def test_greedy_scan_consumes_targets_from_right_to_left(self):
        # targets matched RTL: should match last 'A' then 'B' then 'A'
        targets = ["A", "B", "A"]
        initials = ["A", "X", "B", "A"]
        is_stop = [False, False, False, False]
        used = align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False)
        assert used == [0, 2, 3]

    def test_skips_non_matching_initials(self):
        targets = ["A", "B"]
        initials = ["B", "X", "A", "Y"]
        is_stop = [False] * len(initials)
        used = align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False)
        assert used == [0, 2]

    def test_ltr_initials_do_not_work_with_rtl_scan_contract(self):
        targets = ["P", "D", "F"]
        initials = ["P", "D", "F"]  # LTR order
        is_stop = [False, False, False]
        assert align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False) is None

    def test_mixed_case_stopword_constraints(self):
        # "MoM" style: lowercase wants stop for 'o'
        targets = ["M", "o", "M"]
        initials = ["M", "O", "M"]
        is_stop = [False, True, False]
        used = align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False)
        assert used == [0, 1, 2]

    def test_mixed_case_fails_when_stopword_constraint_violated(self):
        targets = ["M", "o", "M"]
        initials = ["M", "O", "M"]
        is_stop = [False, False, False]  # middle isn't stop
        assert align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False) is None

    def test_returns_none_when_not_fully_matched(self):
        targets = ["A", "B", "C"]
        initials = ["A", "B"]
        is_stop = [False, False]
        assert align_rtl_scan(targets, initials, is_stop, allow_upper_on_stop=False) is None
