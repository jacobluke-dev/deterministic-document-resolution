from plainera_unacronym.nlp.extraction.acronyms.matchers.defs.common import InitialsStream, _align_ltr_min_window


class TestAlignLtrMinWindowBasics:
    def test_returns_none_when_no_alignment_possible(self, hit_cfg):
        stream = InitialsStream(
            letters=["X", "Y"],
            owners=[0, 1],
            is_stop=[False, False],
        )
        out = _align_ltr_min_window(
            ["A"],
            stream=stream,
            tokens=["Alpha", "Beta"],
            is_stop_token=[False, False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )
        assert out is None

    def test_picks_smallest_token_span_when_multiple_matches_exist(self, hit_cfg):
        # Two ways to match "AB":
        # - positions [0,1] => owners [0,2] span = 2
        # - positions [2,3] => owners [0,1] span = 1  (should win)
        stream = InitialsStream(
            letters=["A", "B", "A", "B"],
            owners=[0, 2, 0, 1],
            is_stop=[False, False, False, False],
        )
        out = _align_ltr_min_window(
            ["A", "B"],
            stream=stream,
            tokens=["Alpha", "Beta", "Bravo"],
            is_stop_token=[False, False, False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )
        assert out is not None
        assert out.tok_left == 0
        assert out.tok_right == 1
        assert out.used_letter_pos == [2, 3]
        assert out.hit_tokens == {0, 1}


class TestAlignLtrMinWindowStopwordAndCaseConstraints:
    def test_uppercase_letter_cannot_land_on_stopword_when_not_allowed(self):
        # Alignment wants "O" (uppercase => want_stop False), but token[0] is stopword
        stream = InitialsStream(
            letters=["O"],
            owners=[0],
            is_stop=[True],
        )
        out = _align_ltr_min_window(
            ["O"],
            stream=stream,
            tokens=["of"],
            is_stop_token=[True],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )
        assert out is None

    def test_uppercase_letter_can_land_on_stopword_when_allowed(self):
        stream = InitialsStream(
            letters=["O"],
            owners=[0],
            is_stop=[True],
        )
        out = _align_ltr_min_window(
            ["O"],
            stream=stream,
            tokens=["of"],
            is_stop_token=[True],
            allow_upper_on_stop=True,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )
        assert out is not None
        assert out.tok_left == 0 and out.tok_right == 0

    def test_lowercase_letter_requires_stopword_unless_exception_path_enabled(self):
        # alignment "i" (lowercase => want_stop True) against non-stop token
        stream = InitialsStream(
            letters=["I"],
            owners=[0],
            is_stop=[False],
        )

        # 1) strict: disallow lower-on-non-stop
        out = _align_ltr_min_window(
            ["i"],
            stream=stream,
            tokens=["iOS"],
            is_stop_token=[False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )
        assert out is None

        # 2) allow lower-on-non-stop but NOT exception => still fail
        out = _align_ltr_min_window(
            ["i"],
            stream=stream,
            tokens=["iOS"],
            is_stop_token=[False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,
            lowercase_prefix_exception=False,
        )
        assert out is None

        # 3) allow + exception => should pass if token0 starts with 'i' and isn't ALLCAPS alpha upper
        out = _align_ltr_min_window(
            ["i"],
            stream=stream,
            tokens=["iOS"],
            is_stop_token=[False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,
            lowercase_prefix_exception=True,
        )
        assert out is not None
        assert out.tok_left == 0 and out.tok_right == 0

    def test_lowercase_prefix_exception_rejects_allcaps_word_token(self):
        # token0 is alpha+upper+len>1, so exception should refuse it
        stream = InitialsStream(
            letters=["I"],
            owners=[0],
            is_stop=[False],
        )
        out = _align_ltr_min_window(
            ["i"],
            stream=stream,
            tokens=["IOS"],
            is_stop_token=[False],
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,
            lowercase_prefix_exception=True,
        )
        assert out is None
