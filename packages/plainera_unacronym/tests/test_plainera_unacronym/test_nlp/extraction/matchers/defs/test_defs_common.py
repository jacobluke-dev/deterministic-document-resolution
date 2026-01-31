import pytest

from plainera_unacronym.nlp.extraction.matchers.defs.common import (
    build_initials_stream,
    align_acronym_to_initials,
    _lowercase_prefix_ok,
)


class TestBuildInitialsStreamPrecedence:
    def test_acronym_like_wins_over_allcaps_and_compound_split_ltr(self, _patch):
        calls = {"acronym_letters": 0, "split_compound": 0, "first_alnum": 0}

        def fake_is_acronym_like_token(tok):
            return True

        def fake_acronym_letters_rtl(tok):
            calls["acronym_letters"] += 1
            return ["A", "B", "C"]  # already RTL per contract

        def fake_split_compound(tok):
            calls["split_compound"] += 1
            raise AssertionError("split_compound should NOT be called for acronym-like tokens")

        def fake_first_alnum(part):
            calls["first_alnum"] += 1
            raise AssertionError("first_alnum_char_upper should NOT be called for acronym-like tokens")

        _patch(
            build_initials_stream,
            is_acronym_like_token=fake_is_acronym_like_token,
            _acronym_letters_rtl=fake_acronym_letters_rtl,
            split_compound=fake_split_compound,
            first_alnum_char_upper=fake_first_alnum,
            PUNCT_TRIM="",
        )

        stream = build_initials_stream(
            ["HTTP"],
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=True,
            split_compounds=True,
            treat_acronym_tokens_as_multi_letter=True,
        )

        assert calls["acronym_letters"] == 1
        assert calls["split_compound"] == 0
        assert calls["first_alnum"] == 0

        # ltr reverses the rtl list
        assert stream.letters == ["C", "B", "A"]
        assert stream.owners == [0, 0, 0]
        assert stream.is_stop == [False, False, False]

    def test_acronym_like_wins_over_allcaps_and_compound_split_rtl(self, _patch):
        def fake_is_acronym_like_token(tok):
            return True

        def fake_acronym_letters_rtl(tok):
            return ["A", "B", "C"]  # RTL order

        _patch(
            build_initials_stream,
            is_acronym_like_token=fake_is_acronym_like_token,
            _acronym_letters_rtl=fake_acronym_letters_rtl,
            split_compound=lambda tok: (_ for _ in ()).throw(AssertionError("should not split")),
            first_alnum_char_upper=lambda part: (_ for _ in ()).throw(AssertionError("should not read first alnum")),
            PUNCT_TRIM="",
        )

        stream = build_initials_stream(
            ["HTTP"],
            stopwords=set(),
            scan="rtl",
            expand_allcaps_tokens=True,
            split_compounds=True,
            treat_acronym_tokens_as_multi_letter=True,
        )

        # rtl uses the rtl list as-is
        assert stream.letters == ["A", "B", "C"]


class TestBuildInitialsStreamAllcapsNegativeCases:
    def test_allcaps_not_expanded_when_len_is_1(self, _patch):
        # Expansion branch requires len(tok_clean) > 1, so this should behave like normal (single initial).
        _patch(
            build_initials_stream,
            is_acronym_like_token=lambda tok: False,
            first_alnum_char_upper=lambda part: part[0].upper() if part else None,
            split_compound=lambda tok: [tok],
            PUNCT_TRIM="",
        )

        stream = build_initials_stream(
            ["A"],
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=True,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["A"]
        assert stream.owners == [0]

    def test_allcaps_not_expanded_when_not_alpha(self, _patch):
        # "HTTP2" is not .isalpha(), so must NOT expand H,T,T,P (should just take first alnum char).
        _patch(
            build_initials_stream,
            is_acronym_like_token=lambda tok: False,
            first_alnum_char_upper=lambda part: part[0].upper() if part else None,
            split_compound=lambda tok: [tok],
            PUNCT_TRIM="",
        )

        stream = build_initials_stream(
            ["HTTP2"],
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=True,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["H"]
        assert stream.owners == [0]


class TestBuildInitialsStreamPunctTrimAndInvariants:
    def test_punct_trim_is_applied_before_first_alnum(self, _patch):
        seen = {"part": None}

        def fake_first_alnum(part):
            seen["part"] = part
            return part[0].upper() if part else None

        _patch(
            build_initials_stream,
            is_acronym_like_token=lambda tok: False,
            first_alnum_char_upper=fake_first_alnum,
            split_compound=lambda tok: [tok],
            PUNCT_TRIM="()",
        )

        stream = build_initials_stream(
            ["(PDF)"],
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert seen["part"] == "PDF"
        assert stream.letters == ["P"]

    def test_stream_invariants_lengths_match(self, _patch):
        # Mixed inputs with some punctuation/stopword; ensure len(letters)==len(owners)==len(is_stop).
        _patch(
            build_initials_stream,
            is_acronym_like_token=lambda tok: False,
            split_compound=lambda tok: [tok],
            first_alnum_char_upper=lambda part: next((ch.upper() for ch in part if ch.isalnum()), None),
            PUNCT_TRIM="()",
        )

        tokens = ["(Portable)", "of", "Document", "Format", "123"]
        stream = build_initials_stream(
            tokens,
            stopwords={"of"},
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=True,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert len(stream.letters) == len(stream.owners) == len(stream.is_stop)
        # sanity: stopword "of" contributes 'O' but is_stop should mark it True for that letter
        # depending on your first_alnum/part selection, it should contribute once.
        assert True in stream.is_stop


class TestAlignAcronymToInitialsPreflightAndPlumbing:
    def test_numeric_evidence_is_passed_into_acr_alignment_targets(self, _patch, hit_cfg):
        seen = {"has_num": None, "A": None, "mode": None}

        def fake_has_numeric_evidence(tokens):
            return True

        def fake_acr_alignment_targets(acr, has_numeric_evidence):
            seen["has_num"] = has_numeric_evidence
            return ["A"]  # minimal non-empty

        def fake_align_rtl_scan(A, stream_letters, stream_is_stop, **kwargs):
            seen["A"] = A
            return [0]  # "used positions"

        _patch(
            align_acronym_to_initials,
            has_numeric_evidence=fake_has_numeric_evidence,
            acr_alignment_targets=fake_acr_alignment_targets,
            align_rtl_scan=fake_align_rtl_scan,
        )

        # Minimal stream object shape for rtl_scan
        class Stream:
            letters = ["A"]
            owners = [0]
            is_stop = [False]

        out = align_acronym_to_initials(
            "A",
            Stream(),
            tokens=["Alpha"],
            stopwords=set(),
            mode="rtl_scan",
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )

        assert seen["has_num"] is True
        assert seen["A"] == ["A"]
        assert out is not None
        assert out.tok_left == 0 and out.tok_right == 0


class TestLowercasePrefixOkEdges:
    def test_allows_lowercase_prefix_on_token0_even_if_not_alpha(self, _patch):
        _patch(_lowercase_prefix_ok, is_mixed_case_acronym=lambda acr: True, PUNCT_TRIM="")

        assert _lowercase_prefix_ok(
            acr="iOS",
            tokens=["iOS-compatible"],
            token_idx=0,
            acr_pos=0,
            lowercase_prefix_exception=True,
        ) is True

    def test_empty_token_after_trim_returns_false(self, _patch):
        _patch(_lowercase_prefix_ok, is_mixed_case_acronym=lambda acr: True, PUNCT_TRIM=".")

        assert _lowercase_prefix_ok(
            acr="iOS",
            tokens=["..."],
            token_idx=0,
            acr_pos=0,
            lowercase_prefix_exception=True,
        ) is False


import pytest

from plainera_unacronym.nlp.extraction.matchers.defs.common import (
    expand_numeric_leading_window,
    is_acronym_parenthetical_with_tail,
    inline_clause_tail,
    InitialsStream,
)


class TestExpandNumericLeadingWindow:
    def test_expands_left_and_right_over_adjacent_numeric_leading_tokens(self, _patch):
        def fake_first_alnum_char_upper(tok: str):
            for ch in tok:
                if ch.isalnum():
                    return ch.upper()
            return None

        _patch(expand_numeric_leading_window, first_alnum_char_upper=fake_first_alnum_char_upper)

        tokens = ["3M", "Portable", "Format", "2", "PDF"]
        # window covering "Portable Format" should grow to include "3M" on the left and "2" on the right
        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=2) == (0, 3)

    def test_does_not_expand_when_adjacent_tokens_are_not_numeric_leading(self, _patch):
        def fake_first_alnum_char_upper(tok: str):
            for ch in tok:
                if ch.isalnum():
                    return ch.upper()
            return None

        _patch(expand_numeric_leading_window, first_alnum_char_upper=fake_first_alnum_char_upper)

        tokens = ["Alpha", "Beta", "Gamma"]
        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=1) == (1, 1)


class TestIsAcronymParentheticalWithTail:
    @pytest.mark.parametrize(
        "snippet, acr",
        [
            ("(PDF, see Appendix A)", "PDF"),
            ('("PDF": see Appendix A)', "PDF"),
            ("('PDF' - see Appendix A)", "PDF"),
            ("(PDF—see Appendix A)", "PDF"),
            ("(PDF – see Appendix A)", "PDF"),
            ("(PDF; see Appendix A)", "PDF"),
        ],
    )
    def test_true_for_acronym_parenthetical_with_tail(self, snippet, acr):
        assert is_acronym_parenthetical_with_tail(snippet, acr) is True

    @pytest.mark.parametrize(
        "snippet, acr",
        [
            ("(PDF)", "PDF"),
            ("(Portable Document Format)", "PDF"),
            ("(PDF )", "PDF"),
            ("(PDF, )", "PDF"),  # needs a non-whitespace tail token
        ],
    )
    def test_false_for_parenthetical_without_tail(self, snippet, acr):
        assert is_acronym_parenthetical_with_tail(snippet, acr) is False


class TestInlineClauseTail:
    def test_returns_full_string_when_no_boundary(self):
        s = "Alpha Beta Gamma"
        tail, end = inline_clause_tail(s)
        assert tail == s
        assert end == len(s)

    def test_stops_at_boundary_dot_when_followed_by_space(self):
        s = "Alpha. Beta"
        tail, end = inline_clause_tail(s)
        assert tail == "Alpha"
        assert end == len("Alpha")

    def test_does_not_stop_at_dot_when_followed_by_letter(self):
        # boundary regex is [.;:](?=\\s|$) so "e.g." should not stop at the first dot
        s = "e.g. Alpha"
        tail, end = inline_clause_tail(s)
        # first dot is followed by 'g' (no stop), second dot followed by space (stop at that dot)
        assert tail == "e.g"
        assert end == len("e.g")

    def test_stops_at_colon_or_semicolon_when_followed_by_space(self):
        s = "Alpha: Beta"
        tail, end = inline_clause_tail(s)
        assert tail == "Alpha"
        assert end == len("Alpha")

        s2 = "Alpha; Beta"
        tail2, end2 = inline_clause_tail(s2)
        assert tail2 == "Alpha"
        assert end2 == len("Alpha")

    def test_stops_on_newline(self):
        s = "Alpha\nBeta"
        tail, end = inline_clause_tail(s)
        assert tail == "Alpha"
        assert end == len("Alpha")
