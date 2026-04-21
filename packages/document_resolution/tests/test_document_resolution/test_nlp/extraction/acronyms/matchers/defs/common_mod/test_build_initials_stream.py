from document_resolution.nlp.extraction.acronyms.matchers.defs.defs_common import build_initials_stream


class TestBuildInitialsStreamBasics:
    def test_ltr_basic_single_initial_per_token(self, _patch):
        # keep it deterministic: no acronym-like, no compound splitting beyond identity
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
            is_acronym_like_token=lambda s: False,
        )

        tokens = ["Portable", "Document", "Format"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["P", "D", "F"]
        assert stream.owners == [0, 1, 2]
        assert stream.is_stop == [False, False, False]

    def test_rtl_reverses_token_traversal(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
            is_acronym_like_token=lambda s: False,
        )

        tokens = ["Portable", "Document", "Format"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="rtl",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["F", "D", "P"]
        assert stream.owners == [2, 1, 0]
        assert stream.is_stop == [False, False, False]

    def test_stopword_status_is_per_letter_via_owner(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
            is_acronym_like_token=lambda s: False,
        )

        tokens = ["Ministry", "of", "Magic"]
        stopwords = {"of"}

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["M", "O", "M"]
        assert stream.owners == [0, 1, 2]
        assert stream.is_stop == [False, True, False]


class TestBuildInitialsStreamOptions:
    def test_split_compounds_ltr_uses_part_order(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            is_acronym_like_token=lambda s: False,
            # pretend CamelCase split
            split_compound=lambda s: ["Hyper", "Text"] if s == "HyperText" else [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
        )

        tokens = ["HyperText", "Transfer"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=True,
            treat_acronym_tokens_as_multi_letter=False,
        )

        # HyperText -> H, T; Transfer -> T
        assert stream.letters == ["H", "T", "T"]
        assert stream.owners == [0, 0, 1]

    def test_split_compounds_rtl_reverses_part_order(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            is_acronym_like_token=lambda s: False,
            split_compound=lambda s: ["Hyper", "Text"] if s == "HyperText" else [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
        )

        tokens = ["HyperText", "Transfer"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="rtl",
            expand_allcaps_tokens=False,
            split_compounds=True,
            treat_acronym_tokens_as_multi_letter=False,
        )

        # rtl: token1 first => T; then HyperText parts reversed => T, H
        assert stream.letters == ["T", "T", "H"]
        assert stream.owners == [1, 0, 0]

    def test_expand_allcaps_tokens_ltr(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            is_acronym_like_token=lambda s: False,
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
        )

        tokens = ["HTTP"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=True,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["H", "T", "T", "P"]
        assert stream.owners == [0, 0, 0, 0]

    def test_expand_allcaps_tokens_rtl(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            is_acronym_like_token=lambda s: False,
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
        )

        tokens = ["HTTP"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="rtl",
            expand_allcaps_tokens=True,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == ["P", "T", "T", "H"]
        assert stream.owners == [0, 0, 0, 0]

    def test_treat_acronym_like_tokens_as_multi_letter_ltr_reverses_rtl_letters(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="()",
            is_acronym_like_token=lambda s: True,
            _acronym_letters_rtl=lambda s: iter(["F", "D", "P"]),  # rtl order
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
        )

        tokens = ["(PDF)"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=True,
        )

        # ltr converts rtl letters into ltr order
        assert stream.letters == ["P", "D", "F"]
        assert stream.owners == [0, 0, 0]

    def test_treat_acronym_like_tokens_as_multi_letter_rtl_keeps_rtl_letters(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="()",
            is_acronym_like_token=lambda s: True,
            _acronym_letters_rtl=lambda s: iter(["F", "D", "P"]),  # rtl order
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: s[0].upper() if s else None,
        )

        tokens = ["(PDF)"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="rtl",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=True,
        )

        assert stream.letters == ["F", "D", "P"]
        assert stream.owners == [0, 0, 0]

    def test_skips_parts_with_no_alnum(self, _patch):
        _patch(
            build_initials_stream,
            PUNCT_TRIM="",
            is_acronym_like_token=lambda s: False,
            split_compound=lambda s: [s],
            first_alnum_char_upper=lambda s: None,  # everything is "non-alnum"
        )

        tokens = ["---", "###"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        assert stream.letters == []
        assert stream.owners == []
        assert stream.is_stop == []


class TestBuildInitialsStreamIntegration:
    def test_plain_words_ltr(self):
        tokens = ["Portable", "Document", "Format"]
        stream = build_initials_stream(
            tokens,
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )
        assert stream.letters == ["P", "D", "F"]
        assert stream.owners == [0, 1, 2]
        assert stream.is_stop == [False, False, False]

    def test_plain_words_rtl_changes_order(self):
        tokens = ["Portable", "Document", "Format"]
        stream = build_initials_stream(
            tokens,
            stopwords=set(),
            scan="rtl",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )
        # In RTL scan, tokens are visited right-to-left; initials follow scan order.
        assert stream.letters == ["F", "D", "P"]
        assert stream.owners == [2, 1, 0]

    def test_stopword_status_propagates(self):
        tokens = ["Ministry", "of", "Magic"]
        stopwords = {"of"}
        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )
        assert stream.letters == ["M", "O", "M"]
        assert stream.is_stop == [False, True, False]

    def test_acronym_like_token_expands_when_enabled(self):
        tokens = ["U.S.A", "Passport"]
        stream = build_initials_stream(
            tokens,
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=True,
        )
        # Expect multiple letters owned by token 0, plus P from token 1
        assert stream.letters[-1] == "P"
        assert stream.owners[-1] == 1
        assert stream.owners.count(0) >= 2

    def test_allcaps_token_expands_only_when_flag_true(self):
        tokens = ["HTTP", "Server"]

        no_expand = build_initials_stream(
            tokens,
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )
        assert no_expand.letters[:2] == ["H", "S"]

        expand = build_initials_stream(
            tokens,
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=True,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )
        # Should expand HTTP into multiple letters owned by token 0
        assert expand.letters[:4] == ["H", "T", "T", "P"]
        assert expand.owners[:4] == [0, 0, 0, 0]
        assert expand.letters[4] == "S"
        assert expand.owners[4] == 1

    def test_numeric_leading_token_produces_numeric_initial(self):
        tokens = ["3M", "Command"]
        stream = build_initials_stream(
            tokens,
            stopwords=set(),
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )
        # first_alnum_char_upper should typically return '3' for '3M'
        assert stream.owners[0] == 0
        assert stream.letters[0].isdigit() or stream.letters[0] == "3"


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
        # depending on the first_alnum/part selection, it should contribute once.
        assert True in stream.is_stop
