from plainera_unacronym.nlp.extraction.matchers.defs.common import AlignmentHit, align_acronym_to_initials, \
    InitialsStream


class TestAlignAcronymToInitials:
    def test_returns_none_when_acr_empty_or_stream_empty(self):
        stream0 = InitialsStream(letters=[], owners=[], is_stop=[])
        assert (
            align_acronym_to_initials(
                "",
                stream0,
                tokens=[],
                stopwords=set(),
                mode="rtl_scan",
                allow_upper_on_stop=False,
                allow_lower_on_non_stop=False,
                lowercase_prefix_exception=False,
            )
            is None
        )

        stream1 = InitialsStream(letters=[], owners=[], is_stop=[])
        assert (
            align_acronym_to_initials(
                "PDF",
                stream1,
                tokens=["Portable"],
                stopwords=set(),
                mode="rtl_scan",
                allow_upper_on_stop=False,
                allow_lower_on_non_stop=False,
                lowercase_prefix_exception=False,
            )
            is None
        )

    def test_returns_none_when_alignment_targets_empty(self, _patch):
        _patch(
            align_acronym_to_initials,
            has_numeric_evidence=lambda tokens: False,
            acr_alignment_targets=lambda acr, has_numeric_evidence: [],
        )

        stream = InitialsStream(letters=["P"], owners=[0], is_stop=[False])
        assert (
            align_acronym_to_initials(
                "???",
                stream,
                tokens=["Portable"],
                stopwords=set(),
                mode="rtl_scan",
                allow_upper_on_stop=False,
                allow_lower_on_non_stop=False,
                lowercase_prefix_exception=False,
            )
            is None
        )

    def test_rtl_mode_delegates_to_align_rtl_wrapper(self, _patch):
        sentinel = AlignmentHit(used_letter_pos=[1], hit_tokens={0}, tok_left=0, tok_right=0)
        seen = {}

        def fake_align_rtl(A, *, stream, allow_upper_on_stop, allow_lower_on_non_stop):
            seen["A"] = A
            seen["allow_upper_on_stop"] = allow_upper_on_stop
            seen["allow_lower_on_non_stop"] = allow_lower_on_non_stop
            return sentinel

        _patch(
            align_acronym_to_initials,
            has_numeric_evidence=lambda tokens: False,
            acr_alignment_targets=lambda acr, has_numeric_evidence: ["P", "D", "F"],
            _align_rtl_scan=fake_align_rtl,
        )

        stream = InitialsStream(letters=["P", "D", "F"], owners=[0, 1, 2], is_stop=[False, False, False])

        out = align_acronym_to_initials(
            "PDF",
            stream,
            tokens=["Portable", "Document", "Format"],
            stopwords=set(),
            mode="rtl_scan",
            allow_upper_on_stop=True,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )

        assert out is sentinel
        assert seen["A"] == ["P", "D", "F"]
        assert seen["allow_upper_on_stop"] is True
        assert seen["allow_lower_on_non_stop"] is False

    def test_ltr_min_window_delegates_to_ltr_impl_and_passes_is_stop_token(self, _patch):
        sentinel = AlignmentHit(used_letter_pos=[0], hit_tokens={1}, tok_left=1, tok_right=1)
        seen = {}

        def fake_ltr(
            A,
            *,
            stream,
            tokens,
            is_stop_token,
            allow_upper_on_stop,
            allow_lower_on_non_stop,
            lowercase_prefix_exception,
        ):
            seen["A"] = A
            seen["tokens"] = tokens
            seen["is_stop_token"] = is_stop_token
            seen["allow_upper_on_stop"] = allow_upper_on_stop
            seen["allow_lower_on_non_stop"] = allow_lower_on_non_stop
            seen["lowercase_prefix_exception"] = lowercase_prefix_exception
            return sentinel

        _patch(
            align_acronym_to_initials,
            has_numeric_evidence=lambda tokens: False,
            acr_alignment_targets=lambda acr, has_numeric_evidence: ["M", "O", "M"],
            _align_ltr_min_window=fake_ltr,
        )

        tokens = ["Ministry", "of", "Magic"]
        stopwords = {"of"}

        stream = InitialsStream(letters=["M", "O", "M"], owners=[0, 1, 2], is_stop=[False, True, False])

        out = align_acronym_to_initials(
            "MOM",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="ltr_min_window",
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,
            lowercase_prefix_exception=True,
        )

        assert out is sentinel
        assert seen["A"] == ["M", "O", "M"]
        assert seen["tokens"] == tokens
        assert seen["is_stop_token"] == [False, True, False]
        assert seen["allow_upper_on_stop"] is False
        assert seen["allow_lower_on_non_stop"] is True
        assert seen["lowercase_prefix_exception"] is True
