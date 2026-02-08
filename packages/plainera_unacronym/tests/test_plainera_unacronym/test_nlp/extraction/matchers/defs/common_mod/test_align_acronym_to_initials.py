
from plainera_unacronym.nlp.common.constants_regex import DEFAULT_STOPWORDS
from plainera_unacronym.nlp.extraction.matchers.defs.common import (
    AlignmentHit,
    InitialsStream,
    align_acronym_to_initials,
    build_initials_stream,
)


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
            acr_alignment_targets=lambda acr,
            has_numeric_evidence: ["P", "D", "F"],
            _align_rtl_scan_wrapper=fake_align_rtl,
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


class TestAlignAcronymToInitialsIntegration:
    def test_ltr_min_window_basic_aligns_and_span_is_correct(self):
        tokens = ["Portable", "Document", "Format"]
        stopwords = DEFAULT_STOPWORDS  # fine; none of these are stopwords anyway

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        hit = align_acronym_to_initials(
            "PDF",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="ltr_min_window",
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )

        assert hit is not None
        assert hit.tok_left == 0
        assert hit.tok_right == 2
        assert hit.hit_tokens == {0, 1, 2}

    def test_rtl_scan_basic_aligns_and_span_is_correct(self):
        tokens = ["Portable", "Document", "Format"]
        stopwords = DEFAULT_STOPWORDS

        # build RTL stream (letters emitted in RTL scan order)
        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="rtl",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        hit = align_acronym_to_initials(
            "PDF",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="rtl_scan",
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,  # irrelevant in rtl mode
        )

        assert hit is not None
        assert hit.tok_left == 0
        assert hit.tok_right == 2
        assert hit.hit_tokens == {0, 1, 2}

    def test_stopword_constraint_blocks_when_uppercase_letter_lands_on_stopword(self):
        tokens = ["Ministry", "of", "Magic"]
        stopwords = {"of"}  # keep it explicit

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        # "MOM": middle letter is 'O' which corresponds to token "of" (stopword)
        hit = align_acronym_to_initials(
            "MOM",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="ltr_min_window",
            allow_upper_on_stop=False,  # strict: uppercase may NOT land on stopword
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )

        assert hit is None

    def test_allow_upper_on_stop_allows_stopword_landing(self):
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

        hit = align_acronym_to_initials(
            "MOM",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="ltr_min_window",
            allow_upper_on_stop=True,  # relaxed
            allow_lower_on_non_stop=False,
            lowercase_prefix_exception=False,
        )

        assert hit is not None
        assert hit.hit_tokens == {0, 1, 2}
        assert (hit.tok_left, hit.tok_right) == (0, 2)

    def test_mixed_case_prefix_exception_required_for_mrna_style(self):
        # First acronym char is lowercase => matcher treats it as "wants stopword",
        # and the ONLY intended escape hatch is lowercase_prefix_exception for token[0].
        tokens = ["molecule", "Ribonucleic", "Nucleic", "Acid"]
        stopwords = set()

        stream = build_initials_stream(
            tokens,
            stopwords=stopwords,
            scan="ltr",
            expand_allcaps_tokens=False,
            split_compounds=False,
            treat_acronym_tokens_as_multi_letter=False,
        )

        # Without lowercase_prefix_exception, this should fail (by design).
        hit_no_exc = align_acronym_to_initials(
            "mRNA",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="ltr_min_window",
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,  # mixed-case acronyms usually set this True
            lowercase_prefix_exception=False,
        )
        assert hit_no_exc is None

        # With lowercase_prefix_exception=True, it should align.
        hit_exc = align_acronym_to_initials(
            "mRNA",
            stream,
            tokens=tokens,
            stopwords=stopwords,
            mode="ltr_min_window",
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=True,
            lowercase_prefix_exception=True,
        )
        assert hit_exc is not None
        assert hit_exc.hit_tokens == {0, 1, 2, 3}
        assert (hit_exc.tok_left, hit_exc.tok_right) == (0, 3)

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
