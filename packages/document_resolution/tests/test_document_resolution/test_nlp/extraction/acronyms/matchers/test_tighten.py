import pytest
from document_resolution.nlp.extraction.acronyms.matchers.tighten import (
    _best_window_for_acronym,
    _numeric_leading,
    _phrase_from_best_window,
    _tokenize_preserve,
    _try_split_acronym_initials_window,
    tighten_label_by_acronym,
)


class TestTokenizePreserve:
    def test_empty(self):
        assert _tokenize_preserve("") == []

    def test_whitespace_only(self):
        assert _tokenize_preserve("   \n\t ") == []

    def test_basic_words(self):
        assert _tokenize_preserve("Hello world 123") == ["Hello", "world", "123"]

    def test_preserves_hyphen_slash_ampersand_dot(self):
        s = "Foo-Bar Foo/Bar R&D U.S.A. A/B/C"
        assert _tokenize_preserve(s) == ["Foo-Bar", "Foo/Bar", "R&D", "U.S.A.", "A/B/C"]

    def test_preserves_ascii_and_curly_apostrophes(self):
        s = "can't don’t O'Reilly ‘no’ “quotes”"
        # Note: the quotes characters are NOT in the regex; only apostrophes are.
        assert _tokenize_preserve(s) == ["can't", "don’t", "O'Reilly", "no’", "quotes"]

    def test_parentheses_and_commas_are_boundaries(self):
        s = "PF (3M Portable format), v1.2"
        assert _tokenize_preserve(s) == ["PF", "3M", "Portable", "format", "v1.2"]

    def test_non_ascii_letters_split_ascii_runs(self):
        assert _tokenize_preserve("Ångström") == ["ngstr", "m"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", []),  # empty
            ("   \t\n", []),  # whitespace only
            ("Hello", ["Hello"]),  # simple word
            ("Hello world", ["Hello", "world"]),  # spaces split
            ("O'Reilly", ["O'Reilly"]),  # straight apostrophe
            ("Queen’s Award", ["Queen’s", "Award"]),  # curly apostrophe
            ("R&D", ["R&D"]),  # ampersand preserved
            ("C/C++", ["C/C"]),  # slash ok; '+' not allowed -> dropped
            ("U.S.A.", ["U.S.A."]),  # dots preserved inside token
            ("v1.2.3", ["v1.2.3"]),  # mixed digits + dots
            ("foo_bar", ["foo", "bar"]),  # underscore splits (not allowed)
            ("Email a.b@c.com now", ["Email", "a.b", "c.com", "now"]),  # '@' splits
            ("dash-separated-words", ["dash-separated-words"]),  # hyphen preserved
            ("mix&match/okay.now", ["mix&match/okay.now"]),  # combo delimiters preserved
            ("(Portable) Document, Format!", ["Portable", "Document", "Format"]),  # strip punctuation
            ("β-blocker", ["-blocker"]),  # leading non-ASCII splits; hyphen+word captured
            ("Заказ-123", ["-123"]),  # Cyrillic splits; digits captured (ASCII-only)
            ("3/4-inch", ["3/4-inch"]),  # digits + slash + hyphen
        ],
    )
    def test_tokenization_various(self, text, expected):
        assert _tokenize_preserve(text) == expected

    def test_multiple_delimiters_collapse(self):
        text = "a--b///c..d&&e"
        assert _tokenize_preserve(text) == ["a--b///c..d&&e"]  # all allowed -> one token

    def test_parentheses_and_quotes(self):
        text = "\"'Hello' (world).\""
        assert _tokenize_preserve(text) == ["'Hello'", "world", "."]


class TestNumericLeading:
    @pytest.mark.parametrize(
        "tok, expected",
        [
            # ---- basic digit-leading ----
            ("3M", True),
            ("5th", True),
            ("12V", True),
            ("0-day", True),
            # ---- leading punctuation should be ignored ----
            ("(3M)", True),
            ("'5th'", True),
            ('"12V"', True),
            ("—7Zip", True),
            ("..9", True),
            # ---- alpha-leading ----
            ("GPU", False),
            ("v1", False),  # first alnum is 'v'
            ("x86", False),  # first alnum is 'x'
            ("B2B", False),  # first alnum is 'B'
            # ---- no alnum ----
            ("", False),
            ("--", False),
            ("…—", False),
            # ---- whitespace ----
            ("   5G", True),
            ("   GPU", False),
            # ---- non-ascii digits (optional: depends on your policy) ----
            # If you *don't* want this to count, change implementation to `ch in "0123456789"`.
            ("١٢V", True),  # Arabic-Indic digit '١' isdigit() == True
        ],
    )
    def test_numeric_leading(self, tok: str, expected: bool):
        assert _numeric_leading(tok) is expected


class TestTrySplitAcronymInitialsWindow:
    def test_returns_none_when_not_split_acronym(self):
        assert (
            _try_split_acronym_initials_window(
                tokens=["Portable", "Document", "Format"],
                acronym="PDF",
                bridges={"and", "of"},
                keep_case=True,
            )
            is None
        )

    def test_returns_none_when_too_short_after_stripping(self):
        # e.g. "/" only => letters list too short
        assert (
            _try_split_acronym_initials_window(
                tokens=["foo", "bar"],
                acronym="/",
                bridges=set(),
                keep_case=True,
            )
            is None
        )

    def test_subsequence_match_can_skip_non_bridge_tokens(self):
        got = _try_split_acronym_initials_window(
            tokens=["Cost", "Benefit", "Analysis"],
            acronym="C/A",
            bridges=set(),
            keep_case=True,
        )
        assert got == "Cost Analysis"

    def test_basic_split_acronym_extracts_matched_tokens_only(self):
        got = _try_split_acronym_initials_window(
            tokens=["Cost", "Analysis"],
            acronym="C/A",
            bridges=set(),
            keep_case=True,
        )
        assert got == "Cost Analysis"

    def test_keeps_bridges_inside_window(self):
        got = _try_split_acronym_initials_window(
            tokens=["Research", "and", "Development"],
            acronym="R&D",
            bridges={"and", "of"},
            keep_case=True,
        )
        assert got == "Research and Development"

    def test_does_not_keep_non_bridge_fillers(self):
        got = _try_split_acronym_initials_window(
            tokens=["Research", "very", "Development"],
            acronym="R&D",
            bridges={"and", "of"},
            keep_case=True,
        )
        assert got == "Research Development"

    def test_expands_to_include_numeric_leading_neighbour_on_left(self):
        # window would be "Portable format" (P/F); numeric-leading "3M" sits immediately left
        got = _try_split_acronym_initials_window(
            tokens=["3M", "Portable", "format"],
            acronym="P/F",
            bridges=set(),
            keep_case=True,
        )
        assert got == "3M Portable format"

    def test_expands_to_include_numeric_leading_neighbour_on_right(self):
        got = _try_split_acronym_initials_window(
            tokens=["Portable", "format", "3M"],
            acronym="P/F",
            bridges=set(),
            keep_case=True,
        )
        assert got == "Portable format 3M"

    def test_keeps_numeric_leading_tokens_inside_window_even_if_not_hit_or_bridge(self):
        got = _try_split_acronym_initials_window(
            tokens=["Research", "3M", "and", "Development"],
            acronym="R&D",
            bridges={"and"},
            keep_case=True,
        )
        # 3M is within low..high so it should be retained by numeric-leading rule
        assert got == "Research 3M and Development"

    def test_lowercases_when_keep_case_false(self):
        got = _try_split_acronym_initials_window(
            tokens=["Research", "and", "Development"],
            acronym="R&D",
            bridges={"and"},
            keep_case=False,
        )
        assert got == "research and development"

    def test_handles_multiple_markers_abc(self):
        got = _try_split_acronym_initials_window(
            tokens=["Alpha", "Beta", "and", "Charlie"],
            acronym="A/B/C",
            bridges={"and"},
            keep_case=True,
        )
        assert got == "Alpha Beta and Charlie"

    def test_allows_digits_in_acronym_letters_stream(self):
        # Acronym letters include digit; initials are derived from token[0] only.
        # This should fail unless you have tokens starting with a digit in the sequence.
        got = _try_split_acronym_initials_window(
            tokens=["3M", "Portable", "format"],
            acronym="3/P/F",
            bridges=set(),
            keep_case=True,
        )
        assert got == "3M Portable format"


class TestPhraseFromBestWindow:
    def test_returns_none_when_no_window(self, monkeypatch):
        def _fake_best_window(tokens, acronym):
            return None

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["Portable", "Format"],
            acronym="PF",
            bridges=set(),
            keep_case=True,
        )
        assert got is None

    def test_keeps_hits_and_bridges_only(self, monkeypatch):
        # Window is tokens[0:4] inclusive; hits are indices 0 and 3.
        # Bridge "of" should be retained; "the" should be dropped if not a bridge/hit.
        def _fake_best_window(tokens, acronym):
            return 0, 3, {0, 3}

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["Portable", "of", "the", "Format"],
            acronym="PF",
            bridges={"of"},
            keep_case=True,
        )
        assert got == "Portable of Format"

    def test_expands_window_to_include_numeric_leading_neighbors(self, monkeypatch):
        # Base window hits PF in "Portable Format" => indices 1..2.
        # Numeric-leading "3M" at index 0 should be included by expansion.
        def _fake_best_window(tokens, acronym):
            return 1, 2, {1, 2}

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["3M", "Portable", "Format"],
            acronym="PF",
            bridges=set(),
            keep_case=True,
        )
        assert got == "3M Portable Format"

    def test_numeric_leading_on_right_is_included(self, monkeypatch):
        # Window is indices 0..1 ("Portable Format"), numeric-leading "2FA" at 2 should be included.
        def _fake_best_window(tokens, acronym):
            return 0, 1, {0, 1}

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["Portable", "Format", "2FA"],
            acronym="PF",
            bridges=set(),
            keep_case=True,
        )
        assert got == "Portable Format 2FA"

    def test_lowercases_when_keep_case_false(self, monkeypatch):
        def _fake_best_window(tokens, acronym):
            return 0, 1, {0, 1}

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["Portable", "Format"],
            acronym="PF",
            bridges=set(),
            keep_case=False,
        )
        assert got == "portable format"

    def test_falls_back_to_full_window_if_pruning_removes_everything(self, monkeypatch):
        # No hits and no bridges: kept becomes empty, so it must fall back to full expanded window.
        def _fake_best_window(tokens, acronym):
            return 0, 2, set()

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["Alpha", "Beta", "Gamma"],
            acronym="AB",
            bridges=set(),
            keep_case=True,
        )
        assert got == "Alpha Beta Gamma"

    def test_strips_trailing_punct_and_collapses_ws(self, monkeypatch):
        def _fake_best_window(tokens, acronym):
            return 0, 2, {0, 2}

        import document_resolution.nlp.extraction.acronyms.matchers.tighten as mod

        monkeypatch.setattr(mod, "_best_window_for_acronym", _fake_best_window)

        got = mod._phrase_from_best_window(
            tokens=["Portable", "   ", "Format,"],
            acronym="PF",
            bridges=set(),
            keep_case=True,
        )
        # collapse_ws + strip_trailing_punct_str should remove trailing comma and excess whitespace.
        assert got == "Portable Format"


class TestPhraseFromBestWindowIntegration:
    def test_aligns_and_prunes_to_hits_only(self):
        # PF should align to Portable + Format
        tokens = ["Portable", "Format", "blah", "blah"]
        got = _phrase_from_best_window(tokens=tokens, acronym="PF", bridges=set(), keep_case=True)

        assert got == "Portable Format"

    def test_keeps_bridge_words_inside_window(self):
        # Common: "Terms of Service" should keep "of" if it's a bridge
        tokens = ["Terms", "of", "Service"]
        got = _phrase_from_best_window(tokens=tokens, acronym="TOS", bridges={"of"}, keep_case=True)

        assert got == "Terms of Service"

    def test_keeps_stopword_when_it_is_a_hit(self):
        tokens = ["Terms", "of", "Service"]
        got = _phrase_from_best_window(tokens=tokens, acronym="TOS", bridges=set(), keep_case=True)
        assert got == "Terms of Service"

    def test_expands_to_include_numeric_leading_left_neighbour(self):
        # Regression guard: numeric-leading tokens should be preserved (e.g., "3M").
        # Window for PF should naturally be "Portable format" then expand to include "3M".
        tokens = ["3M", "Portable", "format"]
        got = _phrase_from_best_window(tokens=tokens, acronym="PF", bridges=set(), keep_case=True)

        assert got == "3M Portable format"

    def test_expands_to_include_numeric_leading_right_neighbour(self):
        # Numeric-leading token immediately after the best window should be preserved too.
        tokens = ["Portable", "format", "3M"]
        got = _phrase_from_best_window(tokens=tokens, acronym="PF", bridges=set(), keep_case=True)

        assert got == "Portable format 3M"

    def test_returns_none_when_no_alignment_window_exists(self):
        tokens = ["nothing", "here", "matches"]
        got = _phrase_from_best_window(tokens=tokens, acronym="XYZ", bridges=set(), keep_case=True)

        assert got is None

    def test_lowercases_when_keep_case_false(self):
        tokens = ["Portable", "Format"]
        got = _phrase_from_best_window(tokens=tokens, acronym="PF", bridges=set(), keep_case=False)

        assert got == "portable format"


class TestBestWindowForAcronymUnit:
    def test_returns_none_when_acronym_has_no_alnum(self, _patch):
        # Even if initials exist, empty A should short-circuit
        _patch(_best_window_for_acronym, _initials_seq=lambda t, s: (["A"], [0]))
        assert _best_window_for_acronym(["any"], "--__--") is None

    def test_returns_none_when_no_letters(self, _patch):
        # No initials generated → None
        _patch(_best_window_for_acronym, initials_seq=lambda t, *a, **k: ([], []))
        assert _best_window_for_acronym(["Portable", "Document"], "PD") is None

    def test_picks_shortest_window_and_keeps_first_on_tie(self, _patch):
        # Provide deterministic initials and owners; use real _match_from
        def fake_initials_seq(tokens, expand_allcaps):
            # letters index: 0:X, 1:P, 2:D, 3:F, 4:P, 5:D, 6:F
            return ["X", "P", "D", "F", "P", "D", "F"], [0, 1, 2, 3, 4, 5, 6]

        _patch(_best_window_for_acronym, initials_seq=fake_initials_seq)

        tokens = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]
        out = _best_window_for_acronym(tokens, "PDF")
        # Two equally short windows exist: tokens [1..3] and [4..6]; function keeps the first
        assert out == (1, 3, {1, 2, 3})

    def test_hits_contains_only_tokens_that_contributed_letters(self, _patch):
        # Make a window covering tokens 0..3, but ensure only tokens 0,2,3 supply matched initials
        def fake_initials_seq(tokens):
            # letters: P(0), X(1), D(2), F(3)
            return ["P", "X", "D", "F"], [0, 1, 2, 3]

        _patch(_best_window_for_acronym, _initials_seq=fake_initials_seq)

        tokens = ["Ptok", "Xtok", "Dtok", "Ftok"]
        out = _best_window_for_acronym(tokens, "PDF")
        assert out is not None
        tok_s, tok_e, hits = out
        assert (tok_s, tok_e) == (0, 3)
        # Token 1 didn't contribute a matched initial
        assert hits == {0, 2, 3}


class TestBestWindowForAcronymIntegration:
    def test_basic_pdf(self):
        tokens = ["Portable", "Document", "Format"]
        out = _best_window_for_acronym(tokens, "PDF")
        assert out == (0, 2, {0, 1, 2})

    def test_compound_parts_contribute_multiple_initials(self):
        # Expect per-part initials (e.g., "Graphics/Processing" → 'G','P' from the same token index)
        tokens = ["High", "Performance", "Graphics/Processing", "Unit"]
        out = _best_window_for_acronym(tokens, "GPU")
        # Minimal window should be tokens 2..3; both 'G' and 'P' came from token index 2, 'U' from 3
        assert out == (2, 3, {2, 3})

    def test_acronym_with_punct_is_filtered(self):
        tokens = ["Graphics", "Processing", "Unit"]
        out = _best_window_for_acronym(tokens, "g-p_u")  # → "GPU"
        assert out == (0, 2, {0, 1, 2})

    def test_no_match_returns_none(self):
        tokens = ["Portable", "Document", "Format"]
        assert _best_window_for_acronym(tokens, "PFD") is None

    def test_prefers_shorter_window_over_earlier_longer_one(self):
        tokens = ["Portable", "Xtra", "Document", "Format", "Portable", "Document", "Format"]
        out = _best_window_for_acronym(tokens, "PDF")
        assert out == (4, 6, {4, 5, 6})


class TestTightenLabelByAcronymUnit:
    def test_empty_inputs(self):
        # If raw_label is empty, it should be returned as ""
        # If acronym is empty, it should return raw_label as-is
        assert tighten_label_by_acronym("", "PDF") == ""
        assert tighten_label_by_acronym("Anything", "") == "Anything"

    def test_tokenize_returns_empty_uses_fallback(self, _patch):
        # canonicalize should be called and then fallback through collapse_ws + strip_trailing_punct
        calls = {}

        def fake_canon(s):
            calls["canon"] = s
            return s

        def fake_tokenize(_):
            return []

        def fake_collapse(s):
            calls["collapse"] = s
            return "Foo Bar  "

        def fake_strip(s):
            calls["strip"] = s
            return "Foo Bar"

        _patch(
            tighten_label_by_acronym,
            canonicalize=fake_canon,
            _tokenize_preserve=fake_tokenize,
            collapse_ws=fake_collapse,
            strip_trailing_punct_str=fake_strip,
        )

        out = tighten_label_by_acronym("  Foo   Bar...  ", "PDF")
        assert out == "Foo Bar"
        assert calls == {"canon": "  Foo   Bar...  ", "collapse": "  Foo   Bar...  ", "strip": "Foo Bar  "}

    def test_no_window_found_fallback_respects_keep_case(self, _patch):
        _patch(
            tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=lambda s: ["one", "two"],
            _best_window_for_acronym=lambda toks, acr: None,
            collapse_ws=lambda s: "Foo Bar  ",
            strip_trailing_punct=lambda s: "Foo Bar",
        )
        assert tighten_label_by_acronym("Foo   Bar... ", "PDF", keep_case=True) == "Foo Bar"
        assert tighten_label_by_acronym("Foo   Bar... ", "PDF", keep_case=False) == "foo bar"

    def test_prunes_to_matched_tokens_and_bridges(self, _patch):
        # Simulate tokens + best window + bridges kept inside the chosen span
        tokens = ["Other", "Portable", "of", "Document", "Format", "spec"]

        def fake_tokenize(s):
            return tokens

        # window i..j = 1..4; matched tokens at indices 1,3,4 (PDF), token 2 "of" is a bridge
        def fake_win(toks, acr):
            return 1, 4, {1, 3, 4}

        _patch(
            tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=fake_tokenize,
            _best_window_for_acronym=fake_win,
            collapse_ws=lambda s: s,
            strip_trailing_punct=lambda s: s.rstrip(". "),
        )

        out = tighten_label_by_acronym(
            "Other Portable of Document Format.",
            "PDF",
            bridges={"of"},
            keep_case=True,
        )
        assert out == "Portable of Document Format"

    def test_edge_case_pruning_removes_everything_keep_original_span(self, _patch):
        # If hits set ends up empty (pathological), keep the original span tokens
        tokens = ["foo", "bar", "baz"]

        def fake_tokenize(s):
            return tokens

        def fake_win(toks, acr):
            return 0, 1, set()  # span 0..1, but no hits

        _patch(
            tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=fake_tokenize,
            _best_window_for_acronym=fake_win,
            collapse_ws=lambda s: s,
            strip_trailing_punct=lambda s: s,
        )

        out = tighten_label_by_acronym("foo bar baz", "FB", keep_case=True)
        assert out == "foo bar"  # original span preserved

    def test_keep_case_false_on_success(self, _patch):
        tokens = ["Graphics", "Processing", "Unit"]

        def fake_tokenize(s):
            return tokens

        def fake_win(toks, acr):
            return 0, 2, {0, 1, 2}

        _patch(
            tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=fake_tokenize,
            _best_window_for_acronym=fake_win,
            collapse_ws=lambda s: s,
            strip_trailing_punct=lambda s: s,
        )

        out = tighten_label_by_acronym("Graphics Processing Unit", "GPU", keep_case=False)
        assert out == "graphics processing unit"


class TestTightenLabelByAcronymIntegration:
    def test_preserves_numeric_leading_token_in_pruned_phrase(self):
        # This is the *real* path that bit you in your E2E:
        # tighten_label_by_acronym() must include numeric-leading neighbours in the chosen window.
        raw = "3M Portable format"
        got = tighten_label_by_acronym(raw, "PF", bridges=set(), keep_case=True)

        assert got.startswith("3M "), got
        assert got == "3M Portable format"

    def test_keeps_bridge_words_for_common_phrases(self):
        raw = "Terms of Service"
        got = tighten_label_by_acronym(raw, "TOS", bridges={"of"}, keep_case=True)

        assert got == "Terms of Service"

    def test_basic_pdf_with_bridges(self):
        # Intentionally include a bridge word "of" and a trailing period
        raw = "Other Portable of Document Format."
        out = tighten_label_by_acronym(
            raw,
            "PDF",
            # make behavior explicit/deterministic
            bridges={"of"},
            keep_case=True,
        )
        assert out == "Portable of Document Format"

    def test_compound_token_gpu(self):
        # "Graphics/Processing" should contribute both G and P (per-part initials),
        # so the minimal span is the full phrase.
        raw = "Graphics/Processing Unit (spec)"
        out = tighten_label_by_acronym(
            raw,
            "GPU",
            bridges=set(),
            keep_case=True,
        )
        assert out == "Graphics/Processing Unit"

    def test_fallback_when_no_window_found(self):
        # Acronym letters don't align; fall back to canon + collapse + strip.
        raw = "  Foo   Bar...  "
        out = tighten_label_by_acronym(
            raw,
            "XYZ",
            bridges=set(),
            keep_case=True,
        )
        assert out == "Foo Bar"

    def test_keep_case_false_integration(self):
        raw = "Graphics Processing Unit"
        out = tighten_label_by_acronym(
            raw,
            "GPU",
            bridges=set(),
            keep_case=False,
        )
        assert out == "graphics processing unit"


class TestInitialsRuleBenefit:
    def test_lowercase_span_retained_for_split_acronym(self):
        # Without initials rule, many implementations collapse to "acquisition"
        s = "cost per acquisition"
        # simulate flow: tighten_definition_span -> tighten_label_by_acronym
        # span function likely returns the whole tail (lowercase), then cleaner kicks in
        from document_resolution.nlp.extraction.acronyms.anchored.normalise import tighten_definition_span

        tail = tighten_definition_span(s)
        out = tighten_label_by_acronym(tail, "C/A", bridges={"per", "of", "and", "&"})
        assert out == "cost per acquisition"  # passes only with initials-in-order tweak
