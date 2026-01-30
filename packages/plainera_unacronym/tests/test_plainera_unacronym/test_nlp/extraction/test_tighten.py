import pytest

from plainera_unacronym.nlp.extraction.matchers.tighten import (
    _best_window_for_acronym,
    _tokenize_preserve,
    tighten_label_by_acronym,
)


class TestTokenizePreserve:
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

        def fake_tokenize(s): return tokens

        # window i..j = 1..4; matched tokens at indices 1,3,4 (PDF), token 2 "of" is a bridge
        def fake_win(toks, acr, stop): return 1, 4, {1, 3, 4}

        _patch(
            tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=fake_tokenize,
            _best_window_for_acronym=fake_win,
            collapse_ws=lambda s: s,
            strip_trailing_punct=lambda s: s.rstrip(". "),
        )

        out = tighten_label_by_acronym(
            "Other Portable of Document Format.", "PDF",
             bridges={"of"}, keep_case=True,
        )
        assert out == "Portable of Document Format"

    def test_edge_case_pruning_removes_everything_keep_original_span(self, _patch):
        # If hits set ends up empty (pathological), keep the original span tokens
        tokens = ["foo", "bar", "baz"]

        def fake_tokenize(s): return tokens

        def fake_win(toks, acr, stop): return 0, 1, set()  # span 0..1, but no hits

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

        def fake_tokenize(s): return tokens

        def fake_win(toks, acr, stop): return 0, 2, {0, 1, 2}

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
    def test_basic_pdf_with_bridges(self):
        # Intentionally include a bridge word "of" and a trailing period
        raw = "Other Portable of Document Format."
        out = tighten_label_by_acronym(
            raw, "PDF",
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
            raw, "GPU",

            bridges=set(),
            keep_case=True,
        )
        assert out == "Graphics/Processing Unit"

    def test_fallback_when_no_window_found(self):
        # Acronym letters don't align; fall back to canon + collapse + strip.
        raw = "  Foo   Bar...  "
        out = tighten_label_by_acronym(
            raw, "XYZ",

            bridges=set(),
            keep_case=True,
        )
        assert out == "Foo Bar"

    def test_keep_case_false_integration(self):
        raw = "Graphics Processing Unit"
        out = tighten_label_by_acronym(
            raw, "GPU",

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
        from plainera_unacronym.nlp.extraction.anchored.normalise import (
            tighten_definition_span
        )
        tail = tighten_definition_span(s)
        out = tighten_label_by_acronym(tail, "C/A", bridges={"per", "of", "and", "&"})
        assert out == "cost per acquisition"  # passes only with initials-in-order tweak
