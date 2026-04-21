import pytest
from document_resolution.nlp.extraction.acronyms.matchers.tighten import (
    _best_window_for_acronym,
    _numeric_leading,
    _tokenize_preserve,
)


class TestTokenizePreserve:

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
            # ---- non-ascii digits  ----
            ("١٢V", True),  # Arabic-Indic digit '١' isdigit() == True
        ],
    )
    def test_numeric_leading(self, tok: str, expected: bool):
        assert _numeric_leading(tok) is expected


class TestBestWindowForAcronymUnit:
    def test_returns_none_when_acronym_has_no_alnum(self, _patch):
        # Even if initials exist, empty A should short-circuit
        _patch(_best_window_for_acronym, initials_seq=lambda t, s: (["A"], [0]))
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
        def fake_initials_seq(tokens, expand_allcaps):
            # letters: P(0), X(1), D(2), F(3)
            return ["P", "X", "D", "F"], [0, 1, 2, 3]

        _patch(_best_window_for_acronym, initials_seq=fake_initials_seq)

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
