import pytest
from plainera_unacronym.nlp.extraction.tighten import (
    _best_window_for_acronym,
    _initials_seq,
    _match_from,
    _split_compound,
    _tokenize_preserve,
    tighten_label_by_acronym,
)


class TestSplitCompound:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("GPU", ["GPU"]),                              # no split
            ("read-only", ["read", "only"]),               # hyphen
            ("C/CPP", ["C", "CPP"]),                       # slash
            ("U.S.A.", ["U", "S", "A", ""]),               # NOTE: trailing '' would be filtered out by impl
            ("Foo.Bar", ["Foo", "Bar"]),                   # dot
            ("R&D", ["R", "D"]),                           # ampersand
            ("A-B/C.D&E", ["A", "B", "C", "D", "E"]),      # mixed delimiters
            ("--GPU--", ["", "GPU", ""]),                  # leading/trailing delimiters (empties dropped)
            ("a--b", ["a", "", "b"]),                      # repeated delimiter (middle '' dropped)
            ("", []),                                      # empty token -> []
            ("----", []),                                  # only delimiters -> []
            ("co-op", ["co", "op"]),                       # splits on '-'
            ("Queen’s", ["Queen’s"]),                      # apostrophe does not split
            ("snake_case", ["snake_case"]),                # underscore does not split
            ("β-blocker", ["β", "blocker"]),               # Unicode letters + hyphen
            ("3D-Print", ["3D", "Print"]),                 # alnum pieces
            ("A&B&C", ["A", "B", "C"]),                    # multiple &
            ("v1.2.3", ["v1", "2", "3"]),                  # dot with numbers
            ("HyperText", ["Hyper", "Text"])
        ],
    )
    def test_split_various(self, token, expected):
        # Filtered empties: replicate function’s behavior for cases where parametrization shows '' parts
        out = _split_compound(token)
        assert out == [p for p in expected if p]

    def test_repeated_mixed_delimiters(self):
        token = "a--b///c..d&&e"
        assert _split_compound(token) == ["a", "b", "c", "d", "e"]


class TestTokenizePreserve:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", []),                                    # empty
            ("   \t\n", []),                             # whitespace only
            ("Hello", ["Hello"]),                        # simple word
            ("Hello world", ["Hello", "world"]),         # spaces split
            ("O'Reilly", ["O'Reilly"]),                  # straight apostrophe
            ("Queen’s Award", ["Queen’s", "Award"]),     # curly apostrophe
            ("R&D", ["R&D"]),                            # ampersand preserved
            ("C/C++", ["C/C"]),                          # slash ok; '+' not allowed -> dropped
            ("U.S.A.", ["U.S.A."]),                      # dots preserved inside token
            ("v1.2.3", ["v1.2.3"]),                      # mixed digits + dots
            ("foo_bar", ["foo", "bar"]),                 # underscore splits (not allowed)
            ("Email a.b@c.com now", ["Email", "a.b", "c.com", "now"]),  # '@' splits
            ("dash-separated-words", ["dash-separated-words"]),         # hyphen preserved
            ("mix&match/okay.now", ["mix&match/okay.now"]),             # combo delimiters preserved
            ("(Portable) Document, Format!", ["Portable", "Document", "Format"]),  # strip punctuation
            ("β-blocker", ["-blocker"]),                 # leading non-ASCII splits; hyphen+word captured
            ("Заказ-123", ["-123"]),                     # Cyrillic splits; digits captured (ASCII-only)
            ("3/4-inch", ["3/4-inch"]),                  # digits + slash + hyphen
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


def _patch(monkeypatch, func, **replacements):
    g = func.__globals__
    for name, impl in replacements.items():
        monkeypatch.setitem(g, name, impl)


class TestInitialsSeqUnit:
    def test_basic_three_tokens(self, monkeypatch):
        # Force a single part per token; check letters+owners map 1:1 to tokens
        _patch(
            monkeypatch, _initials_seq,
            _split_compound=lambda tok: [tok],
            re=__import__("re"),
        )
        tokens = ["Portable", "Document", "Format"]
        letters, owners = _initials_seq(tokens, stopwords=set())
        assert letters == ["P", "D", "F"]
        assert owners == [0, 1, 2]

    def test_stopwords_skip_whole_token(self, monkeypatch):
        _patch(
            monkeypatch, _initials_seq,
            _split_compound=lambda tok: [tok],
            re=__import__("re"),
        )
        tokens = ["Read", "Only", "Memory"]
        letters, owners = _initials_seq(tokens, stopwords={"only"})
        assert letters == ["R", "M"]
        assert owners == [0, 2]

    def test_compound_parts_all_count_but_owner_is_token_index(self, monkeypatch):
        # Simulate "C++" -> ["C", "Plus", "Plus"] from the *same* token index
        parts_map = {
            "C++": ["C", "Plus", "Plus"],
            "GPU": ["GPU"],
        }
        _patch(
            monkeypatch, _initials_seq,
            _split_compound=lambda tok: parts_map[tok],
            re=__import__("re"),
        )
        tokens = ["C++", "GPU"]
        letters, owners = _initials_seq(tokens, stopwords=set())
        assert letters == ["C", "P", "P", "G"]
        assert owners == [0, 0, 0, 1]

    def test_tokens_with_no_alnum_parts_are_ignored(self, monkeypatch):
        # Parts with no [A-Za-z0-9] yield no initials
        _patch(
            monkeypatch, _initials_seq,
            _split_compound=lambda tok: ["—", "…"],  # emdash, ellipsis
            re=__import__("re"),
        )
        tokens = ["—…"]
        letters, owners = _initials_seq(tokens, stopwords=set())
        assert letters == []
        assert owners == []

    def test_stopword_checked_before_split(self, monkeypatch):
        # If the whole token is a stopword, we skip it entirely (no splitting)
        called = {"split": 0}

        def spy_split(tok):
            called["split"] += 1
            return [tok]  # would have produced something if not skipped

        _patch(monkeypatch, _initials_seq, _split_compound=spy_split, re=__import__("re"))
        tokens = ["and-or", "Useful"]
        letters, owners = _initials_seq(tokens, stopwords={"and-or"})
        assert letters == ["U"]
        assert owners == [1]
        # Ensure split was *not* called for the stopword token
        assert called["split"] == 1  # only for "Useful"


class TestInitialsSeqIntegration:
    def test_compound_splitting_and_digits(self):
        tokens = ["3/4-inch", "co-op", "R&D", "v1.2.3"]
        letters, owners = _initials_seq(tokens, stopwords=set())
        # Expected from real _split_compound:
        # "3/4-inch" -> ["3","4","inch"]      -> 3,4,I (owners 0,0,0)
        # "co-op"    -> ["co","op"]           -> C,O   (owners 1,1)
        # "R&D"      -> ["R","D"]             -> R,D   (owners 2,2)
        # "v1.2.3"   -> ["v1","2","3"]        -> V,2,3 (owners 3,3,3)
        assert letters == ["3", "4", "I", "C", "O", "R", "D", "V", "2", "3"]
        assert owners  == [ 0,   0,   0,   1,   1,   2,   2,   3,   3,   3 ]

    def test_stopwords_filter_entire_tokens(self):
        tokens = ["Read", "Only", "Memory", "of", "Computers"]
        letters, owners = _initials_seq(tokens, stopwords={"of", "and"})
        # "of" is skipped entirely; others contribute initials
        assert letters == ["R", "O", "M", "C"]
        assert owners  == [ 0,   1,   2,   4 ]

    def test_unicode_letters_in_tokens(self):
        tokens = ["β-blocker", "Ångström", "GPU"]
        letters, owners = _initials_seq(tokens, stopwords=set())
        # "β-blocker" -> parts ["β","blocker"] → first alpha is 'β' (Unicode) → 'Β' (Greek beta uppercase)
        # 2nd B is from blocker
        # "Ångström"  -> first alpha is 'Å'     → 'Å'
        # "GPU"       -> 'G'
        assert letters == ["Β", "B", "Å", "G"]  # Python uppercases β to Β
        assert owners == [0, 0, 1, 2]


class TestMatchFrom:
    def test_exact_match_from_zero(self):
        letters = list("PDFX")
        acronym = list("PDF")
        end, used = _match_from(letters, acronym, 0)
        assert end == 3
        assert used == [0, 1, 2]

    def test_no_match(self):
        letters = list("PFX")
        acronym = list("PDF")
        assert _match_from(letters, acronym, 0) is None

    def test_start_offset(self):
        letters = list("APDFZ")
        acronym = list("PDF")
        # Starting at 1 should match P(1), D(2), F(3) → end index 4
        end, used = _match_from(letters, acronym, 1)
        assert end == 4
        assert used == [1, 2, 3]

    def test_start_offset_scans_forward_not_strict_anchor(self):
        letters = list("ABCABC")
        acronym = list("ABC")
        # From start=0 it matches at [0,1,2]
        assert _match_from(letters, acronym, 0) == (3, [0, 1, 2])
        # From start=1 it can skip to the next 'A' and match at [3,4,5]
        assert _match_from(letters, acronym, 1) == (6, [3, 4, 5])

    def test_greedy_end_index_is_exclusive(self):
        letters = list("AXBYCZD")
        acronym = list("ABCD")
        # Match A(0), B(2), C(4), D(6) → last match at 6; end should be 7
        end, used = _match_from(letters, acronym, 0)
        assert used == [0, 2, 4, 6]
        assert end == 7  # exclusive

    def test_letters_shorter_than_acronym(self):
        letters = list("PD")
        acronym = list("PDF")
        assert _match_from(letters, acronym, 0) is None

    def test_empty_acronym_matches_immediately(self):
        letters = list("ANY")
        acronym = []  # empty target
        end, used = _match_from(letters, acronym, 2)
        # With empty acronym, loop doesn't run; returns (start, [])
        assert (end, used) == (2, [])

    def test_start_past_end_returns_none(self):
        letters = list("PDF")
        acronym = list("P")
        assert _match_from(letters, acronym, 5) is None

    def test_case_sensitive(self):
        letters = list("Pdf")
        acronym = list("PDF")
        # Exact equality is required; pipeline should uppercase upstream
        assert _match_from(letters, acronym, 0) is None


class TestBestWindowForAcronymUnit:
    def test_returns_none_when_acronym_has_no_alnum(self, monkeypatch):
        # Even if initials exist, empty A should short-circuit
        _patch(monkeypatch, _best_window_for_acronym, _initials_seq=lambda t, s: (["A"], [0]))
        assert _best_window_for_acronym(["any"], "--__--", stopwords=set()) is None

    def test_returns_none_when_no_letters(self, monkeypatch):
        # No initials generated → None
        _patch(monkeypatch, _best_window_for_acronym, _initials_seq=lambda t, s: ([], []))
        assert _best_window_for_acronym(["Portable", "Document"], "PD", stopwords=set()) is None

    def test_picks_shortest_window_and_keeps_first_on_tie(self, monkeypatch):
        # Provide deterministic initials and owners; use real _match_from
        def fake_initials_seq(tokens, stopwords):
            # letters index: 0:X, 1:P, 2:D, 3:F, 4:P, 5:D, 6:F
            return ["X", "P", "D", "F", "P", "D", "F"], [0, 1, 2, 3, 4, 5, 6]

        _patch(monkeypatch, _best_window_for_acronym, _initials_seq=fake_initials_seq)

        tokens = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]
        out = _best_window_for_acronym(tokens, "PDF", stopwords=set())
        # Two equally short windows exist: tokens [1..3] and [4..6]; function keeps the first
        assert out == (1, 3, {1, 2, 3})

    def test_hits_contains_only_tokens_that_contributed_letters(self, monkeypatch):
        # Make a window covering tokens 0..3, but ensure only tokens 0,2,3 supply matched initials
        def fake_initials_seq(tokens, stopwords):
            # letters: P(0), X(1), D(2), F(3)
            return ["P", "X", "D", "F"], [0, 1, 2, 3]

        _patch(monkeypatch, _best_window_for_acronym, _initials_seq=fake_initials_seq)

        tokens = ["Ptok", "Xtok", "Dtok", "Ftok"]
        out = _best_window_for_acronym(tokens, "PDF", stopwords=set())
        assert out is not None
        tok_s, tok_e, hits = out
        assert (tok_s, tok_e) == (0, 3)
        # Token 1 didn't contribute a matched initial
        assert hits == {0, 2, 3}


STOP = {"the", "of", "and", "for"}  # illustrative stopwords set for tests


class TestBestWindowForAcronymIntegration:
    def test_basic_pdf(self):
        tokens = ["Portable", "Document", "Format"]
        out = _best_window_for_acronym(tokens, "PDF", stopwords=set())
        assert out == (0, 2, {0, 1, 2})

    def test_ignores_stopwords_and_finds_min_window(self):
        tokens = ["the", "Portable", "Document", "of", "Format"]
        out = _best_window_for_acronym(tokens, "PDF", stopwords=STOP)
        # Stopwords are skipped when creating initials → window [1..4] but hits only on 1,2,4
        assert out == (1, 4, {1, 2, 4})

    def test_compound_parts_contribute_multiple_initials(self):
        # Expect per-part initials (e.g., "Graphics/Processing" → 'G','P' from the same token index)
        tokens = ["High", "Performance", "Graphics/Processing", "Unit"]
        out = _best_window_for_acronym(tokens, "GPU", stopwords=set())
        # Minimal window should be tokens 2..3; both 'G' and 'P' came from token index 2, 'U' from 3
        assert out == (2, 3, {2, 3})

    def test_acronym_with_punct_is_filtered(self):
        tokens = ["Graphics", "Processing", "Unit"]
        out = _best_window_for_acronym(tokens, "g-p_u", stopwords=set())  # → "GPU"
        assert out == (0, 2, {0, 1, 2})

    def test_no_match_returns_none(self):
        tokens = ["Portable", "Document", "Format"]
        assert _best_window_for_acronym(tokens, "PFD", stopwords=set()) is None

    def test_prefers_shorter_window_over_earlier_longer_one(self):
        tokens = ["Portable", "Xtra", "Document", "Format", "Portable", "Document", "Format"]
        out = _best_window_for_acronym(tokens, "PDF", stopwords=set())
        assert out == (4, 6, {4, 5, 6})


class TestTightenLabelByAcronymUnit:
    def test_empty_inputs(self, monkeypatch):
        # If raw_label is empty, it should be returned as ""
        # If acronym is empty, it should return raw_label as-is
        assert tighten_label_by_acronym("", "PDF") == ""
        assert tighten_label_by_acronym("Anything", "") == "Anything"

    def test_tokenize_returns_empty_uses_fallback(self, monkeypatch):
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
            monkeypatch, tighten_label_by_acronym,
            canonicalize=fake_canon,
            _tokenize_preserve=fake_tokenize,
            collapse_ws=fake_collapse,
            strip_trailing_punct=fake_strip,
        )

        out = tighten_label_by_acronym("  Foo   Bar...  ", "PDF")
        assert out == "Foo Bar"
        assert calls == {"canon": "  Foo   Bar...  ", "collapse": "  Foo   Bar...  ", "strip": "Foo Bar  "}

    def test_no_window_found_fallback_respects_keep_case(self, monkeypatch):
        _patch(
            monkeypatch, tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=lambda s: ["one", "two"],
            _best_window_for_acronym=lambda toks, acr, stop: None,
            collapse_ws=lambda s: "Foo Bar  ",
            strip_trailing_punct=lambda s: "Foo Bar",
        )
        assert tighten_label_by_acronym("Foo   Bar... ", "PDF", keep_case=True) == "Foo Bar"
        assert tighten_label_by_acronym("Foo   Bar... ", "PDF", keep_case=False) == "foo bar"

    def test_prunes_to_matched_tokens_and_bridges(self, monkeypatch):
        # Simulate tokens + best window + bridges kept inside the chosen span
        tokens = ["Other", "Portable", "of", "Document", "Format", "spec"]
        def fake_tokenize(s): return tokens
        # window i..j = 1..4; matched tokens at indices 1,3,4 (PDF), token 2 "of" is a bridge
        def fake_win(toks, acr, stop): return 1, 4, {1, 3, 4}

        _patch(
            monkeypatch, tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=fake_tokenize,
            _best_window_for_acronym=fake_win,
            collapse_ws=lambda s: s,
            strip_trailing_punct=lambda s: s.rstrip(". "),
        )

        out = tighten_label_by_acronym(
            "Other Portable of Document Format.", "PDF",
            stopwords=set(), bridges={"of"}, keep_case=True,
        )
        assert out == "Portable of Document Format"

    def test_edge_case_pruning_removes_everything_keep_original_span(self, monkeypatch):
        # If hits set ends up empty (pathological), keep the original span tokens
        tokens = ["foo", "bar", "baz"]
        def fake_tokenize(s): return tokens
        def fake_win(toks, acr, stop): return 0, 1, set()  # span 0..1, but no hits

        _patch(
            monkeypatch, tighten_label_by_acronym,
            canonicalize=lambda s: s,
            _tokenize_preserve=fake_tokenize,
            _best_window_for_acronym=fake_win,
            collapse_ws=lambda s: s,
            strip_trailing_punct=lambda s: s,
        )

        out = tighten_label_by_acronym("foo bar baz", "FB", keep_case=True)
        assert out == "foo bar"  # original span preserved

    def test_keep_case_false_on_success(self, monkeypatch):
        tokens = ["Graphics", "Processing", "Unit"]
        def fake_tokenize(s): return tokens
        def fake_win(toks, acr, stop): return 0, 2, {0, 1, 2}

        _patch(
            monkeypatch, tighten_label_by_acronym,
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
            stopwords=set(),         # make behavior explicit/deterministic
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
            stopwords=set(),
            bridges=set(),
            keep_case=True,
        )
        assert out == "Graphics/Processing Unit"

    def test_fallback_when_no_window_found(self):
        # Acronym letters don't align; fall back to canon + collapse + strip.
        raw = "  Foo   Bar...  "
        out = tighten_label_by_acronym(
            raw, "XYZ",
            stopwords=set(),
            bridges=set(),
            keep_case=True,
        )
        assert out == "Foo Bar"

    def test_keep_case_false_integration(self):
        raw = "Graphics Processing Unit"
        out = tighten_label_by_acronym(
            raw, "GPU",
            stopwords=set(),
            bridges=set(),
            keep_case=False,
        )
        assert out == "graphics processing unit"

    def test_stopwords_do_not_contribute_initials_but_bridges_keep_them(self):
        # "of" is a stopword for initials, but we still want it in the final phrase when inside span
        raw = "Portable of Document Format"
        out = tighten_label_by_acronym(
            raw, "PDF",
            stopwords={"of"},        # ignored in initials
            bridges={"of"},          # kept in the pruned phrase if inside span
            keep_case=True,
        )
        assert out == "Portable of Document Format"
