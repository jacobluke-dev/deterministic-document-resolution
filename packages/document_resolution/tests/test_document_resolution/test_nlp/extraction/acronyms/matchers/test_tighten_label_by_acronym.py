import pytest

from document_resolution.nlp.extraction.acronyms.matchers.tighten import (
    _phrase_from_best_window,
    _try_split_acronym_initials_window,
    tighten_label_by_acronym,
)


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
        # This should fail unless there's tokens starting with a digit in the sequence.
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
        assert tighten_label_by_acronym("Foo Bar... ", "PDF", keep_case=True) == "Foo Bar"
        assert tighten_label_by_acronym("Foo Bar... ", "PDF", keep_case=False) == "foo bar"

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
        # This is the *real* path in the E2E:
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
