import pytest

from plainera_unacronym.nlp.extraction.matchers.defs import find_parenthetical_longform_after_acr


class TestFindParentheticalLongformAfterAcrUnit:
    def test_no_parenthesized_match_returns_empty(self, _patch, dummy_cfg):
        # Patching anyway to prove independence; they won't be called
        _patch(
            find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            _initials_match=lambda acr, phrase: True,
        )
        cfg = dummy_cfg()
        assert find_parenthetical_longform_after_acr("no parens here", cfg, acr="PDF") == []

    def test_requires_letters_gate(self, _patch, dummy_cfg):
        calls = {}

        def spy_has_letters(s):
            calls["raw"] = s
            return False  # force gate fail

        _patch(
            find_parenthetical_longform_after_acr,
            has_letters=spy_has_letters,
            tighten_definition_span=lambda s: "IGNORED",
            normalize_definition=lambda s: "IGNORED",
            _initials_match=lambda acr, phrase: True,
        )
        cfg = dummy_cfg()
        snip = "   (1234) tail"
        assert find_parenthetical_longform_after_acr(snip, cfg, acr="X") == []
        # ensure we passed the raw inner text to _has_letters
        assert calls["raw"] == "1234"

    def test_normalize_pipeline_and_span_preserved(self, _patch, dummy_cfg):
        seen = {}

        def fake_tighten(s):
            seen["tighten_in"] = s
            return " Foo   Bar... "

        def fake_normalize(s):
            seen["normalize_in"] = s
            return "Foo Bar"  # collapsed + stripped

        _patch(
            find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=fake_tighten,
            normalize_definition=fake_normalize,
            _initials_match=lambda acr, phrase: True,
        )
        cfg = dummy_cfg()
        raw = " noisy    RAW "
        snip = f"  ({raw}) and more"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="FB", require_initials_match=False)
        assert len(out) == 1
        m = out[0]
        # Output definition is the normalized value
        assert m.definition == "Foo Bar"

        # Indices hug the content (no inner padding)
        assert snip[m.def_start:m.def_end] == raw.strip()

        # Verify pipeline call args: we now feed the *tight* captured def
        assert seen["tighten_in"] == raw.strip()

        # And normalize is called with whatever tighten returned
        assert seen["normalize_in"] == " Foo   Bar... "

    def test_require_initials_match_guard_true_allows(self, _patch, dummy_cfg):
        seen = {}

        def fake_tighten(s):
            seen["tighten_in"] = s
            return s  # or " Foo   Bar... " if test normalization too

        def fake_normalize(s):
            seen["normalize_in"] = s
            return s.strip()

        _patch(
            find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=fake_tighten,
            normalize_definition=fake_normalize,
        )

        cfg = dummy_cfg()
        snip = "(Portable Document Format)"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

        # (Optional) verify the pipeline inputs were what can be expected
        assert seen["tighten_in"] == "Portable Document Format"
        assert seen["normalize_in"] == "Portable Document Format"

    def test_require_initials_match_guard_false_blocks(self, _patch, dummy_cfg):
        _patch(
            find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s
        )
        cfg = dummy_cfg()
        snip = "Portable Document Format"
        assert find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=True) == []

    def test_max_chars_respected(self, _patch, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=3)
        assert find_parenthetical_longform_after_acr("(Portable)", cfg, acr="P") == []


class TestFindLongformAfterAcrIntegration:
    def test_no_parenthesized_match_returns_empty(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "  not a parenthetical here"
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF") == []

    def test_requires_letters(self, dummy_cfg):
        # "(1234)" contains no letters, should be rejected
        cfg = dummy_cfg()
        snippet = "   (1234)   trailing"
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF") == []

    def test_respects_max_chars(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=5)
        # "Portable" exceeds max=5, so regex won't match at all
        snippet = " (Portable) "
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="P") == []

        # Fits within the limit
        cfg2 = dummy_cfg(max_phrase_chars=12)
        snippet2 = " (Portable) "
        out = find_parenthetical_longform_after_acr(snippet2, cfg2, acr="P")
        assert len(out) == 1
        assert out[0].definition == "Portable"

    def test_normalization_pipeline_applied(self, dummy_cfg):
        cfg = dummy_cfg()
        # Extra whitespace + trailing punctuation → normalized by tighten_definition_span + normalize_definition
        snippet = "   (  Portable   Document   Format...   )  "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF")
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_require_initials_match_guard_allows_good_match(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = " (Graphics Processing Unit) "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="GPU", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_require_initials_match_guard_blocks_bad_match(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = " (Portable Document Format) "
        # Wrong order: 'PFD' does not fit initials 'PDF'
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PFD", require_initials_match=True)
        assert out == []

    def test_disable_require_initials_match_guard(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = " (Portable Document Format) "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PFD", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_def_span_indices_are_correct(self, dummy_cfg):
        cfg = dummy_cfg()
        raw_def = "Portable Document Format"
        snippet = f"   ({raw_def}) and more"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF")
        assert len(out) == 1
        m = out[0]
        # Ensure the span points exactly to the definition characters within snippet
        assert snippet[m.def_start:m.def_end] == raw_def

    def test_forward_form_pdf(self, dummy_cfg):
        cfg = dummy_cfg()
        # Caller slices snippet to start at acr_end; we simulate by starting at '('
        snippet = "(Portable Document Format) please proceed"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        item = out[0]
        assert item.definition == "Portable Document Format"
        assert snippet[item.def_start:item.def_end] == "Portable Document Format"

    def test_whitespace_and_punct_cleaned(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "   (  Graphics    Processing  Unit... ) more"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="GPU", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_non_alpha_initial_words_are_ignored_in_require_initials_match(self, dummy_cfg):
        cfg = dummy_cfg()
        # Non-alpha-leading words are ignored; we also avoid TitleCase tail trimming
        snippet = "(3M Portable format)"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PF", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "3M Portable format"
        # PF != PDF
        out2 = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert out2 == []

    def test_require_require_initials_match_false_allows_generic_parenthetical(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "(see below for details)"
        # Contains letters, normalizes to same text; pass when require_initials_match disabled
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="ANY", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "see below for details"

    def test_respects_max_phrase_chars(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=10)
        snippet = "(Hypertext Transfer Protocol)"
        # Longer than max → no match at all
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="HTTP") == []


def _patch(monkeypatch, func, **replacements):
    g = func.__globals__
    for name, impl in replacements.items():
        monkeypatch.setitem(g, name, impl)
