import pytest

from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.common.shared import has_letters
from plainera_unacronym.nlp.extraction.matchers.defs import (
    find_parenthetical_longform_after_acr,
    find_parenthetical_longform_before_acr,
)


# TODO needs mergging with other has_letters tests
class TestHasLetters:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("", False),  # empty
            ("   \t", False),  # whitespace only
            ("123456", False),  # digits
            ("--._", False),  # punctuation/symbols
            ("\u0301", False),  # combining acute accent (not a letter)
            ("🧠💡", False),  # emoji
            ("A", True),  # ASCII letter
            ("abc123", True),  # mixed alnum
            ("42 is the answer", True),  # sentence with letters
            ("Straße", True),  # Latin letter ß
            ("Ångström", True),  # Latin with diacritics
            ("中文", True),  # CJK
            ("Ж9", True),  # Cyrillic + digit
            ("β-blocker", True),  # Greek + hyphen
        ],
    )
    def test_various_inputs(self, s, expected):
        assert has_letters(s) is expected

    def test_long_string_performance_smoke(self):
        s = "1234567" * 1000 + "X" + "!" * 1000
        assert has_letters(s) is True


def _msg(acr, phrase):
    return f"acr={acr!r}, phrase={phrase!r}"

# TODO merge initials_match function
class Test_require_initials_matchOk:
    @pytest.mark.parametrize(
        "acr,phrase,expected",
        [
            # Exact contiguous match
            ("PDF", "Portable Document Format", True),
            # Case-insensitive, subsequence across words
            ("ROM", "Read Only Memory", True),
            ("NHS", "National Health Service", True),
            # ignores symbols
            ("g-p_u", "Graphics Processing Unit", True),
            # Order must be preserved
            ("PFD", "Portable Document Format", False),
            ("LLO", "Lots Of Llamas", False),  # initials "LOL": L, then L ok, but O after second L fails
            # Missing letters
            ("ABC", "Alpha Beta", False),
        ],
    )
    def test_basic(self, acr, phrase, expected):
        assert initials_match(acr, phrase) is expected, _msg(acr, phrase)

    def test_non_alpha_leading_words_handling(self):
        # Clarify behavior with non-alpha-leading words: they are ignored
        # Initials from this phrase: ["Portable", "Format"] -> "PF"
        phrase = "3M Portable 7-Document Format"
        assert initials_match("PF", phrase) is True
        assert initials_match("PDF", phrase) is False

    def test_empty_inputs(self):
        assert initials_match("", "anything at all") is True  # no letters to match
        assert initials_match("123-._", "anything at all") is True  # acronym has no letters
        assert initials_match("A", "") is False  # no initials available

    def test_unicode_letters(self):
        # Works with Unicode alpha; initials will include 'É', 'N', 'S'
        assert initials_match("ÉNS", "École Normale Supérieure") is True
        # ASCII 'E' won't match 'É' initial
        assert initials_match("ENS", "École Normale Supérieure") is False

    def test_repeated_letters(self):
        # initials "LOL" -> L, then O, then L : OK
        assert initials_match("LOL", "Lots Of Llamas") is True
        # initials "LOL": trying L, L, O fails on the final O (order)
        assert initials_match("LLO", "Lots Of Llamas") is False


class DummyCfg:
    def __init__(self, max_phrase_chars=80):
        self.max_phrase_chars = max_phrase_chars
        self.require_initials_match = False

class TestFindLongformAfterAcrIntegration:
    def test_no_parenthesized_match_returns_empty(self):
        cfg = DummyCfg()
        snippet = "  not a parenthetical here"
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF") == []

    def test_requires_letters(self):
        # "(1234)" contains no letters, should be rejected
        cfg = DummyCfg()
        snippet = "   (1234)   trailing"
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF") == []

    def test_respects_max_chars(self):
        cfg = DummyCfg(max_phrase_chars=5)
        # "Portable" exceeds max=5, so regex won't match at all
        snippet = " (Portable) "
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="P") == []

        # Fits within the limit
        cfg2 = DummyCfg(max_phrase_chars=12)
        snippet2 = " (Portable) "
        out = find_parenthetical_longform_after_acr(snippet2, cfg2, acr="P")
        assert len(out) == 1
        assert out[0].definition == "Portable"

    def test_normalization_pipeline_applied(self):
        cfg = DummyCfg()
        # Extra whitespace + trailing punctuation → normalized by tighten_definition_span + normalize_definition
        snippet = "   (  Portable   Document   Format...   )  "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF")
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_require_initials_match_guard_allows_good_match(self):
        cfg = DummyCfg()
        snippet = " (Graphics Processing Unit) "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="GPU", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_require_initials_match_guard_blocks_bad_match(self):
        cfg = DummyCfg()
        snippet = " (Portable Document Format) "
        # Wrong order: 'PFD' does not fit initials 'PDF'
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PFD", require_initials_match=True)
        assert out == []

    def test_disable_require_initials_match_guard(self):
        cfg = DummyCfg()
        snippet = " (Portable Document Format) "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PFD", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_def_span_indices_are_correct(self):
        cfg = DummyCfg()
        raw_def = "Portable Document Format"
        snippet = f"   ({raw_def}) and more"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF")
        assert len(out) == 1
        m = out[0]
        # Ensure the span points exactly to the definition characters within snippet
        assert snippet[m.def_start:m.def_end] == raw_def

    def test_forward_form_pdf(self):
        cfg = DummyCfg()
        # Caller slices snippet to start at acr_end; we simulate by starting at '('
        snippet = "(Portable Document Format) please proceed"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        item = out[0]
        assert item.definition == "Portable Document Format"
        assert snippet[item.def_start:item.def_end] == "Portable Document Format"

    def test_whitespace_and_punct_cleaned(self):
        cfg = DummyCfg()
        snippet = "   (  Graphics    Processing  Unit... ) more"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="GPU", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_non_alpha_initial_words_are_ignored_in_require_initials_match(self):
        cfg = DummyCfg()
        # Non-alpha-leading words are ignored; we also avoid TitleCase tail trimming
        snippet = "(3M Portable format)"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PF", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "3M Portable format"
        # PF != PDF
        out2 = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert out2 == []

    def test_require_require_initials_match_false_allows_generic_parenthetical(self):
        cfg = DummyCfg()
        snippet = "(see below for details)"
        # Contains letters, normalizes to same text; pass when require_initials_match disabled
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="ANY", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "see below for details"

    def test_respects_max_phrase_chars(self):
        cfg = DummyCfg(max_phrase_chars=10)
        snippet = "(Hypertext Transfer Protocol)"
        # Longer than max → no match at all
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="HTTP") == []


def _patch(monkeypatch, func, **replacements):
    g = func.__globals__
    for name, impl in replacements.items():
        monkeypatch.setitem(g, name, impl)


class TestFindParentheticalLongformAfterAcrUnit:
    def test_no_parenthesized_match_returns_empty(self, monkeypatch):
        # Patching anyway to prove independence; they won't be called
        _patch(
            monkeypatch, find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            _initials_match=lambda acr, phrase: True,
        )
        cfg = DummyCfg()
        assert find_parenthetical_longform_after_acr("no parens here", cfg, acr="PDF") == []

    def test_requires_letters_gate(self, monkeypatch):
        calls = {}
        def spy_has_letters(s):
            calls["raw"] = s
            return False  # force gate fail
        _patch(
            monkeypatch, find_parenthetical_longform_after_acr,
            has_letters=spy_has_letters,
            tighten_definition_span=lambda s: "IGNORED",
            normalize_definition=lambda s: "IGNORED",
            _initials_match=lambda acr, phrase: True,
        )
        cfg = DummyCfg()
        snip = "   (1234) tail"
        assert find_parenthetical_longform_after_acr(snip, cfg, acr="X") == []
        # ensure we passed the raw inner text to _has_letters
        assert calls["raw"] == "1234"

    def test_normalize_pipeline_and_span_preserved(self, monkeypatch):
        seen = {}
        def fake_tighten(s):
            seen["tighten_in"] = s
            return " Foo   Bar... "
        def fake_normalize(s):
            seen["normalize_in"] = s
            return "Foo Bar"  # collapsed + stripped
        _patch(
            monkeypatch, find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=fake_tighten,
            normalize_definition=fake_normalize,
            _initials_match=lambda acr, phrase: True,
        )
        cfg = DummyCfg()
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

    def test_require_initials_match_guard_true_allows(self, monkeypatch):
        seen = {}

        def fake_tighten(s):
            seen["tighten_in"] = s
            return s  # or " Foo   Bar... " if test normalization too

        def fake_normalize(s):
            seen["normalize_in"] = s
            return s.strip()

        _patch(
            monkeypatch, find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=fake_tighten,
            normalize_definition=fake_normalize,
        )

        cfg = DummyCfg()
        snip = "(Portable Document Format)"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

        # (Optional) verify the pipeline inputs were what can be expected
        assert seen["tighten_in"] == "Portable Document Format"
        assert seen["normalize_in"] == "Portable Document Format"

    def test_require_initials_match_guard_false_blocks(self, monkeypatch):
        _patch(
            monkeypatch, find_parenthetical_longform_after_acr,
            has_letters=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s
        )
        cfg = DummyCfg()
        snip = "Portable Document Format"
        assert find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=True) == []

    def test_max_chars_respected(self, monkeypatch):
        cfg = DummyCfg(max_phrase_chars=3)
        assert find_parenthetical_longform_after_acr("(Portable)", cfg, acr="P") == []


class TestFindParentheticalLongformBeforeAcrIntegration:
    def test_basic_forward_phrase_then_acr(self):
        cfg = DummyCfg()
        snippet = "Portable Document Format (PDF)"
        out = find_parenthetical_longform_before_acr(snippet, "PDF", cfg)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_trailing_punctuation_is_normalized(self):
        cfg = DummyCfg()
        snippet = "Graphics   Processing   Unit...   (GPU)"
        out = find_parenthetical_longform_before_acr(snippet, "GPU", cfg)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_titlecase_tail_preference_is_respected(self):
        cfg = DummyCfg()
        # If the tighten_definition_span favors the last TitleCase/UPPER chunk,
        # ensure we still get the meaningful tail.
        snippet = "See also the HyperText Transfer Protocol (HTTP)"
        out = find_parenthetical_longform_before_acr(snippet, "HTTP", cfg)
        assert len(out) == 1
        assert out[0].definition == "HyperText Transfer Protocol"

    def test_boundary_and_whitespace_variants(self):
        cfg = DummyCfg()
        snippet = "Read Only Memory   ( ROM )   "
        out = find_parenthetical_longform_before_acr(snippet, "ROM", cfg)
        assert len(out) == 1
        assert out[0].definition == "Read Only Memory"

    def test_respects_max_chars_integration(self):
        cfg = DummyCfg(max_phrase_chars=10)
        snippet = "Hypertext Transfer Protocol (HTTP)"
        # Def > 10 chars → no match
        assert find_parenthetical_longform_before_acr(snippet, "HTTP", cfg) == []
