import pytest

from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.common.shared import has_letters


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
