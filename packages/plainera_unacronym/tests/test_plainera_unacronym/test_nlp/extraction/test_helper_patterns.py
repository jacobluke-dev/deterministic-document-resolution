import pytest

from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.common.shared import has_letters


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
