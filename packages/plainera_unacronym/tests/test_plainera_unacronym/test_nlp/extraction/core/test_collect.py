import pytest

from plainera_unacronym.nlp.extraction.core.collect import initials_match
from test_plainera_unacronym.test_nlp.extraction.test_helper_patterns import _msg


class TestRequireInitialsMatchOk:
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

    def test_all_caps_token_expands_into_multiple_initials(self):
        # "RNA" expands to "RNA", "Polymerase" -> "P" => initials "RNAP"
        assert initials_match("RNAP", "RNA Polymerase") is True
        # If expansion works, "RNP" should also match as subsequence of "RNAP"
        assert initials_match("RNP", "RNA Polymerase") is True
        # But order still matters
        assert initials_match("RAPN", "RNA Polymerase") is False

    def test_all_caps_with_punct_or_digits_does_not_expand(self):
        # "RNA-seq" is not alpha => only "R" contributes, then "P" => initials "RP"
        assert initials_match("RNAP", "RNA-seq Polymerase") is False
        assert initials_match("RP", "RNA-seq Polymerase") is True

        # Token with punctuation does not expand ("U.S.A." -> "U", not "USA")
        assert initials_match("USA", "U.S.A. Agency") is False
        assert initials_match("UA", "U.S.A. Agency") is True


    def test_leading_punctuation_words_are_ignored(self):
        phrase = '(Portable) Document Format'
        # initials from this phrase: "D" + "F" => "DF"
        assert initials_match("PDF", phrase) is False
        assert initials_match("DF", phrase) is True

    def test_dotted_acronym_letters_are_matched(self):
        assert initials_match("P.D.F.", "Portable Document Format") is True
        assert initials_match("U.K.", "United Kingdom") is True

    def test_whitespace_variants(self):
        assert initials_match("PDF", "Portable\tDocument\nFormat") is True
        assert initials_match("PDF", "  Portable   Document   Format  ") is True
