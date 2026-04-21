import pytest
from document_resolution.nlp.extraction.acronyms.matchers.defs.defs_common import _acronym_letters_rtl


class TestAcronymLettersRtl:
    @pytest.mark.parametrize(
        "tok, expected",
        [
            ("RNA", ["A", "N", "R"]),
            ("U.S.A", ["A", "S", "U"]),
            ("HTTP2", ["2", "P", "T", "T", "H"]),
            ("10GbE", ["E", "B", "G", "0", "1"]),  # lowercase -> uppercase; digits kept
        ],
    )
    def test_extracts_alnum_upper_and_reverses(self, tok, expected):
        assert _acronym_letters_rtl(tok) == expected

    def test_strips_light_punctuation_only_at_edges(self):
        assert _acronym_letters_rtl('("PDF")') == ["F", "D", "P"]
        assert _acronym_letters_rtl("PDF,") == ["F", "D", "P"]
        assert _acronym_letters_rtl("»PDF«") == ["F", "D", "P"]

    def test_drops_non_alnum_characters_inside_token(self):
        # internal punctuation is ignored (keeps only alnum)
        assert _acronym_letters_rtl("U..S") == ["S", "U"]
        assert _acronym_letters_rtl("U-S") == ["S", "U"]

    def test_empty_or_no_alnum_returns_empty_list(self):
        assert _acronym_letters_rtl("") == []
        assert _acronym_letters_rtl("...") == []
        assert _acronym_letters_rtl("—–") == []
