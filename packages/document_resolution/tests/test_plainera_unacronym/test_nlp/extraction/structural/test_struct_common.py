import pytest
from plainera_unacronym.nlp.extraction.structural.common import roman_to_int


class TestRomanToInt:

    def test_roman_to_int_complex_subtractive_numeral(self):
        assert roman_to_int("MCMXCIV") == 1994

    def test_roman_to_int_complex_additive_numeral(self):
        assert roman_to_int("MMMDCCCLXXXVIII") == 3888

    def test_roman_to_int_mcmxciv(self):
        assert roman_to_int("MCMXCIV") == 1994

    def test_roman_to_int_rejects_il(self):
        with pytest.raises(ValueError, match="Invalid Roman numeral"):
            roman_to_int("IL")

    def test_roman_to_int_rejects_ic(self):
        with pytest.raises(ValueError, match="Invalid Roman numeral"):
            roman_to_int("IC")

    def test_roman_to_int_rejects_vx(self):
        with pytest.raises(ValueError, match="Invalid Roman numeral"):
            roman_to_int("VX")

    def test_roman_to_int_rejects_iiii(self):
        with pytest.raises(ValueError, match="Invalid Roman numeral"):
            roman_to_int("IIII")
