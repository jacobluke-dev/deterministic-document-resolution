from document_resolution.nlp.extraction.acronyms.matchers.defs.defs_common import _numeric_leading


class TestNumericLeadingUnit:
    def test_returns_false_when_include_numeric_leading_false_even_if_numeric(self, _patch):
        _patch(_numeric_leading, first_alnum_char_upper=lambda _t: "3")
        assert _numeric_leading("3M", include_numeric_leading=False) is False

    def test_returns_false_when_no_alnum_found(self, _patch):
        _patch(_numeric_leading, first_alnum_char_upper=lambda _t: None)
        assert _numeric_leading("...", include_numeric_leading=True) is False

    def test_returns_false_when_first_alnum_is_alpha(self, _patch):
        _patch(_numeric_leading, first_alnum_char_upper=lambda _t: "A")
        assert _numeric_leading("Alpha", include_numeric_leading=True) is False

    def test_returns_true_when_first_alnum_is_digit(self, _patch):
        _patch(_numeric_leading, first_alnum_char_upper=lambda _t: "2")
        assert _numeric_leading("2FA", include_numeric_leading=True) is True

    def test_returns_true_when_first_alnum_is_non_alpha_symbol_like_underscore(self, _patch):
        # if the first_alnum_char_upper ever returns "_" (it shouldn't, underscore isn't alnum),
        # but keeping this here as a guard for “non-alpha” behaviour:
        _patch(_numeric_leading, first_alnum_char_upper=lambda _t: "7")
        assert _numeric_leading("__7zip", include_numeric_leading=True) is True


class TestNumericLeadingIntegration:
    def test_digit_leading_true(self):
        assert _numeric_leading("3M", include_numeric_leading=True) is True
        assert _numeric_leading("10GbE", include_numeric_leading=True) is True
        assert _numeric_leading("2", include_numeric_leading=True) is True

    def test_letter_leading_false(self):
        assert _numeric_leading("PDF", include_numeric_leading=True) is False
        assert _numeric_leading("U.S.A", include_numeric_leading=True) is False

    def test_punct_then_digit_true(self):
        # first alnum is '3'
        assert _numeric_leading("(3M)", include_numeric_leading=True) is True
        assert _numeric_leading("...2FA", include_numeric_leading=True) is True

    def test_punct_then_letter_false(self):
        # first alnum is 'P'
        assert _numeric_leading("(PDF)", include_numeric_leading=True) is False

    def test_include_numeric_leading_gate_overrides_everything(self):
        assert _numeric_leading("3M", include_numeric_leading=False) is False
        assert _numeric_leading("(2FA)", include_numeric_leading=False) is False
