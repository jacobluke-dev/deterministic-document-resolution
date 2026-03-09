from plainera_unacronym.nlp.detection.defined_terms.normalise import normalize_defined_term_key


class TestNormalizeDefinedTermKey:
    def test_normalizes_simple_multiword_term(self):
        assert normalize_defined_term_key("Effective Date") == "effective_date"

    def test_preserves_bridge_words(self):
        assert normalize_defined_term_key("Change of Control") == "change_of_control"

    def test_strips_surrounding_quotes_and_whitespace(self):
        assert normalize_defined_term_key('  "Confidential Information"  ') == "confidential_information"

    def test_folds_curly_apostrophes_via_canon_table(self):
        assert normalize_defined_term_key("Customer’s Materials") == "customers_materials"

    def test_folds_unicode_dashes_via_canon_table(self):
        assert normalize_defined_term_key("Change–of–Control") == "change_of_control"

    def test_removes_non_word_punctuation_noise(self):
        assert normalize_defined_term_key("Services,") == "services"
        assert normalize_defined_term_key("(Services)") == "services"

    def test_collapses_internal_whitespace(self):
        assert normalize_defined_term_key("  Master   Services   Agreement  ") == "master_services_agreement"

    def test_replaces_ascii_hyphens_with_underscores(self):
        assert normalize_defined_term_key("Non-Disclosure Agreement") == "non_disclosure_agreement"

    def test_handles_mixed_quotes_dashes_and_spacing(self):
        term = '  "Change–of–Control"  '
        assert normalize_defined_term_key(term) == "change_of_control"
