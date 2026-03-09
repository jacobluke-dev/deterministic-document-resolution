from plainera_unacronym.nlp.detection.defined_terms.compiler import compile_defined_term_patterns


class TestCompileDefinedTermPatterns:
    def test_quoted_means_matches(self):
        pats = compile_defined_term_patterns()
        text = '"Effective Date" means the date of signature.'
        m = pats.quoted_means.search(text)

        assert m is not None
        assert m.group("term_q") == '"Effective Date"'

    def test_quoted_shall_mean_matches(self):
        pats = compile_defined_term_patterns()
        text = '"Confidential Information" shall mean non-public information.'
        m = pats.quoted_shall_mean.search(text)

        assert m is not None
        assert m.group("term_q") == '"Confidential Information"'

    def test_bare_means_matches_bridge_words(self):
        pats = compile_defined_term_patterns()
        text = 'Change of Control means any sale of assets.'
        m = pats.bare_means.search(text)

        assert m is not None
        assert m.group("term_b") == "Change of Control"

    def test_bare_shall_mean_matches(self):
        pats = compile_defined_term_patterns()
        text = 'Confidential Information shall mean non-public information.'
        m = pats.bare_shall_mean.search(text)

        assert m is not None
        assert m.group("term_b") == "Confidential Information"

    def test_parenthetical_alias_matches(self):
        pats = compile_defined_term_patterns()
        text = 'This Master Services Agreement (the "Agreement") is entered into...'
        m = pats.parenthetical_alias.search(text)

        assert m is not None
        assert m.group("term_q") == '"Agreement"'

    def test_quoted_occurrence_matches(self):
        pats = compile_defined_term_patterns()
        text = 'The "Services" will begin on the Effective Date.'
        m = pats.quoted_occurrence.search(text)

        assert m is not None
        assert m.group("term") == "Services"

    def test_capitalised_occurrence_matches_bridge_words(self):
        pats = compile_defined_term_patterns()
        text = "Following a Change of Control, the Customer may terminate."

        matches = [m.group("term") for m in pats.capitalised_occurrence.finditer(text)]

        assert "Change of Control" in matches

    def test_bare_means_does_not_cross_paragraph_boundary(self):
        pats = compile_defined_term_patterns()
        text = "Schedule A.\n\nChange of Control means any sale of assets."
        m = pats.bare_means.search(text)

        assert m is not None
        assert m.group("term_b") == "Change of Control"
