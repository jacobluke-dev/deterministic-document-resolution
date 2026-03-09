import pytest

class TestDefinedTermFalsePositiveRegressions:
    def test_detect_ignores_heading_like_phrase(self, defined_term_detector_factory):

        text = """
        Termination

        The Supplier may terminate this Agreement on written notice.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset()).detect(text)


        assert "termination" not in result.unique_terms
        assert all(o.normalized_key != "termination" for o in result.occurrences)

    def test_detect_ignores_sentence_initial_capitalised_word_as_occurrence(self, defined_term_detector_factory):

        text = """
        "Services" means support services.
        Tomorrow the parties will meet.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset()).detect(text)

        assert "services" in result.unique_terms
        assert all(o.term != "Tomorrow" for o in result.occurrences)
        assert all(o.normalized_key != "tomorrow" for o in result.occurrences)

    def test_detect_ignores_unanchored_bare_reference(self, defined_term_detector_factory):

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"})).detect("The Customer shall pay within 30 days.")

        assert result.unique_terms == {}
        assert result.occurrences == []

    def test_detect_ignores_party_name_when_not_defined(self, defined_term_detector_factory):

        text = "Acme Limited entered into this agreement with Beta Systems plc."

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset()).detect(text)

        assert "acme_limited" not in result.unique_terms
        assert "beta_systems_plc" not in result.unique_terms
        assert all(o.normalized_key not in {"acme_limited", "beta_systems_plc"} for o in result.occurrences)

    def test_detect_ignores_statute_or_authority_name_as_defined_term(self, defined_term_detector_factory):

        text = (
            "The parties shall comply with the Companies Act 2006 and guidance "
            "from the Information Commissioner."
        )

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset()).detect(text)

        assert "companies_act_2006" not in result.unique_terms
        assert "information_commissioner" not in result.unique_terms
        assert all(
            o.normalized_key not in {"companies_act_2006", "information_commissioner"}
            for o in result.occurrences
        )


class TestDefinedTermBoundaryRegressions:
    def test_iter_occurrences_excludes_leading_article_from_bare_occurrence(self, cfg_terms_det_factory,
                                                                            defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}),
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}))

        text = '"Supplier" means Acme Ltd.\nThe Supplier shall provide the Services.'
        intros = detector._iter_term_introductions(text, cfg=cfg, legal_active=True)
        unique_terms = {intro.normalized_key: intro for intro in intros}
        intro_term_spans = {(intro.start_offset, intro.end_offset) for intro in intros}

        occurrences = detector._iter_occurrences(
            text,
            known_keys=set(unique_terms.keys()),
            intro_term_spans=intro_term_spans,
            cfg=cfg,
            legal_active=True,
        )

        assert [o.term for o in occurrences] == ["Supplier"]
        assert text[occurrences[0].start_offset : occurrences[0].end_offset] == "Supplier"

    def test_iter_occurrences_excludes_trailing_verb_from_bare_occurrence(self, cfg_terms_det_factory,
                                                                          defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}),
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}))

        text = '"Supplier" means Acme Ltd.\nSupplier shall provide the Services.'
        intros = detector._iter_term_introductions(text, cfg=cfg, legal_active=True)
        unique_terms = {intro.normalized_key: intro for intro in intros}
        intro_term_spans = {(intro.start_offset, intro.end_offset) for intro in intros}

        occurrences = detector._iter_occurrences(
            text,
            known_keys=set(unique_terms.keys()),
            intro_term_spans=intro_term_spans,
            cfg=cfg,
            legal_active=True,
        )

        assert [o.term for o in occurrences] == ["Supplier"]
        assert text[occurrences[0].start_offset : occurrences[0].end_offset] == "Supplier"

    def test_iter_term_introductions_preserves_full_multiword_term_boundary(self, cfg_terms_det_factory,
                                                                            defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=False,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}),
        )

        text = '"Business Day" means any day other than Saturday or Sunday.'
        intros = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=False,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}))._iter_term_introductions(text, cfg=cfg, legal_active=True)

        assert [i.term for i in intros] == ["Business Day"]
        intro = intros[0]
        assert text[intro.start_offset : intro.end_offset] == "Business Day"
        assert intro.normalized_key == "business_day"
