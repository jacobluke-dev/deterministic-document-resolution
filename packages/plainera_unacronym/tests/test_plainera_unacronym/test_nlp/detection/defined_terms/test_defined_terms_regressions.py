from plainera_unacronym.nlp.detection.defined_terms.detector import DefinedTermDetector


class TestDefinedTermFalsePositiveRegressions:
    def test_detect_ignores_heading_like_phrase(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

        text = """
        Termination

        The Supplier may terminate this Agreement on written notice.
        """

        result = detector.detect(text)

        assert "termination" not in result.unique_terms
        assert all(o.normalized_key != "termination" for o in result.occurrences)

    def test_detect_ignores_sentence_initial_capitalised_word_as_occurrence(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

        text = """
        "Services" means support services.
        Tomorrow the parties will meet.
        """

        result = detector.detect(text)

        assert "services" in result.unique_terms
        assert all(o.term != "Tomorrow" for o in result.occurrences)
        assert all(o.normalized_key != "tomorrow" for o in result.occurrences)

    def test_detect_ignores_unanchored_bare_reference(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

        text = "The Customer shall pay within 30 days."

        result = detector.detect(text)

        assert result.unique_terms == {}
        assert result.occurrences == []

    def test_detect_ignores_party_name_when_not_defined(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

        text = "Acme Limited entered into this agreement with Beta Systems plc."

        result = detector.detect(text)

        assert "acme_limited" not in result.unique_terms
        assert "beta_systems_plc" not in result.unique_terms
        assert all(o.normalized_key not in {"acme_limited", "beta_systems_plc"} for o in result.occurrences)

    def test_detect_ignores_statute_or_authority_name_as_defined_term(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

        text = (
            "The parties shall comply with the Companies Act 2006 and guidance "
            "from the Information Commissioner."
        )

        result = detector.detect(text)

        assert "companies_act_2006" not in result.unique_terms
        assert "information_commissioner" not in result.unique_terms
        assert all(
            o.normalized_key not in {"companies_act_2006", "information_commissioner"}
            for o in result.occurrences
        )


class TestDefinedTermBoundaryRegressions:
    def test_iter_occurrences_excludes_leading_article_from_bare_occurrence(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

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

    def test_iter_occurrences_excludes_trailing_verb_from_bare_occurrence(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

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

    def test_iter_term_introductions_preserves_full_multiword_term_boundary(self, cfg_terms_det_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=False,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"}),
        )
        detector = DefinedTermDetector(cfg)

        text = '"Business Day" means any day other than Saturday or Sunday.'
        intros = detector._iter_term_introductions(text, cfg=cfg, legal_active=True)

        assert [i.term for i in intros] == ["Business Day"]
        intro = intros[0]
        assert text[intro.start_offset : intro.end_offset] == "Business Day"
        assert intro.normalized_key == "business_day"
