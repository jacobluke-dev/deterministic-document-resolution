import plainera_unacronym.nlp.detection.defined_terms.detector as det_mod
from plainera_unacronym.nlp.detection.defined_terms.detector import (
    _overlaps_any,
    _spans_overlap,
)


class TestDefinedTermDetectorHelpers:
    def test__spans_overlap_true_for_partial_overlap(self):
        assert _spans_overlap(10, 20, 15, 25) is True

    def test__spans_overlap_false_for_touching_only(self):
        assert _spans_overlap(10, 20, 20, 30) is False

    def test__overlaps_any_true_when_any_span_overlaps(self):
        spans = {(100, 120), (200, 220)}
        assert _overlaps_any(110, 115, spans) is True

    def test__overlaps_any_false_when_no_span_overlaps(self):
        spans = {(100, 120), (200, 220)}
        assert _overlaps_any(121, 130, spans) is False


class TestDefinedTermDetectorWithAutoDomains:
    def test__with_auto_domains_merges_new_domains(self, _patch, defined_term_detector_factory):
        detector = defined_term_detector_factory(enabled_domains=frozenset({"bio"}))

        _patch(
            det_mod.DefinedTermDetector._with_auto_domains,
            autodetect_domains=lambda text, cfg_: frozenset({"bio", "legal"}),
        )

        out = detector._with_auto_domains("some contract text")
        assert out is not detector.cfg
        assert out.enabled_domains == frozenset({"bio", "legal"})

    def test__with_auto_domains_returns_same_config_when_no_new_domains(self, _patch, defined_term_detector_factory):
        detector = defined_term_detector_factory(enabled_domains=frozenset({"legal"}))

        _patch(
            det_mod.DefinedTermDetector._with_auto_domains,
            autodetect_domains=lambda text, cfg_: frozenset({"legal"}),
        )

        out = detector._with_auto_domains("some contract text")
        assert out is detector.cfg


class TestDefinedTermDetectorResolveKnownTermFromRun:
    def test_resolve_known_term_from_run_returns_exact_match(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        known_keys = {"confidential_information", "effective_date"}

        out = detector._resolve_known_term_from_run("Confidential Information", known_keys)

        assert out == ("Confidential Information", "confidential_information")

    def test_resolve_known_term_from_run_resolves_suffix_match(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        known_keys = {"confidential_information", "effective_date"}

        out = detector._resolve_known_term_from_run("Party's Confidential Information", known_keys)

        assert out == ("Confidential Information", "confidential_information")

    def test_resolve_known_term_from_run_returns_none_for_unknown_phrase(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        known_keys = {"confidential_information", "effective_date"}

        out = detector._resolve_known_term_from_run("Master Services Agreement", known_keys)

        assert out is None

    def test_resolve_known_term_from_run_returns_none_for_empty_input(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        known_keys = {"confidential_information"}

        out = detector._resolve_known_term_from_run("", known_keys)

        assert out is None


class TestDefinedTermDetectorExtractDefinitionText:
    def test_extract_definition_text_stops_at_period(self, defined_term_detector_factory):
        text = 'Intro. "Effective Date" means the date of signature. Next sentence.'

        anchor_end = text.index("means") + len("means")
        definition, start, end = (defined_term_detector_factory(max_definition_chars=200)
                                  ._extract_definition_text(text, anchor_end))

        assert definition == "the date of signature"
        assert text[start:end] == "the date of signature"

    def test_extract_definition_text_stops_at_semicolon(self, defined_term_detector_factory):
        text = '"Services" means software support and maintenance; provided remotely.'

        anchor_end = text.index("means") + len("means")
        definition, start, end = (defined_term_detector_factory(max_definition_chars=200)
                                  ._extract_definition_text(text, anchor_end))

        assert definition == "software support and maintenance"
        assert text[start:end] == "software support and maintenance"

    def test_extract_definition_text_stops_at_newline(self, defined_term_detector_factory):
        text = '"Services" means software support and maintenance\nAdditional text follows'

        anchor_end = text.index("means") + len("means")
        definition, start, end = (defined_term_detector_factory(max_definition_chars=200)
                                  ._extract_definition_text(text, anchor_end))

        assert definition == "software support and maintenance"
        assert text[start:end] == "software support and maintenance"

    def test_extract_definition_text_respects_max_definition_chars(self, defined_term_detector_factory):
        text = '"Services" means software support and maintenance without punctuation'

        anchor_end = text.index("means") + len("means")
        definition, start, end = (defined_term_detector_factory(max_definition_chars=12)
                                  ._extract_definition_text(text, anchor_end))

        assert definition == "software su"
        assert end - start == len("software su")

    def test_extract_definition_text_strips_leading_spacing_and_punctuation(self, defined_term_detector_factory):
        text = '"Services" means :,- software support and maintenance.'

        anchor_end = text.index("means") + len("means")
        definition, start, end = (defined_term_detector_factory(max_definition_chars=200)
                                  ._extract_definition_text(text, anchor_end))

        assert definition == "software support and maintenance"
        assert text[start:end] == definition


class TestDefinedTermDetectorIterTermIntroductions:
    def test_extracts_quoted_means_introduction(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        text = '"Effective Date" means the date of signature.'

        intros = detector._iter_term_introductions(
            text,
            cfg=detector.cfg,
            legal_active=False,
        )

        assert [i.term for i in intros] == ["Effective Date"]
        assert intros[0].normalized_key == "effective_date"

    def test_extracts_quoted_shall_mean_introduction(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        text = '"Confidential Information" shall mean non-public information.'

        intros = detector._iter_term_introductions(
            text,
            cfg=detector.cfg,
            legal_active=False,
        )

        assert [i.term for i in intros] == ["Confidential Information"]
        assert intros[0].normalized_key == "confidential_information"

    def test_extracts_bare_means_with_bridge_words_when_legal_active(self,
                                                                     cfg_terms_det_factory,
                                                                     defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True)
        text = "Change of Control means any sale of assets."

        intros = detector._iter_term_introductions(
            text,
            cfg=cfg,
            legal_active=True,
        )

        assert [i.term for i in intros] == ["Change of Control"]
        assert intros[0].normalized_key == "change_of_control"

    def test_skips_bare_means_when_legal_inactive_and_required(self,
                                                               cfg_terms_det_factory,
                                                               defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True, )
        text = "Change of Control means any sale of assets."

        intros = detector._iter_term_introductions(
            text,
            cfg=cfg,
            legal_active=False,
        )

        assert intros == []

    def test_skips_bare_means_when_unquoted_terms_disabled(self,
                                                           cfg_terms_det_factory,
                                                           defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=False,
            require_legal_domain_for_unquoted=False,
        )
        detector = defined_term_detector_factory(allow_unquoted_capitalised_terms=False,
                                                 require_legal_domain_for_unquoted=False, )
        text = "Change of Control means any sale of assets."

        intros = detector._iter_term_introductions(
            text,
            cfg=cfg,
            legal_active=True,
        )

        assert intros == []

    def test_extracts_parenthetical_alias_introduction(self, cfg_terms_det_factory, defined_term_detector_factory):
        cfg = cfg_terms_det_factory()
        detector = defined_term_detector_factory()
        text = 'This Master Services Agreement (the "Agreement") is entered into...'

        intros = detector._iter_term_introductions(
            text,
            cfg=cfg,
            legal_active=False,
        )

        assert [i.term for i in intros] == ["Agreement"]
        assert intros[0].normalized_key == "agreement"

    def test_extracts_parenthetical_alias_without_the(self, cfg_terms_det_factory, defined_term_detector_factory):
        cfg = cfg_terms_det_factory()
        detector = defined_term_detector_factory()
        text = 'Acme Ltd ("Supplier") agrees to provide the Services.'

        intros = detector._iter_term_introductions(
            text,
            cfg=cfg,
            legal_active=False,
        )

        assert [i.term for i in intros] == ["Supplier"]
        assert intros[0].normalized_key == "supplier"


class TestDefinedTermDetectorIterOccurrences:
    def test_detects_quoted_occurrence_for_known_term(self, defined_term_detector_factory):
        detector = defined_term_detector_factory()
        text = 'The "Services" will begin tomorrow.'

        out = detector._iter_references(
            text,
            known_keys={"services"},
            first_intro_end_by_key={"services": 0},
            intro_term_spans=set(),
            cfg=detector.cfg,
            legal_active=False,
        )

        assert [o.term for o in out] == ["Services"]

    def test_skips_quoted_occurrence_when_not_known_term(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(enabled_domains=frozenset({"legal"}))
        text = 'The "Agreement" will begin tomorrow.'

        out = detector._iter_references(
            text,
            known_keys={"services"},
            first_intro_end_by_key={},
            intro_term_spans=set(),
            cfg=detector.cfg,
            legal_active=False,
        )

        assert out == []

    def test_skips_intro_span_for_quoted_occurrence(self, cfg_terms_det_factory, defined_term_detector_factory):
        text = '"Services" means support services.'

        out = defined_term_detector_factory(enabled_domains=frozenset({"legal"}))._iter_references(
            text,
            known_keys={"services"},
            first_intro_end_by_key={},
            intro_term_spans={(0, 10)},  # span for Services without quotes
            cfg=cfg_terms_det_factory(),
            legal_active=False,
        )

        assert out == []

    def test_detects_unquoted_occurrence_when_legal_active(self, cfg_terms_det_factory, defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True, )
        text = "Following a Change of Control, the Customer may terminate."

        out = detector._iter_references(
            text,
            known_keys={"change_of_control"},
            first_intro_end_by_key={"change_of_control": 0},
            intro_term_spans=set(),
            cfg=cfg,
            legal_active=True,
        )

        assert [o.term for o in out] == ["Change of Control"]
        assert out[0].normalized_key == "change_of_control"

    def test_skips_unquoted_occurrence_when_legal_inactive_and_required(self,
                                                                        cfg_terms_det_factory,
                                                                        defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"})
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
        enabled_domains=frozenset({"legal"}))
        text = "Following a Change of Control, the Customer may terminate."

        out = detector._iter_references(
            text,
            known_keys={"change_of_control"},
            intro_term_spans=set(),
            first_intro_end_by_key={},
            cfg=cfg,
            legal_active=False,
        )

        assert out == []

    def test_resolves_suffix_from_broader_capitalised_run(self, cfg_terms_det_factory, defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False)
        text = "Each Party shall protect the other Party's Confidential Information."

        out = detector._iter_references(
            text,
            known_keys={"confidential_information"},
            first_intro_end_by_key={"confidential_information": 0},
            intro_term_spans=set(),
            cfg=cfg,
            legal_active=False,
        )

        assert [o.term for o in out] == ["Confidential Information"]
        assert out[0].normalized_key == "confidential_information"
        assert text[out[0].start_offset: out[0].end_offset] == "Confidential Information"

    def test_skips_broader_capitalised_run_when_suffix_not_known(self, cfg_terms_det_factory,
                                                                 defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False)
        text = "Each Party shall protect the other Party's Confidential Information."

        out = detector._iter_references(
            text,
            known_keys={"effective_date"},
            first_intro_end_by_key={},
            intro_term_spans=set(),
            cfg=cfg,
            legal_active=False,
        )

        assert out == []

    def test_skips_unquoted_occurrence_when_intro_span_overlaps(self,
                                                                cfg_terms_det_factory,
                                                                defined_term_detector_factory):
        cfg = cfg_terms_det_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False)
        text = "Change of Control means any sale of assets."

        out = detector._iter_references(
            text,
            known_keys={"change_of_control"},
            first_intro_end_by_key={},
            intro_term_spans={(0, 17)},
            cfg=cfg,
            legal_active=False,
        )

        assert out == []


class TestDefinedTermDetectorDetect:
    def test_detect_returns_unique_terms_and_occurrences_for_quoted_introductions(self, defined_term_detector_factory):
        text = """
        "Effective Date" means the date of signature.
        "Services" shall mean support and maintenance services.

        The Services begin on the Effective Date.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"})).detect(text)

        assert set(result.unique_terms.keys()) == {"effective_date", "services"}
        assert [o.normalized_key for o in result.mentions] == ["services", "effective_date"]

    def test_detect_includes_parenthetical_alias_as_unique_term(self, defined_term_detector_factory):
        text = """
        This Master Services Agreement (the "Agreement") is entered into today.
        The Agreement begins on the Effective Date.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"})).detect(text)

        assert "agreement" in result.unique_terms
        assert [o.term for o in result.mentions] == ["Agreement"]

    def test_detect_allows_bare_introduction_when_legal_active(self, defined_term_detector_factory):
        text = """
        Change of Control means any sale of assets.
        Following a Change of Control, the Customer may terminate.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"})).detect(text)

        assert "change_of_control" in result.unique_terms
        assert [o.term for o in result.mentions] == ["Change of Control"]

    def test_detect_skips_bare_introduction_when_legal_inactive(self, defined_term_detector_factory):
        text = """
        Change of Control means any sale of assets.
        Following a Change of Control, the Customer may terminate.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset()).detect(text)

        assert result.unique_terms == {}
        assert result.mentions == []

    def test_detect_skips_unquoted_terms_when_disabled_even_if_legal_active(self, defined_term_detector_factory):
        text = """
        Change of Control means any sale of assets.
        Following a Change of Control, the Customer may terminate.
        """

        result = defined_term_detector_factory(allow_unquoted_capitalised_terms=False,
                                               require_legal_domain_for_unquoted=False,
                                               enabled_domains=frozenset({"legal"}), ).detect(text)

        assert result.unique_terms == {}
        assert result.mentions == []

    def test_detect_does_not_duplicate_intro_spans_as_occurrences(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(allow_unquoted_capitalised_terms=True,
                                                 require_legal_domain_for_unquoted=False,
                                                 enabled_domains=frozenset({"legal"}), )

        text = """
        "Services" means support and maintenance.
        The Services begin tomorrow.
        """

        result = detector.detect(text)

        assert set(result.unique_terms.keys()) == {"services"}
        assert [o.term for o in result.mentions] == ["Services"]

    def test_detect_resolves_suffix_match_from_broader_capitalised_run(self, defined_term_detector_factory):
        text = """
        "Confidential Information" means non-public information.
        Each Party shall protect the other Party's Confidential Information.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"})).detect(text)

        assert "confidential_information" in result.unique_terms
        assert [o.term for o in result.mentions] == ["Confidential Information"]
        occ = result.mentions[0]
        assert text[occ.start_offset: occ.end_offset] == "Confidential Information"

    def test_detect_builds_unique_terms_by_normalized_key(self, defined_term_detector_factory):
        text = """
        "Effective Date" means the date of signature.
        "Services" means support and maintenance.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=False,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset({"legal"})).detect(text)

        assert sorted(result.unique_terms.keys()) == ["effective_date", "services"]
        assert result.unique_terms["effective_date"].term == "Effective Date"
        assert result.unique_terms["services"].term == "Services"

    def test_detect_uses_auto_domains_when_not_preenabled(self, _patch, defined_term_detector_factory):
        _patch(
            det_mod.DefinedTermDetector._with_auto_domains,
            autodetect_domains=lambda text, cfg_: frozenset({"legal"}),
        )

        text = """
        Change of Control means any sale of assets.
        Following a Change of Control, the Customer may terminate.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=frozenset()).detect(text)

        assert "change_of_control" in result.unique_terms
        assert [o.term for o in result.mentions] == ["Change of Control"]

    def test_detect_handles_mixed_introduction_styles(self, defined_term_detector_factory):
        text = """
        This Master Services Agreement (the "Agreement") is entered into today.
        "Effective Date" means the date of signature.
        Change of Control means any sale of assets.

        The Agreement starts on the Effective Date.
        Following a Change of Control, the Customer may terminate.
        """

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
            enabled_domains=frozenset({"legal"})).detect(text)

        assert set(result.unique_terms.keys()) == {
            "agreement",
            "effective_date",
            "change_of_control",
        }
        assert [o.normalized_key for o in result.mentions] == [
            "agreement",
            "effective_date",
            "change_of_control",
        ]
