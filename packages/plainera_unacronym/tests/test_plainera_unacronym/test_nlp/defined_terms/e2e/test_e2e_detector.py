import pytest

class TestDefinedTermDetectorMentions:
    def test_detect_logs_later_mentions_after_introductions(self, defined_term_detector_factory):

        text = '''
        This Master Services Agreement (the "Agreement") is entered into on the "Effective Date".
        "Effective Date" means the date on which both Parties sign this Agreement.
        "Services" means the software development, support, and maintenance services described in Schedule A.
        "Confidential Information" shall mean any non-public business, technical, or commercial information
        disclosed by one Party to the other.

        The Agreement shall commence on the Effective Date.
        The Services shall be performed with reasonable skill and care.
        Each Party shall protect Confidential Information.
        Under this Agreement, the Services may change from time to time after the Effective Date.
        '''.strip()

        result = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            enabled_domains=("legal",)
            ).detect(text)

        assert len(result.introductions) == 4

        mention_keys = [m.normalized_key for m in result.mentions]
        print("MENTION_KEYS", mention_keys)

        assert "agreement" in mention_keys
        assert "effective_date" in mention_keys
        assert "services" in mention_keys
        assert "confidential_information" in mention_keys

        assert mention_keys.count("effective_date") >= 1
        assert mention_keys.count("services") >= 2

class TestDefinedTermDetectorE2E:
    @staticmethod
    def _intro_keys(result) -> list[str]:
        return [intro.normalized_key for intro in result.introductions]

    @staticmethod
    def _mention_keys(result) -> list[str]:
        return [m.normalized_key for m in result.mentions]

    @staticmethod
    def _mention_spans(result) -> set[tuple[int, int]]:
        return {(m.start_offset, m.end_offset) for m in result.mentions}

    @staticmethod
    def _intro_spans(result) -> set[tuple[int, int]]:
        return {(intro.start_offset, intro.end_offset) for intro in result.introductions}

    def test_detect_collects_introductions_and_later_mentions(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Effective Date" means the date on which both Parties sign this Agreement.
        "Services" means the software development, support, and maintenance services described in Schedule A.
        "Confidential Information" shall mean any non-public business, technical, or commercial information
        disclosed by one Party to the other.

        The Agreement shall commence on the Effective Date.
        The Services shall be performed with reasonable skill and care.
        Each Party shall protect Confidential Information.
        Under this Agreement, the Services may change from time to time after the Effective Date.
        """.strip()

        result = detector.detect(text)

        assert len(result.introductions) == 4
        assert set(self._intro_keys(result)) == {
            "agreement",
            "effective_date",
            "services",
            "confidential_information",
        }

        mention_keys = self._mention_keys(result)

        assert "agreement" in mention_keys
        assert "effective_date" in mention_keys
        assert "services" in mention_keys
        assert "confidential_information" in mention_keys

        assert mention_keys.count("services") >= 2
        assert mention_keys.count("effective_date") >= 2

        assert self._intro_spans(result).isdisjoint(self._mention_spans(result))

    def test_detect_preserves_multiple_introductions_for_same_term(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        "Services" means the consultancy services described in the main body.

        Schedule A
        "Services" means the software maintenance services described in this schedule.

        The Services shall be delivered in accordance with this Agreement.
        """.strip()

        result = detector.detect(text)

        services_intros = [i for i in result.introductions if i.normalized_key == "services"]

        assert len(services_intros) == 2
        assert "services" in result.unique_terms
        assert any(m.normalized_key == "services" for m in result.mentions)

    def test_detect_unquoted_mentions_when_enabled(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        "Agreement" means this Master Services Agreement.
        "Services" means the software development services.
        "Effective Date" means the date of signature.

        The Agreement shall commence on the Effective Date.
        The Services shall begin after the Effective Date.
        """.strip()

        result = detector.detect(text)
        mention_keys = self._mention_keys(result)

        assert "agreement" in mention_keys
        assert "services" in mention_keys
        assert "effective_date" in mention_keys

    def test_detect_does_not_count_pre_intro_quoted_reference_as_later_mention(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        The parties discussed the "Effective Date" before execution.
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Effective Date shall be recorded in writing.
        """.strip()

        result = detector.detect(text)

        effective_mentions = [m for m in result.mentions if m.normalized_key == "effective_date"]

        assert len(result.introductions) == 1
        assert len(effective_mentions) == 1
        assert effective_mentions[0].start_offset > result.introductions[0].end_offset

    def test_detect_parenthetical_alias_is_introduction_not_later_mention(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Agreement shall commence on the Effective Date.
        """.strip()

        result = detector.detect(text)

        agreement_intros = [i for i in result.introductions if i.normalized_key == "agreement"]
        agreement_mentions = [m for m in result.mentions if m.normalized_key == "agreement"]

        assert len(agreement_intros) == 1
        assert len(agreement_mentions) >= 1

        intro_span = (agreement_intros[0].start_offset, agreement_intros[0].end_offset)
        mention_spans = {(m.start_offset, m.end_offset) for m in agreement_mentions}

        assert intro_span not in mention_spans


    def test_detect_does_not_emit_intro_spans_as_mentions(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Effective Date" means the date on which both Parties sign this Agreement.
        "Services" means the software development services described in Schedule A.

        The Agreement shall commence on the Effective Date.
        The Services shall begin on the Effective Date.
        """.strip()

        result = detector.detect(text)

        intro_spans = self._intro_spans(result)
        mention_spans = set(self._mention_spans(result))

        assert len(result.introductions) == 3
        assert intro_spans.isdisjoint(mention_spans)

    def test_detect_dedupes_same_mention_span(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.

        The "Effective Date" shall be recorded in writing.
        """.strip()

        result = detector.detect(text)

        effective_mentions = [m for m in result.mentions if m.normalized_key == "effective_date"]
        mention_spans = [(m.start_offset, m.end_offset) for m in effective_mentions]

        assert len(effective_mentions) == 1
        assert len(mention_spans) == len(set(mention_spans))

    def test_detect_unquoted_mentions_require_legal_mode_when_configured(self, defined_term_detector_factory):
        text = """
        "Agreement" means this contract.
        "Services" means the software development services.

        The Agreement shall commence next week.
        The Services shall begin after signature.
        """.strip()

        detector_no_legal = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=(),
        )
        result_no_legal = detector_no_legal.detect(text)

        detector_legal = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=True,
            enabled_domains=("legal",),
        )
        result_legal = detector_legal.detect(text)

        assert self._mention_keys(result_no_legal) == []
        assert "agreement" in self._mention_keys(result_legal)
        assert "services" in self._mention_keys(result_legal)

    def test_detect_does_not_emit_unknown_terms_as_mentions(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        "Agreement" means this Master Services Agreement.

        Delivery Date shall be discussed later.
        Termination may occur on notice.
        The Agreement shall commence on signature.
        """.strip()

        result = detector.detect(text)
        mention_keys = self._mention_keys(result)

        assert "agreement" in mention_keys
        assert "delivery_date" not in mention_keys
        assert "termination" not in mention_keys

    def test_detect_handles_multiple_later_mentions_of_same_term(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        "Services" means the software development, support, and maintenance services.

        The Services shall commence on Monday.
        The Services shall be provided with reasonable skill and care.
        Changes to the Services must be agreed in writing.
        """.strip()

        result = detector.detect(text)

        service_mentions = [m for m in result.mentions if m.normalized_key == "services"]

        assert len(result.introductions) == 1
        assert len(service_mentions) == 3

    def test_detect_parenthetical_alias_and_means_intros_coexist(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Services" means the software development services described in Schedule A.

        The Agreement shall commence on signature.
        The Services shall be performed with reasonable skill and care.
        """.strip()

        result = detector.detect(text)

        intro_keys = self._intro_keys(result)
        mention_keys = self._mention_keys(result)

        assert "agreement" in intro_keys
        assert "services" in intro_keys
        assert "agreement" in mention_keys
        assert "services" in mention_keys

    def test_detect_preserves_document_order_of_introductions(self, defined_term_detector_factory):
        detector = defined_term_detector_factory(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        text = """
        This Master Services Agreement (the "Agreement") is made on the Effective Date.
        "Services" means the consultancy services described in the main body.

        Schedule A
        "Services" means the software maintenance services described in this schedule.
        """.strip()

        result = detector.detect(text)

        intro_keys = self._intro_keys(result)
        intro_starts = [intro.start_offset for intro in result.introductions]
        service_intros = [intro for intro in result.introductions if intro.normalized_key == "services"]

        assert intro_starts == sorted(intro_starts)
        assert intro_keys == ["agreement", "services", "services"]
        assert len(service_intros) == 2
        assert service_intros[0].start_offset < service_intros[1].start_offset
