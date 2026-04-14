from document_resolution.nlp.detection.structural import StructuralReferenceDetector


class _DetCfg:
    pass


class TestFalsePositives:

    def test_does_not_detect_plain_capitalised_phrase_as_structural_reference(self) -> None:
        text = "The Services Agreement sets out the commercial terms."

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert out.references == []

    def test_does_not_detect_schedule_word_without_valid_label(self) -> None:
        text = "The project schedule will be reviewed next week."

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert out.references == []

    def test_does_not_detect_section_word_used_generically(self) -> None:
        text = "Please read the next section carefully before proceeding."

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert out.references == []

    def test_does_not_detect_article_word_used_in_plain_language(self) -> None:
        text = "The article discusses recent developments in competition law."

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert out.references == []

    def test_does_not_detect_appendix_word_without_structural_label(self) -> None:
        text = "Further background is provided in the appendix to this guide."

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert out.references == []

    def test_does_not_detect_annex_word_without_valid_reference_form(self) -> None:
        text = "The team will annex additional evidence if required."

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert out.references == []

    def test_detects_valid_structural_reference_without_matching_nearby_plain_language_uses(self) -> None:
        text = (
            "The project schedule will be reviewed next week. "
            "If pricing changes, the Parties shall update Schedule A."
        )

        out = StructuralReferenceDetector(config=_DetCfg()).detect(text)

        assert [ref.normalized_key for ref in out.references] == ["schedule_a"]
