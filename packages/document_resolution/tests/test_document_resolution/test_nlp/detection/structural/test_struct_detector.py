from __future__ import annotations

import pytest
from document_resolution.nlp.detection.structural import StructuralReferenceDetector


class _Cfg:
    pass


@pytest.fixture
def detector(_patch) -> StructuralReferenceDetector:
    detector = StructuralReferenceDetector(config=_Cfg())

    _patch(
        detector.detect.__func__,
        message_logger=lambda *args, **kwargs: None,
    )

    return detector


class TestStructuralReferenceDetector:
    def test_detect_schedule_reference(self, detector: StructuralReferenceDetector):
        text = "See Schedule A for the pricing details."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Schedule"
        assert ref.label == "A"
        assert ref.start_offset == text.index("Schedule A")
        assert ref.end_offset == text.index("Schedule A") + len("Schedule A")
        assert ref.normalized_key == "schedule_a"
        assert ref.provenance == "structural_reference_detector"

    def test_detect_section_reference(self, detector: StructuralReferenceDetector):
        text = "The obligations are described in Section 4.2 of this Agreement."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Section"
        assert ref.label == "4.2"
        assert ref.start_offset == text.index("Section 4.2")
        assert ref.end_offset == text.index("Section 4.2") + len("Section 4.2")
        assert ref.normalized_key == "section_4_2"

    def test_detect_clause_reference(self, detector: StructuralReferenceDetector):
        text = "Termination rights are set out in Clause 7.3."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Clause"
        assert ref.label == "7.3"
        assert ref.start_offset == text.index("Clause 7.3")
        assert ref.end_offset == text.index("Clause 7.3") + len("Clause 7.3")
        assert ref.normalized_key == "clause_7_3"

    def test_detect_article_reference(self, detector: StructuralReferenceDetector):
        text = "The governance rules appear in Article III."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Article"
        assert ref.label == "III"
        assert ref.start_offset == text.index("Article III")
        assert ref.end_offset == text.index("Article III") + len("Article III")
        assert ref.normalized_key == "article_iii"

    def test_detect_multiple_structural_references_in_source_order(self, detector: StructuralReferenceDetector):
        text = "See Clause 7, Section 4.2 and Schedule A."

        out = detector.detect(text)

        assert [ref.kind for ref in out.references] == ["Clause", "Section", "Schedule"]
        assert [ref.label for ref in out.references] == ["7", "4.2", "A"]
        assert [ref.normalized_key for ref in out.references] == [
            "clause_7",
            "section_4_2",
            "schedule_a",
        ]

    def test_ignore_capitalised_phrase(self, detector: StructuralReferenceDetector):
        text = "The Board of Directors approved the Annual Budget yesterday."

        out = detector.detect(text)

        assert out.references == []

    def test_detect_is_case_insensitive_for_structural_keyword(self, detector: StructuralReferenceDetector):
        text = "please see section 4.2 for more detail"

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Section"
        assert ref.label == "4.2"
        assert ref.normalized_key == "section_4_2"

    def test_detect_parallel_delegates_to_detect(self, detector: StructuralReferenceDetector):
        text = "See Appendix C for examples."

        out = detector.detect_parallel(text)

        assert len(out.references) == 1
        assert out.references[0].kind == "Appendix"
        assert out.references[0].label == "C"
        assert out.references[0].normalized_key == "appendix_c"

    def test_iter_structural_references_preserves_offsets(self, detector: StructuralReferenceDetector):
        text = "Attached at Exhibit B and Annex 1."

        refs = detector._iter_structural_references(text)

        assert len(refs) == 2
        assert text[refs[0].start_offset:refs[0].end_offset] == "Exhibit B"
        assert text[refs[1].start_offset:refs[1].end_offset] == "Annex 1"

    def test_detect_structural_references_mixed_document_text(self, detector):
        text = (
            "The Board approved the Annual Budget. "
            "See Section 4.2, Clause 7, Schedule A and Article III."
        )

        out = detector.detect(text)

        assert [ref.normalized_key for ref in out.references] == [
            "section_4_2",
            "clause_7",
            "schedule_a",
            "article_iii",
        ]

    def test_detect_lowercase_structural_keyword(self, detector):
        text = "please see section 4.2 for details"

        out = detector.detect(text)

        assert len(out.references) == 1
        assert out.references[0].normalized_key == "section_4_2"

    def test_detect_section_integer_label(self, detector):
        text = "See Section 4 for details."

        out = detector.detect(text)

        assert len(out.references) == 1
        assert out.references[0].label == "4"
        assert out.references[0].normalized_key == "section_4"

    def test_detect_article_numeric_label(self, detector):
        text = "The rules are in Article 2."

        out = detector.detect(text)

        assert len(out.references) == 1
        assert out.references[0].normalized_key == "article_2"

    def test_detect_appendix_reference(self, detector):
        text = "Refer to Appendix C."

        out = detector.detect(text)

        assert len(out.references) == 1
        assert out.references[0].normalized_key == "appendix_c"

    def test_detect_reference_offsets_map_back_to_source_text(self, detector):
        text = "Attached at Exhibit B and Annex 1."

        out = detector.detect(text)

        assert [text[r.start_offset:r.end_offset] for r in out.references] == [
            "Exhibit B",
            "Annex 1",
        ]

    def test_ignore_non_structural_phrase_with_keyword_fragment(self, detector):
        text = "We bought a sectional sofa and discussed article writing."

        out = detector.detect(text)

        assert out.references == []

    def test_ignore_reference_split_across_newline(self, detector):
        text = "See Section\n4.2 for details."

        out = detector.detect(text)

        assert out.references == []

    def test_detect_repeated_reference_at_distinct_spans(self, detector):
        text = "Section 4 applies. Later, Section 4 is varied."

        out = detector.detect(text)

        assert [ref.normalized_key for ref in out.references] == [
            "section_4",
            "section_4",
        ]
        assert out.references[0].start_offset != out.references[1].start_offset

    def test_detect_reference_excludes_trailing_punctuation(self, detector):
        text = "See Clause 7.3, for termination."

        out = detector.detect(text)

        assert len(out.references) == 1
        ref = out.references[0]
        assert text[ref.start_offset:ref.end_offset] == "Clause 7.3"

    def test_detect_appendix_alpha_decimal_reference(self, detector: StructuralReferenceDetector):
        text = "Further details are set out in Appendix C.13."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Appendix"
        assert ref.label == "C.13"
        assert ref.start_offset == text.index("Appendix C.13")
        assert ref.end_offset == text.index("Appendix C.13") + len("Appendix C.13")
        assert ref.normalized_key == "appendix_c_13"

    def test_detect_appendix_alpha_multi_decimal_reference(self, detector: StructuralReferenceDetector):
        text = "See Appendix A.1.2 for worked examples."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Appendix"
        assert ref.label == "A.1.2"
        assert ref.normalized_key == "appendix_a_1_2"

    def test_detect_appendix_numeric_reference_still_supported(self, detector: StructuralReferenceDetector):
        text = "The calculation appears in Appendix 13."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Appendix"
        assert ref.label == "13"
        assert ref.normalized_key == "appendix_13"

    def test_detect_appendix_single_alpha_reference_still_supported(self, detector: StructuralReferenceDetector):
        text = "Refer to Appendix C."

        out = detector.detect(text)

        assert len(out.references) == 1

        ref = out.references[0]
        assert ref.kind == "Appendix"
        assert ref.label == "C"
        assert ref.normalized_key == "appendix_c"

    def test_ignore_invalid_appendix_alpha_decimal_trailing_dot_fragment(self, detector):
        text = "Refer to Appendix C.A for the schedule."

        out = detector.detect(text)

        assert out.references == []
