from __future__ import annotations

from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceDetectorConfig,
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.execute import detect_and_resolve_structural_references


class TestDetectAndResolveStructuralReferences:
    def test_detect_and_resolve_structural_references_returns_detector_and_extraction_results(self):
        text = "See Section 4.2 and Schedule A."

        det_res, extr = detect_and_resolve_structural_references(text)

        assert len(det_res.references) == 2
        assert [ref.normalized_key for ref in det_res.references] == [
            "section_4_2",
            "schedule_a",
        ]

        assert len(extr.references) == 2
        assert [ref.canonical_key for ref in extr.references] == [
            "section_4_2",
            "schedule_a",
        ]

    def test_detect_and_resolve_structural_references_returns_reports_when_requested(self):
        text = "See Clause 7 and Article III."

        det_res, extr, reports = detect_and_resolve_structural_references(
            text,
            return_reports=True,
        )

        assert len(det_res.references) == 2
        assert len(extr.references) == 2
        assert reports
        assert [r.name for r in reports] == [
            "detect_structural_references",
            "build_structural_reference_resolutions",
            "assemble_structural_reference_resolutions",
        ]

    def test_detect_and_resolve_structural_references_returns_state_when_requested(self):
        text = "See Appendix C and Annex 1."

        det_res, extr, state = detect_and_resolve_structural_references(
            text,
            return_state=True,
        )

        assert state.det_res is det_res
        assert state.extr is extr
        assert len(state.reference_entries) == 2

    def test_detect_and_resolve_structural_references_returns_reports_and_state_when_requested(self):
        text = "See Section 4.2."

        det_res, extr, reports, state = detect_and_resolve_structural_references(
            text,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.references) == 1
        assert len(extr.references) == 1
        assert reports
        assert state.det_res is det_res
        assert state.extr is extr


    def test_detect_and_resolve_structural_references_keeps_all_references_and_first_unique_key(self):
        text = "Section 4.2 applies. Later, Section 4.2 is varied."

        det_res, extr = detect_and_resolve_structural_references(text)

        assert [ref.canonical_key for ref in extr.references] == [
            "section_4_2",
            "section_4_2",
        ]
        assert list(extr.unique_keys.keys()) == ["section_4_2"]
        assert extr.unique_keys["section_4_2"] == extr.references[0]

    def test_detect_and_resolve_structural_references_handles_no_matches(self):
        text = "The Board approved the Annual Budget yesterday."

        det_res, extr = detect_and_resolve_structural_references(text)

        assert det_res.references == []
        assert extr.references == []
        assert extr.unique_keys == {}

    def test_detect_and_resolve_structural_references_converts_article_roman_when_enabled(self):
        text = "See Article III."
        det_cfg = StructuralReferenceDetectorConfig()
        ext_cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        det_res, extr = detect_and_resolve_structural_references(
            text,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
        )

        assert det_res.references[0].normalized_key == "article_iii"
        assert extr.references[0].canonical_key == "article_3"
        assert extr.references[0].canonical_label == "3"

    def test_detect_and_resolve_structural_references_preserves_article_roman_when_disabled(self):
        text = "See Article III."
        ext_cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=False)

        det_res, extr = detect_and_resolve_structural_references(
            text,
            ext_cfg=ext_cfg,
        )

        assert det_res.references[0].normalized_key == "article_iii"
        assert extr.references[0].canonical_key == "article_iii"
        assert extr.references[0].canonical_label == "III"

    def test_detect_and_resolve_structural_references_reports_include_stage_info(self):
        text = "See Section 4.2 and Schedule A."

        _, _, reports = detect_and_resolve_structural_references(
            text,
            return_reports=True,
        )

        assert reports[0].name == "detect_structural_references"
        assert "references=2" in reports[0].info

    def test_detect_and_resolve_structural_references_preserves_source_order(self):
        text = "See Clause 7, Section 4.2 and Schedule A."

        _, extr = detect_and_resolve_structural_references(text)

        assert [ref.canonical_key for ref in extr.references] == [
            "clause_7",
            "section_4_2",
            "schedule_a",
        ]

    def test_detect_and_resolve_structural_references_handles_appendix_alpha_decimal(self):
        text = "Refer to Appendix C.13 for the schedule."

        det_res, extr = detect_and_resolve_structural_references(text)

        assert det_res.references[0].normalized_key == "appendix_c_13"
        assert extr.references[0].canonical_key == "appendix_c_13"
