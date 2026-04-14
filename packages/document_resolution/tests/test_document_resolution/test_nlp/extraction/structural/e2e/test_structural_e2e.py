from __future__ import annotations

from document_resolution.nlp.extraction.structural.config import (
    StructuralReferenceDetectorConfig,
    StructuralReferenceExtractionConfig,
)
from document_resolution.nlp.extraction.structural.execute import detect_and_resolve_structural_references


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
            "build_structural_reference_entries",
            "extract_structural_anchors",
            "build_structural_anchor_index",
            "link_structural_references",
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

    def test_resolve_schedule_reference(self) -> None:
        text = (
            "This Agreement incorporates the services described in Schedule A.\n\n"
            "Schedule A: Services Description\n"
            "The Supplier shall provide the Services set out below.\n"
        )

        det_res, extr = detect_and_resolve_structural_references(text)

        assert len(det_res.references) == 2
        assert [ref.normalized_key for ref in det_res.references] == [
            "schedule_a",
            "schedule_a",
        ]

        assert len(extr.references) == 2
        assert [ref.canonical_key for ref in extr.references] == [
            "schedule_a",
            "schedule_a",
        ]

        assert len(extr.links) == 2
        assert all(link.canonical_key == "schedule_a" for link in extr.links)

        assert len(extr.unique_links) == 1
        assert set(extr.unique_links) == {"schedule_a"}

        link = extr.unique_links["schedule_a"]
        assert link.target_span is not None
        assert link.strength == 1.0

        start, end = link.target_span
        assert text[start:end] == "Schedule A: Services Description"

    def test_resolve_section_reference(self) -> None:
        text = (
            "The termination rights in Section 4.2 apply in the following circumstances.\n\n"
            "4.2 Termination\n"
            "Either party may terminate this Agreement on written notice.\n"
        )

        det_res, extr = detect_and_resolve_structural_references(text)

        assert len(det_res.references) == 1
        assert det_res.references[0].normalized_key == "section_4_2"

        assert len(extr.references) == 1
        assert extr.references[0].canonical_key == "section_4_2"

        assert len(extr.links) == 1
        link = extr.links[0]

        assert link.canonical_key == "section_4_2"
        assert link.target_span is not None
        assert link.strength == 1.0

        start, end = link.target_span
        assert text[start:end] == "4.2 Termination"

    def test_unresolved_reference(self) -> None:
        text = (
            "The parties shall comply with the obligations set out in Schedule C.\n\n"
            "Schedule A: Services Description\n"
            "Schedule B: Charges\n"
        )

        det_res, extr = detect_and_resolve_structural_references(text)

        assert len(det_res.references) == 3
        assert [ref.normalized_key for ref in det_res.references] == [
            "schedule_c",
            "schedule_a",
            "schedule_b",
        ]

        assert len(extr.references) == 3
        assert [ref.canonical_key for ref in extr.references] == [
            "schedule_c",
            "schedule_a",
            "schedule_b",
        ]

        assert len(extr.links) == 3
        assert len(extr.unique_links) == 3
        assert set(extr.unique_links) == {"schedule_a", "schedule_b", "schedule_c"}

        unresolved = extr.unique_links["schedule_c"]
        assert unresolved.target_span is None
        assert unresolved.strength == 0.0

    def test_return_state_includes_anchors_index_and_links(self) -> None:
        text = (
            "See Section 4.2 and Schedule A.\n\n"
            "4.2 Termination\n"
            "Schedule A: Services Description\n"
        )

        det_res, extr, state = detect_and_resolve_structural_references(
            text,
            return_state=True,
        )

        assert len(det_res.references) == 3
        assert [ref.normalized_key for ref in det_res.references] == [
            "section_4_2",
            "schedule_a",
            "schedule_a",
        ]

        assert len(extr.references) == 3
        assert len(extr.links) == 3

        assert len(state.anchors) == 2
        assert set(state.anchor_index) == {"section_4_2", "schedule_a"}

        assert len(state.link_entries) == 3

        assert len(extr.unique_links) == 2
        assert set(extr.unique_links) == {"section_4_2", "schedule_a"}

        assert extr.unique_links["section_4_2"].target_span is not None
        assert extr.unique_links["schedule_a"].target_span is not None

    def test_resolve_article_roman_reference(self):
        text = (
            "The rule in Article III applies to all disputes.\n\n"
            "Article III: Interpretation\n"
            "Definitions and interpretive rules appear here.\n"
        )

        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        det_res, extr = detect_and_resolve_structural_references(
            text,
            ext_cfg=cfg,
        )

        assert len(det_res.references) == 2
        assert [ref.normalized_key for ref in det_res.references] == [
            "article_iii",
            "article_iii",
        ]

        assert len(extr.references) == 2
        assert [ref.canonical_key for ref in extr.references] == [
            "article_3",
            "article_3",
        ]

        assert len(extr.links) == 2
        assert len(extr.unique_links) == 1
        assert set(extr.unique_links) == {"article_3"}

        link = extr.unique_links["article_3"]
        assert link.target_span is not None
        assert link.strength == 1.0

        start, end = link.target_span
        assert text[start:end] == "Article III: Interpretation"

    def test_unresolved_article_roman_reference(self) -> None:
        text = (
            "The exception in Article IV applies in limited cases.\n\n"
            "Article III: Interpretation\n"
            "Schedule A: Services Description\n"
        )

        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        det_res, extr = detect_and_resolve_structural_references(
            text,
            ext_cfg=cfg,
        )

        assert "article_iv" in [ref.normalized_key for ref in det_res.references]
        assert "article_4" in [ref.canonical_key for ref in extr.references]

        assert "article_4" in extr.unique_links
        link = extr.unique_links["article_4"]

        assert link.target_span is None
        assert link.strength == 0.0

    def test_return_state_includes_roman_anchor_index_and_links(self) -> None:
        text = (
            "See Article III for interpretation.\n\n"
            "Article III: Interpretation\n"
        )

        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        det_res, extr, state = detect_and_resolve_structural_references(
            text,
            ext_cfg=cfg,
            return_state=True,
        )

        assert len(det_res.references) == 2
        assert [ref.normalized_key for ref in det_res.references] == [
            "article_iii",
            "article_iii",
        ]

        assert len(state.anchors) == 1
        assert set(state.anchor_index) == {"article_3"}

        assert len(state.link_entries) == 2
        assert len(extr.unique_links) == 1
        assert extr.unique_links["article_3"].target_span is not None

    def test_unique_links_deduplicates_resolved_schedule_links(self) -> None:
        text = (
            "The parties shall comply with Schedule C.\n\n"
            "Schedule A: Services Description\n"
            "Schedule C: Charges\n"
        )

        det_res, extr = detect_and_resolve_structural_references(text)

        assert len(det_res.references) == 3
        assert [ref.normalized_key for ref in det_res.references] == [
            "schedule_c",
            "schedule_a",
            "schedule_c",
        ]

        assert len(extr.links) == 3

        schedule_c_links = [link for link in extr.links if link.canonical_key == "schedule_c"]
        assert len(schedule_c_links) == 2
        assert all(link.target_span is not None for link in schedule_c_links)
        assert [link.strength for link in schedule_c_links] == [1.0, 0.5]

        assert "schedule_c" in extr.unique_links
        unique_link = extr.unique_links["schedule_c"]

        assert unique_link.target_span is not None
        assert unique_link.strength == 1.0

        start, end = unique_link.target_span
        assert text[start:end] == "Schedule C: Charges"

    def test_unique_links_preserves_first_when_all_unresolved(self) -> None:
        text = "The parties shall comply with Schedule C and Schedule C."

        det_res, extr = detect_and_resolve_structural_references(text)

        assert len(det_res.references) == 2
        assert [ref.normalized_key for ref in det_res.references] == [
            "schedule_c",
            "schedule_c",
        ]

        assert len(extr.links) == 2
        assert all(link.target_span is None for link in extr.links)
        assert all(link.strength == 0.0 for link in extr.links)

        assert "schedule_c" in extr.unique_links
        unique_link = extr.unique_links["schedule_c"]

        assert unique_link.target_span is None
        assert unique_link.strength == 0.0
        assert unique_link.reference_span == extr.links[0].reference_span
