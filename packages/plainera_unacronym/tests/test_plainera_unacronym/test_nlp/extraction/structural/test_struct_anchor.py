from __future__ import annotations

from plainera_unacronym.nlp.extraction.structural.anchor import (
    build_structural_anchor_index,
    extract_structural_anchors,
)
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)


class TestExtractStructuralAnchors:
    def test_extract_named_schedule_anchor(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        text = "Intro text\nSchedule A: Services Description\nMore text"
        out = extract_structural_anchors(text=text, cfg=cfg)

        assert len(out) == 1
        assert out[0].label == "A"
        assert out[0].normalized_key == "schedule_a"
        assert out[0].ordinal == 0
        assert text[out[0].start_offset : out[0].end_offset] == "Schedule A: Services Description"

    def test_extract_numbered_section_anchor(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        text = "Preamble\n4.2 Termination\nBody"
        out = extract_structural_anchors(text=text, cfg=cfg)

        assert len(out) == 1
        assert out[0].label == "4.2"
        assert out[0].normalized_key == "section_4_2"
        assert out[0].ordinal == 0

    def test_extract_article_roman_anchor_converts_when_enabled(self) -> None:
        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        text = "Article III: Interpretation"
        out = extract_structural_anchors(text=text, cfg=cfg)

        assert len(out) == 1
        assert out[0].label == "III"
        assert out[0].normalized_key == "article_3"

    def test_extract_article_roman_anchor_preserves_roman_when_disabled(self) -> None:
        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=False)

        text = "Article III: Interpretation"
        out = extract_structural_anchors(text=text, cfg=cfg)

        assert len(out) == 1
        assert out[0].normalized_key == "article_iii"

    def test_ignores_non_heading_lines(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        text = (
            "The parties refer to Schedule A in the body text.\n"
            "This is not a heading.\n"
            "Another ordinary sentence."
        )
        out = extract_structural_anchors(text=text, cfg=cfg)

        assert out == []

    def test_assigns_ordinals_only_to_matched_anchors(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        text = (
            "Intro paragraph\n"
            "Schedule A: Services Description\n"
            "Body text\n"
            "4.2 Termination\n"
        )
        out = extract_structural_anchors(text=text, cfg=cfg)

        assert len(out) == 2
        assert [anchor.ordinal for anchor in out] == [0, 1]
        assert [anchor.normalized_key for anchor in out] == ["schedule_a", "section_4_2"]


class TestBuildStructuralAnchorIndex:
    def test_groups_anchors_by_lookup_key_preserving_order(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        text = (
            "Schedule A: First\n"
            "Some text\n"
            "Schedule A: Second\n"
            "4.2 Termination\n"
        )
        anchors = extract_structural_anchors(text=text, cfg=cfg)

        out = build_structural_anchor_index(anchors)

        assert set(out) == {"schedule_a", "section_4_2"}
        assert [anchor.label for anchor in out["schedule_a"]] == ["A", "A"]
        assert [anchor.ordinal for anchor in out["schedule_a"]] == [0, 1]
        assert [anchor.label for anchor in out["section_4_2"]] == ["4.2"]

    def test_returns_empty_index_for_empty_input(self) -> None:
        out = build_structural_anchor_index([])

        assert out == {}

    def test_numbered_heading_anchor_captures_title_text(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        out = extract_structural_anchors(
            text="4.2 Termination\n",
            cfg=cfg,
        )

        assert len(out) == 1
        assert out[0].label == "4.2"
        assert out[0].title == "Termination"
        assert out[0].normalized_key == "section_4_2"

    def test_named_heading_anchor_captures_title_text(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        out = extract_structural_anchors(
            text="Schedule A: Services Description\n",
            cfg=cfg,
        )

        assert len(out) == 1
        assert out[0].label == "A"
        assert out[0].title == "Services Description"
        assert out[0].normalized_key == "schedule_a"

    def test_anchor_lookup_key_does_not_include_title(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        out = extract_structural_anchors(
            text=(
                "Schedule A: Services Description\n"
                "4.2 Termination\n"
            ),
            cfg=cfg,
        )

        assert len(out) == 2
        assert [(anchor.label, anchor.title, anchor.normalized_key) for anchor in out] == [
            ("A", "Services Description", "schedule_a"),
            ("4.2", "Termination", "section_4_2"),
        ]

    def test_heading_without_title_sets_title_none(self) -> None:
        cfg = StructuralReferenceExtractionConfig()

        out = extract_structural_anchors(
            text=(
                "Schedule A\n"
                "Section 4.2\n"
            ),
            cfg=cfg,
        )

        assert len(out) == 2
        assert [anchor.label for anchor in out] == ["A", "4.2"]
        assert [anchor.title for anchor in out] == [None, None]
        assert [anchor.normalized_key for anchor in out] == ["schedule_a", "section_4_2"]
