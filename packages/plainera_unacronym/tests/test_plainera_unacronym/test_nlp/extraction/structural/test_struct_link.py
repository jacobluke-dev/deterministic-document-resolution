from __future__ import annotations

from plainera_unacronym.nlp.extraction.structural.link import (
    _select_best_anchor,
    build_structural_reference_links,
)
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralAnchor,
    StructuralReferenceEntry,
)


class TestSelectBestAnchor:
    def test_prefers_nearest_forward_anchor(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=10,
            end_offset=20,
            provenance="detected",
        )
        candidates = [
            StructuralAnchor(
                label="4.2",
                normalized_key="section_4_2",
                start_offset=80,
                end_offset=95,
                ordinal=1,
            ),
            StructuralAnchor(
                label="4.2",
                normalized_key="section_4_2",
                start_offset=30,
                end_offset=45,
                ordinal=0,
            ),
        ]

        out, _ = _select_best_anchor(ref=ref, candidates=candidates)

        assert out.start_offset == 30
        assert out.end_offset == 45
        assert out.ordinal == 0

    def test_uses_nearest_backward_anchor_when_no_forward_anchor_exists(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=100,
            end_offset=110,
            provenance="detected",
        )
        candidates = [
            StructuralAnchor(
                label="4.2",
                normalized_key="section_4_2",
                start_offset=10,
                end_offset=20,
                ordinal=0,
            ),
            StructuralAnchor(
                label="4.2",
                normalized_key="section_4_2",
                start_offset=70,
                end_offset=90,
                ordinal=1,
            ),
        ]

        out, _ = _select_best_anchor(ref=ref, candidates=candidates)

        assert out.start_offset == 70
        assert out.end_offset == 90
        assert out.ordinal == 1

    def test_falls_back_to_lowest_ordinal_when_candidates_overlap_reference(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=50,
            end_offset=60,
            provenance="detected",
        )
        candidates = [
            StructuralAnchor(
                label="4.2",
                normalized_key="section_4_2",
                start_offset=45,
                end_offset=55,
                ordinal=3,
            ),
            StructuralAnchor(
                label="4.2",
                normalized_key="section_4_2",
                start_offset=48,
                end_offset=58,
                ordinal=1,
            ),
        ]

        out, _ = _select_best_anchor(ref=ref, candidates=candidates)

        assert out.ordinal == 1


class TestBuildStructuralReferenceLinks:
    def test_builds_resolved_link_when_matching_anchor_exists(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Schedule",
            label="A",
            canonical_label="A",
            normalized_key="schedule_a",
            canonical_key="schedule_a",
            start_offset=5,
            end_offset=15,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="A",
            normalized_key="schedule_a",
            start_offset=40,
            end_offset=70,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"schedule_a": [anchor]},
        )

        assert len(out) == 1
        assert out[0].canonical_key == "schedule_a"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span == (40, 70)
        assert out[0].strength == 1.0

    def test_builds_unresolved_link_when_no_matching_anchor_exists(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Schedule",
            label="C",
            canonical_label="C",
            normalized_key="schedule_c",
            canonical_key="schedule_c",
            start_offset=5,
            end_offset=15,
            provenance="detected",
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={},
        )

        assert len(out) == 1
        assert out[0].canonical_key == "schedule_c"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span is None
        assert out[0].strength == 0.0

    def test_uses_canonical_key_for_lookup(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Article",
            label="III",
            canonical_label="3",
            normalized_key="article_iii",
            canonical_key="article_3",
            start_offset=5,
            end_offset=15,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="III",
            normalized_key="article_3",
            start_offset=40,
            end_offset=55,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={
                "article_3": [anchor],
                "article_iii": [],
            },
        )

        assert len(out) == 1
        assert out[0].target_span == (40, 55)
        assert out[0].strength == 1.0

    def test_clause_reference_does_not_link_to_section_anchor_by_default(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Clause",
            label="4.2",
            canonical_label="4.2",
            normalized_key="clause_4_2",
            canonical_key="clause_4_2",
            start_offset=5,
            end_offset=15,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="4.2",
            normalized_key="section_4_2",
            start_offset=40,
            end_offset=70,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].canonical_key == "clause_4_2"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span is None
        assert out[0].strength == 0.0

    def test_matching_kind_links_successfully(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=5,
            end_offset=15,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="4.2",
            normalized_key="section_4_2",
            start_offset=40,
            end_offset=70,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].canonical_key == "section_4_2"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span == (40, 70)
        assert out[0].strength == 1.0

    def test_exact_forward_match_has_high_confidence(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=10,
            end_offset=20,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="4.2",
            normalized_key="section_4_2",
            start_offset=30,
            end_offset=45,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].target_span == (30, 45)
        assert out[0].strength == 1.0

    def test_backward_fallback_has_lower_confidence(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=100,
            end_offset=110,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="4.2",
            normalized_key="section_4_2",
            start_offset=70,
            end_offset=90,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].target_span == (70, 90)
        assert out[0].strength == 0.75

    def test_unresolved_link_has_zero_confidence(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=10,
            end_offset=20,
            provenance="detected",
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={},
        )

        assert len(out) == 1
        assert out[0].target_span is None
        assert out[0].strength == 0.0

    def test_overlap_fallback_has_low_nonzero_confidence(self) -> None:
        ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=50,
            end_offset=60,
            provenance="detected",
        )
        anchor = StructuralAnchor(
            label="4.2",
            normalized_key="section_4_2",
            start_offset=48,
            end_offset=58,
            ordinal=0,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].target_span == (48, 58)
        assert out[0].strength == 0.5
