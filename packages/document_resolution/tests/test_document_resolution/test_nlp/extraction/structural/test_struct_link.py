from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from document_resolution.nlp.extraction.structural.link import (
    build_structural_reference_links,
)
from document_resolution.nlp.extraction.structural.types import (
    StructuralAnchor,
    StructuralReferenceEntry,
)


@dataclass
class LogSpy:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, message, *a, **kw) -> None:
        self.calls.append({"message": message, **kw})

@pytest.fixture
def log_spy() -> LogSpy:
    return LogSpy()

class TestBuildStructuralReferenceLinks:

    def test_builds_resolved_link_when_matching_anchor_exists(self, _patch, log_spy, patch_sink) -> None:
        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )

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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"schedule_a": [anchor]},
        )

        assert log_spy.calls == []

        assert len(out) == 1
        assert out[0].canonical_key == "schedule_a"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span == (40, 70)
        assert out[0].strength == 1.0

    def test_builds_unresolved_link_when_no_matching_anchor_exists(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )

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


        assert log_spy.calls == []

        assert len(out) == 1
        assert out[0].canonical_key == "schedule_c"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span is None
        assert out[0].strength == 0.0

    def test_uses_canonical_key_for_lookup(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )
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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={
                "article_3": [anchor],
                "article_iii": [],
            },
        )

        assert log_spy.calls == []
        assert len(out) == 1
        assert out[0].target_span == (40, 55)
        assert out[0].strength == 1.0

    def test_clause_reference_does_not_link_to_section_anchor_by_default(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )
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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert log_spy.calls == []
        assert len(out) == 1
        assert out[0].canonical_key == "clause_4_2"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span is None
        assert out[0].strength == 0.0

    def test_matching_kind_links_successfully(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )
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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert log_spy.calls == []
        assert len(out) == 1
        assert out[0].canonical_key == "section_4_2"
        assert out[0].reference_span == (5, 15)
        assert out[0].target_span == (40, 70)
        assert out[0].strength == 1.0

    def test_exact_forward_match_has_high_confidence(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )
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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert log_spy.calls == []
        assert len(out) == 1
        assert out[0].target_span == (30, 45)
        assert out[0].strength == 1.0

    def test_backward_fallback_has_lower_confidence(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )
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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert log_spy.calls == []
        assert len(out) == 1
        assert out[0].target_span == (70, 90)
        assert out[0].strength == 0.75

    def test_unresolved_link_has_zero_confidence(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )
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

        assert log_spy.calls == []
        assert len(out) == 1
        assert out[0].target_span is None
        assert out[0].strength == 0.0

    def test_unknown_match_tier_logs_and_defaults_to_unresolved(self, _patch, log_spy, patch_sink) -> None:

        def _bad_select_best_anchor(*, ref, candidates):
            return candidates[0], "sideways"

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
            _select_best_anchor=_bad_select_best_anchor,
        )

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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].target_span == (30, 45)
        assert out[0].match_strategy == "unresolved"
        assert out[0].strength == 0.0

        assert len(log_spy.calls) == 1
        assert log_spy.calls[0]["message"] == "structural.link.unsupported_match_tier"
        assert log_spy.calls[0]["level"].name == "WARNING"
        assert log_spy.calls[0]["details"]["tier"] == "sideways"
        assert log_spy.calls[0]["details"]["canonical_key"] == "section_4_2"

    def test_forward_match_does_not_log(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )

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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].target_span == (30, 45)
        assert out[0].match_strategy == "forward"
        assert out[0].strength == 1.0
        assert log_spy.calls == []

    def test_overlap_fallback_has_low_nonzero_confidence(self, _patch, log_spy, patch_sink) -> None:

        _patch(
            build_structural_reference_links,
            message_logger=log_spy,
            sink=patch_sink,
        )

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
            title=None,
        )

        out = build_structural_reference_links(
            references=[ref],
            anchor_index={"section_4_2": [anchor]},
        )

        assert len(out) == 1
        assert out[0].target_span == (48, 58)
        assert out[0].match_strategy == "overlap"
        assert out[0].strength == 0.5
        assert log_spy.calls == []
