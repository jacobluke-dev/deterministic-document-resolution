from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceLink,
    StructuralReferenceResolutionResult,
)
from public_api.core.processing.structural_chunking import (
    merge_structural_reference_results,
    shift_structural_reference_result,
)


def _make_reference(
    *,
    kind: str = "section",
    label: str = "Section 2",
    canonical_label: str = "Section 2",
    normalized_key: str = "section 2",
    canonical_key: str = "section|2",
    start_offset: int = 10,
    end_offset: int = 19,
    provenance: str = "document",
) -> StructuralReferenceEntry:
    return StructuralReferenceEntry(
        kind=kind,
        label=label,
        canonical_label=canonical_label,
        normalized_key=normalized_key,
        canonical_key=canonical_key,
        start_offset=start_offset,
        end_offset=end_offset,
        provenance=provenance,
    )


def _make_link(
    *,
    kind: str = "section",
    label: str = "Section 2",
    canonical_label: str = "Section 2",
    normalized_key: str = "section 2",
    canonical_key: str = "section|2",
    reference_span: tuple[int, int] = (30, 39),
    target_span: tuple[int, int] | None = (100, 109),
    match_strategy: str = "exact",
    strength: float = 1.0,
    provenance: str = "document",
) -> StructuralReferenceLink:
    return StructuralReferenceLink(
        kind=kind,
        label=label,
        canonical_label=canonical_label,
        normalized_key=normalized_key,
        canonical_key=canonical_key,
        reference_span=reference_span,
        target_span=target_span,
        match_strategy=match_strategy,
        strength=strength,
        provenance=provenance,
    )


class TestShiftStructuralReferenceResult:
    def test_shifts_references_links_and_unique_maps(self) -> None:
        reference = _make_reference(start_offset=10, end_offset=19)
        link = _make_link(reference_span=(30, 39), target_span=(100, 109))

        result = StructuralReferenceResolutionResult(
            references=[reference],
            links=[link],
            unique_keys={reference.canonical_key: reference},
            unique_links={link.canonical_key: link},
        )

        shifted = shift_structural_reference_result(result, 200)

        shifted_reference = shifted.references[0]
        assert shifted_reference.start_offset == 210
        assert shifted_reference.end_offset == 219

        shifted_link = shifted.links[0]
        assert shifted_link.reference_span == (230, 239)
        assert shifted_link.target_span == (300, 309)

        unique_reference = shifted.unique_keys[reference.canonical_key]
        assert unique_reference.start_offset == 210
        assert unique_reference.end_offset == 219

        unique_link = shifted.unique_links[link.canonical_key]
        assert unique_link.reference_span == (230, 239)
        assert unique_link.target_span == (300, 309)

    def test_returns_original_object_when_delta_is_zero(self) -> None:
        result = StructuralReferenceResolutionResult(
            references=[],
            links=[],
            unique_keys={},
            unique_links={},
        )

        shifted = shift_structural_reference_result(result, 0)

        assert shifted is result


class TestMergeStructuralReferenceResults:
    def test_merges_deduplicates_and_prefers_resolved_unique_link(self) -> None:
        reference = _make_reference()
        unresolved_link = _make_link(
            reference_span=(30, 39),
            target_span=None,
            match_strategy="unresolved",
            strength=0.2,
        )
        resolved_link = _make_link(
            reference_span=(30, 39),
            target_span=(100, 109),
            match_strategy="exact",
            strength=1.0,
        )

        result_1 = StructuralReferenceResolutionResult(
            references=[reference],
            links=[unresolved_link],
            unique_keys={reference.canonical_key: reference},
            unique_links={unresolved_link.canonical_key: unresolved_link},
        )
        result_2 = StructuralReferenceResolutionResult(
            references=[reference],
            links=[resolved_link],
            unique_keys={reference.canonical_key: reference},
            unique_links={resolved_link.canonical_key: resolved_link},
        )

        merged = merge_structural_reference_results(
            [
                (0, result_1),
                (0, result_2),
            ]
        )

        assert len(merged.references) == 1
        assert merged.references[0].canonical_key == "section|2"

        assert len(merged.links) == 2
        assert merged.unique_keys["section|2"].start_offset == 10

        chosen_unique_link = merged.unique_links["section|2"]
        assert chosen_unique_link.target_span == (100, 109)
        assert chosen_unique_link.match_strategy == "exact"

    def test_shifts_offsets_before_merging_and_sorts_results(self) -> None:
        later_reference = _make_reference(
            label="Section 3",
            canonical_label="Section 3",
            normalized_key="section 3",
            canonical_key="section|3",
            start_offset=50,
            end_offset=59,
        )
        earlier_reference = _make_reference(
            label="Section 1",
            canonical_label="Section 1",
            normalized_key="section 1",
            canonical_key="section|1",
            start_offset=5,
            end_offset=14,
        )

        later_link = _make_link(
            label="Section 3",
            canonical_label="Section 3",
            normalized_key="section 3",
            canonical_key="section|3",
            reference_span=(70, 79),
            target_span=(150, 159),
        )
        earlier_link = _make_link(
            label="Section 1",
            canonical_label="Section 1",
            normalized_key="section 1",
            canonical_key="section|1",
            reference_span=(15, 24),
            target_span=(80, 89),
        )

        later_result = StructuralReferenceResolutionResult(
            references=[later_reference],
            links=[later_link],
            unique_keys={later_reference.canonical_key: later_reference},
            unique_links={later_link.canonical_key: later_link},
        )
        earlier_result = StructuralReferenceResolutionResult(
            references=[earlier_reference],
            links=[earlier_link],
            unique_keys={earlier_reference.canonical_key: earlier_reference},
            unique_links={earlier_link.canonical_key: earlier_link},
        )

        merged = merge_structural_reference_results(
            [
                (100, later_result),
                (0, earlier_result),
            ]
        )

        assert [ref.canonical_key for ref in merged.references] == ["section|1", "section|3"]
        assert [link.canonical_key for link in merged.links] == ["section|1", "section|3"]

        shifted_later_reference = next(ref for ref in merged.references if ref.canonical_key == "section|3")
        assert shifted_later_reference.start_offset == 150
        assert shifted_later_reference.end_offset == 159

        shifted_later_link = next(link for link in merged.links if link.canonical_key == "section|3")
        assert shifted_later_link.reference_span == (170, 179)
        assert shifted_later_link.target_span == (250, 259)
