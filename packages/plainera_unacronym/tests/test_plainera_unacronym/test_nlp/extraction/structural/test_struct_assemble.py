from __future__ import annotations

from plainera_unacronym.nlp.extraction.structural.assemble import (
    assemble_structural_reference_resolution_result,
)
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceResolutionResult,
)


class _DetCfg:
    pass


class TestAssembleStructuralReferenceResolutionResult:
    def test_assemble_structural_reference_resolution_result_returns_expected_result_type(self):
        state = StructuralFlowState(
            text="See Section 4.2.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        state.reference_entries = [
            StructuralReferenceEntry(
                kind="Section",
                label="4.2",
                canonical_label="4.2",
                normalized_key="section_4_2",
                canonical_key="section_4_2",
                start_offset=4,
                end_offset=15,
                provenance="structural_reference_detector",
            )
        ]

        out = assemble_structural_reference_resolution_result(state)

        assert isinstance(out, StructuralReferenceResolutionResult)

    def test_assemble_structural_reference_resolution_result_preserves_reference_order(self):
        state = StructuralFlowState(
            text="See Clause 7, Section 4.2 and Schedule A.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        state.reference_entries = [
            StructuralReferenceEntry(
                kind="Clause",
                label="7",
                canonical_label="7",
                normalized_key="clause_7",
                canonical_key="clause_7",
                start_offset=4,
                end_offset=12,
                provenance="structural_reference_detector",
            ),
            StructuralReferenceEntry(
                kind="Section",
                label="4.2",
                canonical_label="4.2",
                normalized_key="section_4_2",
                canonical_key="section_4_2",
                start_offset=14,
                end_offset=25,
                provenance="structural_reference_detector",
            ),
            StructuralReferenceEntry(
                kind="Schedule",
                label="A",
                canonical_label="A",
                normalized_key="schedule_a",
                canonical_key="schedule_a",
                start_offset=30,
                end_offset=40,
                provenance="structural_reference_detector",
            ),
        ]

        out = assemble_structural_reference_resolution_result(state)

        assert [ref.canonical_key for ref in out.references] == [
            "clause_7",
            "section_4_2",
            "schedule_a",
        ]

    def test_assemble_structural_reference_resolution_result_builds_unique_key_index(self):
        state = StructuralFlowState(
            text="See Section 4.2 and Schedule A.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        section_ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=4,
            end_offset=15,
            provenance="structural_reference_detector",
        )
        schedule_ref = StructuralReferenceEntry(
            kind="Schedule",
            label="A",
            canonical_label="A",
            normalized_key="schedule_a",
            canonical_key="schedule_a",
            start_offset=20,
            end_offset=30,
            provenance="structural_reference_detector",
        )

        state.reference_entries = [section_ref, schedule_ref]

        out = assemble_structural_reference_resolution_result(state)

        assert out.unique_keys == {
            "section_4_2": section_ref,
            "schedule_a": schedule_ref,
        }

    def test_assemble_structural_reference_resolution_result_keeps_first_resolution_for_duplicate_key(self):
        state = StructuralFlowState(
            text="Section 4.2 applies. Later, Section 4.2 is varied.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        first_ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=0,
            end_offset=11,
            provenance="structural_reference_detector",
        )
        second_ref = StructuralReferenceEntry(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            start_offset=28,
            end_offset=39,
            provenance="structural_reference_detector",
        )

        state.reference_entries = [first_ref, second_ref]

        out = assemble_structural_reference_resolution_result(state)

        assert out.references == [first_ref, second_ref]
        assert out.unique_keys == {"section_4_2": first_ref}

    def test_assemble_structural_reference_resolution_result_handles_empty_entries(self):
        state = StructuralFlowState(
            text="No structural references here.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        out = assemble_structural_reference_resolution_result(state)

        assert out.references == []
        assert out.unique_keys == {}
