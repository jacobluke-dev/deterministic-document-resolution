from __future__ import annotations

import pytest

from plainera_unacronym.nlp.extraction.structural.assemble import (
    assemble_structural_reference_resolution_result,
)
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceResolutionResult, StructuralReferenceLink,
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

    def test_unique_links_prefers_resolved_link_over_earlier_unresolved_link(self):
        state = StructuralFlowState(
            text="Section 4.2 is mentioned twice.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        first_link = StructuralReferenceLink(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            reference_span=(0, 11,),
            target_span=None,
            match_strategy="forward",
            strength=0.0,
            provenance="structural_reference_linker",
        )
        second_link = StructuralReferenceLink(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            reference_span=(24, 35,),
            target_span=(100, 120,),
            match_strategy="forward",
            strength=1.0,
            provenance="structural_reference_linker",
        )

        state.link_entries = [first_link, second_link]

        out = assemble_structural_reference_resolution_result(state)

        assert out.links == [first_link, second_link]
        assert out.unique_links == {"section_4_2": second_link}

    def test_unique_links_preserves_first_when_all_unresolved(self):
        state = StructuralFlowState(
            text="Section 4.2 is mentioned twice.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        first_link = StructuralReferenceLink(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            reference_span=(0, 11,),
            target_span=None,
            match_strategy="forward",
            strength=0.0,
            provenance="structural_reference_linker",
        )
        second_link = StructuralReferenceLink(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            reference_span=(24, 35,),
            target_span=None,
            strength=0.0,
            provenance="structural_reference_linker",
        )

        state.link_entries = [first_link, second_link]

        out = assemble_structural_reference_resolution_result(state)

        assert out.links == [first_link, second_link]
        assert out.unique_links == {"section_4_2": first_link}

    def test_unique_links_keeps_first_when_multiple_resolved_links_exist(self):
        state = StructuralFlowState(
            text="Section 4.2 is mentioned twice.",
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        first_link = StructuralReferenceLink(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            reference_span=(0, 11,),
            target_span=(100, 120,),
            strength=1.0,
            provenance="structural_reference_linker",
        )
        second_link = StructuralReferenceLink(
            kind="Section",
            label="4.2",
            canonical_label="4.2",
            normalized_key="section_4_2",
            canonical_key="section_4_2",
            reference_span=(24, 35,),
            target_span=(200, 220,),
            strength=1.0,
            provenance="structural_reference_linker",
        )

        state.link_entries = [first_link, second_link]

        out = assemble_structural_reference_resolution_result(state)

        assert out.links == [first_link, second_link]
        assert out.unique_links == {"section_4_2": first_link}
