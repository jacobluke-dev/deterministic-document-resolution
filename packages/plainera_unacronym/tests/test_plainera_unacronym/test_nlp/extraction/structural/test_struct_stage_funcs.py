from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.stage_funcs import (
    st_build_structural_anchor_index,
    st_build_structural_reference_entries,
    st_detect_structural_references,
    st_extract_structural_anchors,
    st_link_structural_references,
)
from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState


class _DetCfg:
    pass


class TestStructuralCanonicalizationFlow:
    def test_reference_and_anchor_keys_align_for_schedule_case_variant(self):
        """
        NOTE FOR TEST:
        Reviewed UN-99 and determined no additional linker logic is required. Existing upstream
        canonicalisation already standardises obvious formatting variants onto the same structural
        lookup key, so exact canonical-key linking remains the correct contract.
        """
        text = """
    See Schedule A for details.

    SCHEDULE A: Services Description
    """.strip()

        state = StructuralFlowState(
            text=text,
            det_cfg=_DetCfg(),
            ext_cfg=StructuralReferenceExtractionConfig(),
        )

        st_detect_structural_references(state)
        st_build_structural_reference_entries(state)
        st_extract_structural_anchors(state)
        st_build_structural_anchor_index(state)
        st_link_structural_references(state)

        assert len(state.reference_entries) == 2
        assert [entry.canonical_key for entry in state.reference_entries] == [
            "schedule_a",
            "schedule_a",
        ]

        assert "schedule_a" in state.anchor_index
        assert len(state.anchor_index["schedule_a"]) == 1

        assert len(state.link_entries) == 2
        assert [link.canonical_key for link in state.link_entries] == [
            "schedule_a",
            "schedule_a",
        ]
        assert all(link.target_span is not None for link in state.link_entries)
        assert [link.confidence for link in state.link_entries] == [1.0, 0.5]
