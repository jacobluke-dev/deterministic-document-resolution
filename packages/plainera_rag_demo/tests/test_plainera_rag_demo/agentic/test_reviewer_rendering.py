from __future__ import annotations

from plainera_rag_demo.agentic.reviewer_rendering import (
    extract_ambiguity_indicators,
    first_candidate_source_ref,
    summarize_acronyms,
    summarize_defined_terms,
    summarize_grounding_payload,
    summarize_structural_references,
)


class TestSummarizeAcronyms:
    def test_prefers_selected_definition_and_reason(self) -> None:
        result = summarize_acronyms(
            [
                {
                    "acronym": "MPS",
                    "selected": {
                        "definition": "Metropolitan Police Service",
                        "reason": "in_document_definition",
                    },
                    "candidates": [{"source_ref": "text_span:4-31"}],
                    "conflict": False,
                }
            ]
        )

        assert result == [
            {
                "acronym": "MPS",
                "selected_definition": "Metropolitan Police Service",
                "selection_reason": "in_document_definition",
                "conflict": False,
                "source_ref": "text_span:4-31",
            }
        ]

    def test_falls_back_to_first_definition_text_when_selected_definition_missing(self) -> None:
        result = summarize_acronyms(
            [
                {
                    "acronym": "MPS",
                    "definitions": [{"text": "Master Printing Service"}],
                    "candidates": [{"source_ref": "text_span:10-33"}],
                    "conflict": True,
                }
            ]
        )

        assert result == [
            {
                "acronym": "MPS",
                "selected_definition": "Master Printing Service",
                "selection_reason": None,
                "conflict": True,
                "source_ref": "text_span:10-33",
            }
        ]


class TestSummarizeDefinedTerms:
    def test_extracts_chosen_definition_text_and_resolution_metadata(self) -> None:
        result = summarize_defined_terms(
            [
                {
                    "surface": "Operational Directive",
                    "normalized_key": "operational_directive",
                    "chosen_definition_span": {
                        "text": "any written instruction issued under Section 4",
                    },
                    "resolution_method": "tier1",
                    "resolved": True,
                }
            ]
        )

        assert result == [
            {
                "surface": "Operational Directive",
                "normalized_key": "operational_directive",
                "chosen_definition_text": "any written instruction issued under Section 4",
                "resolution_method": "tier1",
                "resolved": True,
            }
        ]


class TestSummarizeStructuralReferences:
    def test_extracts_reference_and_target_metadata(self) -> None:
        result = summarize_structural_references(
            [
                {
                    "reference_span": {"text": "Section 4.3", "start": 100, "end": 111},
                    "target_span": {"start": 450, "end": 520},
                    "match_strategy": "exact_section_label",
                    "resolved": True,
                }
            ]
        )

        assert result == [
            {
                "reference_span": {"text": "Section 4.3", "start": 100, "end": 111},
                "target_span": {"start": 450, "end": 520},
                "match_strategy": "exact_section_label",
                "resolved": True,
            }
        ]


class TestSummarizeGroundingPayload:
    def test_reduces_payload_to_reviewer_relevant_shape(self) -> None:
        result = summarize_grounding_payload(
            {
                "acronyms": [
                    {
                        "acronym": "MPS",
                        "selected": {
                            "definition": "Metropolitan Police Service",
                            "reason": "in_document_definition",
                        },
                        "candidates": [{"source_ref": "text_span:4-31"}],
                        "conflict": False,
                    }
                ],
                "defined_terms": [
                    {
                        "surface": "Operational Directive",
                        "normalized_key": "operational_directive",
                        "chosen_definition_span": {
                            "text": "any written instruction issued under Section 4",
                        },
                        "resolution_method": "tier1",
                        "resolved": True,
                    }
                ],
                "structural_references": [
                    {
                        "reference_span": {"text": "Section 4.3", "start": 100, "end": 111},
                        "target_span": {"start": 450, "end": 520},
                        "match_strategy": "exact_section_label",
                        "resolved": True,
                    }
                ],
                "meta": {"processing_ms": 12},
            }
        )

        assert result["acronyms"][0]["selected_definition"] == "Metropolitan Police Service"
        assert result["defined_terms"][0]["chosen_definition_text"] == (
            "any written instruction issued under Section 4"
        )
        assert result["structural_references"][0]["match_strategy"] == "exact_section_label"


class TestFirstCandidateSourceRef:
    def test_returns_first_source_ref_when_present(self) -> None:
        result = first_candidate_source_ref(
            {
                "candidates": [
                    {"source_ref": "text_span:4-31"},
                    {"source_ref": "text_span:40-60"},
                ]
            }
        )

        assert result == "text_span:4-31"

    def test_returns_none_when_candidates_missing(self) -> None:
        assert first_candidate_source_ref({}) is None


class TestExtractAmbiguityIndicators:
    def test_counts_unresolved_acronyms_defined_terms_and_structural_references(self) -> None:
        result = extract_ambiguity_indicators(
            {
                "acronyms": [
                    {
                        "acronym": "MPS",
                        "selected": {
                            "definition": "Metropolitan Police Service",
                            "reason": "in_document_definition",
                        },
                    },
                    {
                        "acronym": "GP",
                        "selected": {
                            "definition": "",
                            "reason": "unresolved",
                        },
                    },
                ],
                "defined_terms": [
                    {
                        "surface": "Operational Directive",
                        "resolution_method": "tier1",
                        "resolved": True,
                    },
                    {
                        "surface": "Authority",
                        "resolution_method": "unresolved",
                        "resolved": False,
                    },
                ],
                "structural_references": [
                    {
                        "match_strategy": "exact_section_label",
                        "resolved": True,
                    },
                    {
                        "match_strategy": "unresolved",
                        "resolved": False,
                    },
                ],
            }
        )

        assert result == {
            "acronym_count": 2,
            "defined_term_count": 2,
            "structural_reference_count": 2,
            "unresolved_acronyms": 1,
            "unresolved_defined_terms": 1,
            "unresolved_structural_references": 1,
        }
