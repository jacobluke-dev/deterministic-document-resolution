from __future__ import annotations

from typing import Any

from rag_demo.agentic.orchestrator import SingleAgentEvidenceOrchestrator
from rag_demo.agentic.types import GroundedEvidenceAssessment, GroundedEvidencePacket


class _ReviewerSpy:
    def __init__(self, assessment: GroundedEvidenceAssessment) -> None:
        self.assessment = assessment
        self.calls: list[dict[str, Any]] = []

    def review(
        self,
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        self.calls.append(
            {
                "evidence": evidence,
                "has_second_pass_available": has_second_pass_available,
            }
        )
        return self.assessment


class TestSingleAgentEvidenceOrchestrator:
    def test_builds_structured_evidence_packet_and_delegates_to_reviewer(self, demo_chunk, retrieved_chunk) -> None:
        reviewer = _ReviewerSpy(
            GroundedEvidenceAssessment(
                action="proceed",
                outcome="answer_with_warning",
                answer_text="Metropolitan Police Service (MPS) operates in London.",
                sufficient_evidence=True,
                ambiguity_detected=False,
                requested_second_pass=False,
                abstain_reason=None,
                warning_reason="Answer supported by structured grounded evidence.",
                reasoning_notes=("reviewed",),
                selected_audit_bindings=("police",),
                selected_audit_spans=((0, 1000),),
            )
        )
        orchestrator = SingleAgentEvidenceOrchestrator(reviewer=reviewer)

        assessment = orchestrator.assess(
            question="What does MPS mean?",
            retrieved_chunks=(
                retrieved_chunk(
                    chunk=demo_chunk(
                        chunk_id="police:0",
                        document_id="police",
                        document_name="police.txt",
                        ordinal=0,
                        start_offset=0,
                        end_offset=1000,
                        text=(
                            "[DETERMINISTIC_GROUNDING]\n"
                            '{"acronyms":[{"acronym":"MPS"}]}\n\n'
                            "[DOCUMENT]\n"
                            "The Metropolitan Police Service (MPS) operates in London."
                        ),
                    ),
                    score=0.91,
                ),
            ),
            has_second_pass_available=True,
        )

        assert assessment.outcome == "answer_with_warning"

        assert len(reviewer.calls) == 1
        call = reviewer.calls[0]

        assert call["has_second_pass_available"] is True

        evidence = call["evidence"]
        assert evidence.question == "What does MPS mean?"
        assert len(evidence.documents) == 1

        document = evidence.documents[0]
        assert document.document_id == "police"
        assert document.document_name == "police.txt"
        assert document.chunk_id == "police:0"
        assert document.chunk_span == (0, 1000)
        assert document.score == 0.91
        assert document.grounding_payload == {"acronyms": [{"acronym": "MPS"}]}
        assert document.source_excerpt == "The Metropolitan Police Service (MPS) operates in London."

    def test_passes_through_second_pass_flag_to_reviewer(self) -> None:
        reviewer = _ReviewerSpy(
            GroundedEvidenceAssessment(
                action="retry_once",
                outcome="answer_with_warning",
                answer_text="Initial evidence was insufficient; one additional retrieval pass is requested.",
                sufficient_evidence=False,
                ambiguity_detected=False,
                requested_second_pass=True,
                abstain_reason=None,
                warning_reason="Initial retrieval did not include a usable grounding payload.",
                reasoning_notes=("retry",),
                selected_audit_bindings=(),
                selected_audit_spans=(),
            )
        )
        orchestrator = SingleAgentEvidenceOrchestrator(reviewer=reviewer)

        assessment = orchestrator.assess(
            question="What does GP mean?",
            retrieved_chunks=(),
            has_second_pass_available=False,
        )

        assert assessment.action == "retry_once"
        assert len(reviewer.calls) == 1
        assert reviewer.calls[0]["has_second_pass_available"] is False


class TestSplitGroundingAndDocument:
    def test_returns_payload_and_source_excerpt_when_grounded_text_is_complete(self) -> None:
        text = (
            "[DETERMINISTIC_GROUNDING]\n"
            '{\n  "acronyms": [{"acronym": "MPS"}],\n  "defined_terms": [],\n  "structural_references": []\n}'
            "\n\n[DOCUMENT]\n"
            "The Metropolitan Police Service (MPS) operates in London."
        )

        payload, source_excerpt = SingleAgentEvidenceOrchestrator._split_grounding_and_document(text)

        assert payload == {
            "acronyms": [{"acronym": "MPS"}],
            "defined_terms": [],
            "structural_references": [],
        }
        assert source_excerpt == "The Metropolitan Police Service (MPS) operates in London."

    def test_returns_none_and_original_text_when_grounding_marker_is_absent(self) -> None:
        text = "Just the original source text with no grounding wrapper."

        payload, source_excerpt = SingleAgentEvidenceOrchestrator._split_grounding_and_document(text)

        assert payload is None
        assert source_excerpt == text

    def test_returns_none_and_original_text_when_document_marker_is_absent(self) -> None:
        text = (
            "[DETERMINISTIC_GROUNDING]\n"
            '{"acronyms": [{"acronym": "MPS"}]}'
        )

        payload, source_excerpt = SingleAgentEvidenceOrchestrator._split_grounding_and_document(text)

        assert payload is None
        assert source_excerpt == text

    def test_returns_none_and_source_excerpt_when_grounding_json_is_invalid(self) -> None:
        text = (
            "[DETERMINISTIC_GROUNDING]\n"
            '{"acronyms": [INVALID JSON]}'
            "\n\n[DOCUMENT]\n"
            "The Metropolitan Police Service (MPS) operates in London."
        )

        payload, source_excerpt = SingleAgentEvidenceOrchestrator._split_grounding_and_document(text)

        assert payload is None
        assert source_excerpt == "The Metropolitan Police Service (MPS) operates in London."
