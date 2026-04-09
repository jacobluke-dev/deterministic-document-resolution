from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from plainera_rag_demo.agentic.orchestrator import SingleAgentEvidenceOrchestrator, PromptedGroundingReviewer
from plainera_rag_demo.agentic.types import GroundedEvidenceAssessment, GroundedEvidencePacket, GroundedEvidenceDocument


@dataclass(frozen=True, slots=True)
class _DemoChunk:
    chunk_id: str
    document_id: str
    document_name: str
    ordinal: int
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True, slots=True)
class _RetrievedChunk:
    chunk: _DemoChunk
    score: float


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
    def test_builds_structured_evidence_packet_and_delegates_to_reviewer(self) -> None:
        reviewer = _ReviewerSpy(
            GroundedEvidenceAssessment(
                action="proceed",
                outcome="answer_with_warning",
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
                _RetrievedChunk(
                    chunk=_DemoChunk(
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

    def test_uses_plain_text_as_source_excerpt_when_grounding_marker_is_absent(self) -> None:
        reviewer = _ReviewerSpy(
            GroundedEvidenceAssessment(
                action="abstain",
                outcome="abstain",
                sufficient_evidence=False,
                ambiguity_detected=False,
                requested_second_pass=False,
                abstain_reason="No usable deterministic grounding payload was available after retrieval.",
                warning_reason=None,
                reasoning_notes=("abstain",),
                selected_audit_bindings=(),
                selected_audit_spans=(),
            )
        )
        orchestrator = SingleAgentEvidenceOrchestrator(reviewer=reviewer)

        orchestrator.assess(
            question="What is this document about?",
            retrieved_chunks=(
                _RetrievedChunk(
                    chunk=_DemoChunk(
                        chunk_id="doc-1:0",
                        document_id="doc-1",
                        document_name="doc-1.txt",
                        ordinal=0,
                        start_offset=0,
                        end_offset=120,
                        text="Just the original source text with no grounding wrapper.",
                    ),
                    score=0.5,
                ),
            ),
            has_second_pass_available=True,
        )

        evidence = reviewer.calls[0]["evidence"]
        document = evidence.documents[0]
        assert document.grounding_payload is None
        assert document.source_excerpt == "Just the original source text with no grounding wrapper."

    def test_prompted_reviewer_renders_structured_packet_with_question_grounding_and_excerpt(self) -> None:
        reviewer = PromptedGroundingReviewer(
            model_complete=lambda system, user: (
                '{"action":"proceed","outcome":"answer_with_warning","sufficient_evidence":true,'
                '"ambiguity_detected":false,"requested_second_pass":false,'
                '"abstain_reason":null,"warning_reason":"Grounded support present.",'
                '"reasoning_notes":["ok"],"selected_audit_bindings":["police"],'
                '"selected_audit_spans":[[0,1000]]}'
            )
        )

        prompt = reviewer._render_user_prompt(
            evidence=GroundedEvidencePacket(
                question="What does MPS mean?",
                documents=(
                    GroundedEvidenceDocument(
                        document_id="police",
                        document_name="police.txt",
                        chunk_id="police:0",
                        chunk_span=(0, 1000),
                        score=0.91,
                        grounding_payload={"acronyms": [{"acronym": "MPS"}]},
                        source_excerpt="The Metropolitan Police Service (MPS) operates in London.",
                    ),
                ),
            ),
            has_second_pass_available=True,
        )

        payload = json.loads(prompt)

        assert payload["question"] == "What does MPS mean?"
        assert payload["has_second_pass_available"] is True
        assert len(payload["documents"]) == 1

        document = payload["documents"][0]
        assert document["document_id"] == "police"
        assert document["grounding_present"] is True
        assert document["grounding_payload"] == {"acronyms": [{"acronym": "MPS"}]}
        assert document["source_excerpt"] == "The Metropolitan Police Service (MPS) operates in London."

    def test_prompted_reviewer_parses_valid_model_output(self) -> None:
        reviewer = PromptedGroundingReviewer(
            model_complete=lambda system, user: (
                '{"action":"proceed","outcome":"answer","sufficient_evidence":true,'
                '"ambiguity_detected":false,"requested_second_pass":false,'
                '"abstain_reason":null,"warning_reason":null,'
                '"reasoning_notes":["Grounding was sufficient."],'
                '"selected_audit_bindings":["police"],'
                '"selected_audit_spans":[[0,1000]]}'
            )
        )

        assessment = reviewer.review(
            evidence=GroundedEvidencePacket(
                question="What does MPS mean?",
                documents=(
                    GroundedEvidenceDocument(
                        document_id="police",
                        document_name="police.txt",
                        chunk_id="police:0",
                        chunk_span=(0, 1000),
                        score=0.91,
                        grounding_payload={"acronyms": [{"acronym": "MPS"}]},
                        source_excerpt="The Metropolitan Police Service (MPS) operates in London.",
                    ),
                ),
            ),
            has_second_pass_available=True,
        )

        assert assessment.action == "proceed"
        assert assessment.outcome == "answer"
        assert assessment.sufficient_evidence is True
        assert assessment.reasoning_notes == ("Grounding was sufficient.",)
        assert assessment.selected_audit_bindings == ("police",)
        assert assessment.selected_audit_spans == ((0, 1000),)

    def test_prompted_reviewer_falls_back_conservatively_on_malformed_json(self) -> None:
        reviewer = PromptedGroundingReviewer(
            model_complete=lambda system, user: "{not json"
        )

        assessment = reviewer.review(
            evidence=GroundedEvidencePacket(
                question="What does GP mean?",
                documents=(
                    GroundedEvidenceDocument(
                        document_id="doc-1",
                        document_name="doc-1.txt",
                        chunk_id="doc-1:0",
                        chunk_span=(0, 100),
                        score=0.4,
                        grounding_payload=None,
                        source_excerpt="General practitioner is abbreviated as GP in some contexts.",
                    ),
                ),
            ),
            has_second_pass_available=False,
        )

        assert assessment.action == "abstain"
        assert assessment.outcome == "abstain"
        assert assessment.sufficient_evidence is False
        assert assessment.abstain_reason == "Reviewer output was invalid after bounded review."


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
