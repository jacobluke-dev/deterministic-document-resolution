import json

from plainera_rag_demo.agentic.prompted_reviewer import PromptedGroundingReviewer
from plainera_rag_demo.agentic.types import GroundedEvidencePacket, GroundedEvidenceDocument


class TestReview:
    def test_prompted_reviewer_parses_valid_model_output(self) -> None:
        reviewer = PromptedGroundingReviewer(
            model_complete=lambda system, user: (
                '{"action":"proceed","outcome":"answer","sufficient_evidence":true,'
                '"ambiguity_detected":false,"requested_second_pass":false,'
                '"answer_text":"MPS stands for Metropolitan Police Service.",'
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
