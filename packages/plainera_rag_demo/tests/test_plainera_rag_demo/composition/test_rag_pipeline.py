from __future__ import annotations

import pytest
from plainera_rag_demo.agentic.types import GroundedEvidenceAssessment
from plainera_rag_demo.common import DemoDocument
from plainera_rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage


class _GroundingStageStub:
    async def ground_documents(self, documents):
        return tuple(documents)


class _VectorStoreStub:
    def __init__(self, *, retrieval_batches):
        self._retrieval_batches = list(retrieval_batches)
        self.calls: list[int] = []

    def index_chunks(self, chunks):
        return object()

    def retrieve(self, *, index, question, top_k):
        self.calls.append(top_k)
        return self._retrieval_batches.pop(0)

class _RetryThenProceedOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, *, question, retrieved_chunks, has_second_pass_available):
        self.calls += 1

        if self.calls == 1:
            return GroundedEvidenceAssessment(
                action="retry_once",
                outcome="answer_with_warning",
                sufficient_evidence=False,
                ambiguity_detected=False,
                requested_second_pass=True,
                answer_text="Initial retrieval was insufficient; one additional retrieval pass is required.",
                abstain_reason=None,
                warning_reason="Initial retrieval did not include a usable grounding payload.",
                reasoning_notes=("retry",),
                selected_audit_bindings=("doc-1",),
                selected_audit_spans=((0, 10),),
            )

        return GroundedEvidenceAssessment(
            action="proceed",
            outcome="answer",
            sufficient_evidence=True,
            ambiguity_detected=False,
            requested_second_pass=False,
            answer_text="MPS stands for Metropolitan Police Service.",
            abstain_reason=None,
            warning_reason=None,
            reasoning_notes=("proceed",),
            selected_audit_bindings=("doc-1",),
            selected_audit_spans=((0, 10),),
        )


class _AbstainOrchestrator:
    def assess(self, *, question, retrieved_chunks, has_second_pass_available):
        return GroundedEvidenceAssessment(
            action="abstain",
            outcome="abstain",
            sufficient_evidence=False,
            ambiguity_detected=False,
            requested_second_pass=False,
            answer_text=None,
            abstain_reason="No usable deterministic grounding payload was available after retrieval.",
            warning_reason=None,
            reasoning_notes=("abstain",),
            selected_audit_bindings=("doc-1",),
            selected_audit_spans=((0, 10),),
        )


class _ProceedOrchestrator:
    def assess(self, *, question, retrieved_chunks, has_second_pass_available):
        return GroundedEvidenceAssessment(
            action="proceed",
            outcome="answer_with_warning",
            sufficient_evidence=True,
            ambiguity_detected=False,
            requested_second_pass=False,
            answer_text="MPS stands for Metropolitan Police Service.",
            abstain_reason=None,
            warning_reason="Answer supported by structured grounded evidence.",
            reasoning_notes=("proceed",),
            selected_audit_bindings=("doc-1",),
            selected_audit_spans=((0, 10),),
        )


class _ResolveServiceStub:
    def __init__(self, response) -> None:
        self._response = response
        self.calls = []

    async def resolve(self, payload):
        self.calls.append(payload)
        return self._response


class _ResolveResponseStub:
    def __init__(self, json_text: str) -> None:
        self._json_text = json_text

    def model_dump_json(self, *, indent: int) -> str:
        assert indent == 2
        return self._json_text


class TestGroundedRagPipeline:
    @pytest.mark.anyio
    async def test_pipeline_retries_once_when_assessment_requests_retry(self,
                                                                        retrieved_chunk,
                                                                        demo_chunk,
                                                                        chunk_stub) -> None:
        vector_store = _VectorStoreStub(
            retrieval_batches=[
                (
                    retrieved_chunk(
                        chunk=demo_chunk(
                            chunk_id="doc-1:0",
                            document_id="doc-1",
                            document_name="doc-1.txt",
                            ordinal=0,
                            start_offset=0,
                            end_offset=10,
                            text="weak",
                        ),
                        score=0.25,
                    ),
                ),
                (
                    retrieved_chunk(
                        chunk=demo_chunk(
                            chunk_id="doc-1:1",
                            document_id="doc-1",
                            document_name="doc-1.txt",
                            ordinal=1,
                            start_offset=10,
                            end_offset=20,
                            text="strong",
                        ),
                        score=0.9,
                    ),
                ),
            ]
        )
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStageStub(),
            chunker=chunk_stub,
            vector_store=vector_store,
            evidence_orchestrator=_RetryThenProceedOrchestrator(),
        )

        index = await pipeline.index_documents(
            (
                DemoDocument(
                    document_id="doc-1",
                    name="doc-1.txt",
                    text="text",
                ),
            )
        )

        result = pipeline.answer_question(
            index=index,
            question="What does MPS mean?",
            top_k=5,
        )

        assert vector_store.calls == [5, 10]
        assert result.outcome == "answer"
        assert result.answer == "MPS stands for Metropolitan Police Service."

    @pytest.mark.anyio
    async def test_pipeline_does_not_generate_answer_when_assessment_abstains(self,
                                                                              retrieved_chunk,
                                                                              demo_chunk,
                                                                              chunk_stub) -> None:
        vector_store = _VectorStoreStub(
            retrieval_batches=[
                (
                    retrieved_chunk(
                        chunk=demo_chunk(
                            chunk_id="doc-1:0",
                            document_id="doc-1",
                            document_name="doc-1.txt",
                            ordinal=0,
                            start_offset=0,
                            end_offset=10,
                            text="ambiguous",
                        ),
                        score=0.2,
                    ),
                ),
            ]
        )
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStageStub(),
            chunker=chunk_stub,
            vector_store=vector_store,
            evidence_orchestrator=_AbstainOrchestrator(),
        )

        index = await pipeline.index_documents(
            (
                DemoDocument(
                    document_id="doc-1",
                    name="doc-1.txt",
                    text="text",
                ),
            )
        )

        result = pipeline.answer_question(
            index=index,
            question="What does GP mean?",
            top_k=5,
        )

        assert vector_store.calls == [5]
        assert result.outcome == "abstain"
        assert result.answer is None
        assert result.assessment.abstain_reason == (
            "No usable deterministic grounding payload was available after retrieval."
        )

    @pytest.mark.anyio
    async def test_pipeline_generates_answer_when_assessment_proceeds(self,
                                                                      retrieved_chunk,
                                                                      demo_chunk,
                                                                      chunk_stub) -> None:
        retrieved_chunk = retrieved_chunk(
            chunk=demo_chunk(
                chunk_id="doc-1:0",
                document_id="doc-1",
                document_name="doc-1.txt",
                ordinal=0,
                start_offset=0,
                end_offset=10,
                text="grounded",
            ),
            score=0.95,
        )
        vector_store = _VectorStoreStub(
            retrieval_batches=[
                (retrieved_chunk,),
            ]
        )
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStageStub(),
            chunker=chunk_stub,
            vector_store=vector_store,
            evidence_orchestrator=_ProceedOrchestrator(),
        )

        index = await pipeline.index_documents(
            (
                DemoDocument(
                    document_id="doc-1",
                    name="doc-1.txt",
                    text="text",
                ),
            )
        )

        result = pipeline.answer_question(
            index=index,
            question="What does MPS stand for?",
            top_k=5,
        )

        assert vector_store.calls == [5]
        assert result.outcome == "answer_with_warning"
        assert result.answer == "MPS stands for Metropolitan Police Service."


class TestResolveBackedGroundingStage:
    @pytest.mark.anyio
    async def test_resolve_backed_grounding_stage_prepends_deterministic_context(self) -> None:
        resolve_service = _ResolveServiceStub(
            _ResolveResponseStub(
                '{\n  "acronyms": [{"acronym": "MPS"}],\n  "defined_terms": [],\n  "structural_references": []\n}'
            )
        )
        stage = ResolveBackedGroundingStage(resolve_service=resolve_service)

        grounded_documents = await stage.ground_documents(
            (
                DemoDocument(
                    document_id="police",
                    name="police.txt",
                    text="The Metropolitan Police Service (MPS) operates in London.",
                ),
            )
        )

        assert len(resolve_service.calls) == 1
        grounded = grounded_documents[0]

        assert grounded.document_id == "police"
        assert grounded.name == "police.txt"
        assert grounded.text.startswith("[DETERMINISTIC_GROUNDING]\n{\n")
        assert '"acronyms": [{"acronym": "MPS"}]' in grounded.text
        assert "\n\n[DOCUMENT]\n" in grounded.text
        assert "The Metropolitan Police Service (MPS) operates in London." in grounded.text
