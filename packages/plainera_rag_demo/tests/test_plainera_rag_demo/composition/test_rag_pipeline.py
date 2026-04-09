from __future__ import annotations

from dataclasses import dataclass

import pytest
from plainera_rag_demo.agentic.types import GroundedEvidenceAssessment
from plainera_rag_demo.common import DemoDocument
from plainera_rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage


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


class _GroundingStageStub:
    async def ground_documents(self, documents):
        return tuple(documents)


class _ChunkerStub:
    def chunk_documents(self, documents):
        document = documents[0]
        return (
            _DemoChunk(
                chunk_id=f"{document.document_id}:0",
                document_id=document.document_id,
                document_name=document.name,
                ordinal=0,
                start_offset=0,
                end_offset=len(document.text),
                text=document.text,
            ),
        )


class _VectorStoreStub:
    def __init__(self, *, retrieval_batches):
        self._retrieval_batches = list(retrieval_batches)
        self.calls: list[int] = []

    def index_chunks(self, chunks):
        return object()

    def retrieve(self, *, index, question, top_k):
        self.calls.append(top_k)
        return self._retrieval_batches.pop(0)


class _AnswerGeneratorSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_answer(self, *, question, retrieved_chunks):
        self.calls.append(
            {
                "question": question,
                "retrieved_chunks": tuple(retrieved_chunks),
            }
        )
        return f"{question} :: answer"


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
    async def test_pipeline_retries_once_when_assessment_requests_retry(self) -> None:
        vector_store = _VectorStoreStub(
            retrieval_batches=[
                (
                    _RetrievedChunk(
                        chunk=_DemoChunk(
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
                    _RetrievedChunk(
                        chunk=_DemoChunk(
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
        answer_generator = _AnswerGeneratorSpy()
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStageStub(),
            chunker=_ChunkerStub(),
            vector_store=vector_store,
            answer_generator=answer_generator,
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
        assert len(answer_generator.calls) == 1
        assert result.outcome == "answer"
        assert result.answer == "What does MPS mean? :: answer"

    @pytest.mark.anyio
    async def test_pipeline_does_not_generate_answer_when_assessment_abstains(self) -> None:
        vector_store = _VectorStoreStub(
            retrieval_batches=[
                (
                    _RetrievedChunk(
                        chunk=_DemoChunk(
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
        answer_generator = _AnswerGeneratorSpy()
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStageStub(),
            chunker=_ChunkerStub(),
            vector_store=vector_store,
            answer_generator=answer_generator,
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
        assert answer_generator.calls == []
        assert result.outcome == "abstain"
        assert result.answer is None
        assert result.assessment.abstain_reason == (
            "No usable deterministic grounding payload was available after retrieval."
        )

    @pytest.mark.anyio
    async def test_pipeline_generates_answer_when_assessment_proceeds(self) -> None:
        retrieved_chunk = _RetrievedChunk(
            chunk=_DemoChunk(
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
        answer_generator = _AnswerGeneratorSpy()
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStageStub(),
            chunker=_ChunkerStub(),
            vector_store=vector_store,
            answer_generator=answer_generator,
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
        assert len(answer_generator.calls) == 1
        assert answer_generator.calls[0]["question"] == "What does MPS stand for?"
        assert answer_generator.calls[0]["retrieved_chunks"] == (retrieved_chunk,)
        assert result.outcome == "answer_with_warning"
        assert result.answer == "What does MPS stand for? :: answer"


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
