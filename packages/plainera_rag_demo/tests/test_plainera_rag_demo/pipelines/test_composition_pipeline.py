from __future__ import annotations

from dataclasses import dataclass

import pytest
from plainera_rag_demo.agentic.types import GroundedEvidenceAssessment
from plainera_rag_demo.answering import DemoAnswerGenerator
from plainera_rag_demo.chunking import FixedWindowChunker
from plainera_rag_demo.common import DemoDocument
from plainera_rag_demo.composition.pipeline import build_baseline_pipeline, build_grounded_pipeline
from plainera_rag_demo.pipelines.baseline import BaselineRagPipeline
from plainera_rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage
from plainera_rag_demo.retrieval import FaissVectorStore
from plainera_rag_demo.settings import RagDemoSettings


class _DummyResolveService:
    pass

@dataclass(frozen=True, slots=True)
class _Chunk:
    text: str
    document_id: str = "doc-1"
    start: int = 0
    end: int = 10


class _GroundingStage:
    async def ground_documents(self, documents):
        return tuple(documents)


class _Chunker:
    def chunk_documents(self, documents):
        return (_Chunk(text=documents[0].text),)


class _VectorStore:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[int] = []

    def index_chunks(self, chunks):
        return object()

    def retrieve(self, *, index, question, top_k):
        self.calls.append(top_k)
        return self._responses.pop(0)


class _AnswerGenerator:
    def __init__(self) -> None:
        self.called = False

    def generate_answer(self, *, question, retrieved_chunks):
        self.called = True
        return f"{question} :: answer"


class _RetryThenAnswerOrchestrator:
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
                warning_reason="Initial retrieval support was limited.",
                reasoning_notes=("retry",),
                selected_audit_bindings=(),
                selected_audit_spans=(),
            )

        return GroundedEvidenceAssessment(
            action="proceed",
            outcome="answer",
            sufficient_evidence=True,
            ambiguity_detected=False,
            requested_second_pass=False,
            abstain_reason=None,
            warning_reason=None,
            reasoning_notes=("answer",),
            selected_audit_bindings=(),
            selected_audit_spans=(),
        )


class _AbstainOrchestrator:
    def assess(self, *, question, retrieved_chunks, has_second_pass_available):
        return GroundedEvidenceAssessment(
            action="abstain",
            outcome="abstain",
            sufficient_evidence=False,
            ambiguity_detected=True,
            requested_second_pass=False,
            abstain_reason="Grounded evidence remained insufficient or ambiguous.",
            warning_reason=None,
            reasoning_notes=("abstain",),
            selected_audit_bindings=(),
            selected_audit_spans=(),
        )


class TestGroundedRagPipeline:

    @pytest.mark.asyncio
    async def test_retries_once_before_answering(self) -> None:
        vector_store = _VectorStore(
            responses=[
                (_Chunk(text="weak"),),
                (_Chunk(text="strong-1"), _Chunk(text="strong-2")),
            ]
        )
        answer_generator = _AnswerGenerator()
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStage(),
            chunker=_Chunker(),
            vector_store=vector_store,
            answer_generator=answer_generator,
            evidence_orchestrator=_RetryThenAnswerOrchestrator(),
        )

        index = await pipeline.index_documents(
            [DemoDocument(document_id="doc-1", name="Doc 1", text="text")]
        )
        result = pipeline.answer_question(index=index, question="What does it mean?")

        assert vector_store.calls == [5, 10]
        assert answer_generator.called is True
        assert result.outcome == "answer"
        assert result.answer == "What does it mean? :: answer"

    @pytest.mark.asyncio
    async def test_does_not_generate_answer_when_outcome_is_abstain(self) -> None:
        vector_store = _VectorStore(
            responses=[
                (_Chunk(text='{"chosen_meaning_id": null, "resolution_method": "unresolved"}'),),
            ]
        )
        answer_generator = _AnswerGenerator()
        pipeline = GroundedRagPipeline(
            grounding_stage=_GroundingStage(),
            chunker=_Chunker(),
            vector_store=vector_store,
            answer_generator=answer_generator,
            evidence_orchestrator=_AbstainOrchestrator(),
        )

        index = await pipeline.index_documents(
            [DemoDocument(document_id="doc-1", name="Doc 1", text="text")]
        )
        result = pipeline.answer_question(index=index, question="What does it mean?")

        assert answer_generator.called is False
        assert result.outcome == "abstain"
        assert result.answer is None

class TestBuildBaselinePipeline:
    def test_builds_baseline_pipeline_with_expected_components(self) -> None:
        settings = RagDemoSettings(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            embedding_batch_size=8,
            baseline_chunk_size=400,
            baseline_chunk_overlap=50,
            grounded_chunk_size=600,
            grounded_chunk_overlap=75,
        )

        pipeline = build_baseline_pipeline(settings)

        assert isinstance(pipeline, BaselineRagPipeline)
        assert isinstance(pipeline._chunker, FixedWindowChunker)
        assert isinstance(pipeline._vector_store, FaissVectorStore)
        assert isinstance(pipeline._answer_generator, DemoAnswerGenerator)


class TestBuildGroundedPipeline:
    def test_builds_grounded_pipeline_with_expected_components(self) -> None:
        settings = RagDemoSettings(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            embedding_batch_size=8,
            baseline_chunk_size=400,
            baseline_chunk_overlap=50,
            grounded_chunk_size=600,
            grounded_chunk_overlap=75,
        )
        resolve_service = _DummyResolveService()

        pipeline = build_grounded_pipeline(
            resolve_service=resolve_service,
            settings=settings,
        )

        assert isinstance(pipeline, GroundedRagPipeline)
        assert isinstance(pipeline._grounding_stage, ResolveBackedGroundingStage)
        assert pipeline._grounding_stage._resolve_service is resolve_service
        assert isinstance(pipeline._chunker, FixedWindowChunker)
        assert isinstance(pipeline._vector_store, FaissVectorStore)
        assert isinstance(pipeline._answer_generator, DemoAnswerGenerator)
