from __future__ import annotations

import pytest
from rag_demo.agentic.types import GroundedEvidenceAssessment

from rag_demo.common import DemoDocument, FixedWindowChunker
from rag_demo.composition.pipeline import build_baseline_pipeline, build_grounded_pipeline
from rag_demo.pipelines.baseline import BaselineRagPipeline
from rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage
from rag_demo.retrieval import FaissVectorStore
from rag_demo.settings import RagDemoSettings


class _DummyResolveService:
    pass


class _VectorStore:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[int] = []

    def index_chunks(self, chunks):
        return object()

    def retrieve(self, *, index, question, top_k):
        self.calls.append(top_k)
        return self._responses.pop(0)


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
                answer_text="Initial retrieval support was limited; one additional retrieval pass is required.",
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
            answer_text="MPS stands for Metropolitan Police Service.",
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
            answer_text=None,
            abstain_reason="Grounded evidence remained insufficient or ambiguous.",
            warning_reason=None,
            reasoning_notes=("abstain",),
            selected_audit_bindings=(),
            selected_audit_spans=(),
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

class _ResolveResponseStub:
    def __init__(self, json_text: str) -> None:
        self._json_text = json_text

    def model_dump_json(self, *, indent: int) -> str:
        assert indent == 2
        return self._json_text


class _ResolveServiceStub:
    def __init__(self, response) -> None:
        self._response = response
        self.calls = []

    async def resolve(self, payload):
        self.calls.append(payload)
        return self._response


class TestResolveBackedGroundingStage:
    @pytest.mark.asyncio
    async def test_prepends_deterministic_context(self) -> None:
        resolve_service = _ResolveServiceStub(
            _ResolveResponseStub(
                '{\n  "acronyms": [{"acronym": "MPS"}],\n  "defined_terms": [],\n  "structural_references": []\n}'
            )
        )
        stage = ResolveBackedGroundingStage(resolve_service=resolve_service)

        grounded_documents = await stage.ground_documents(
            [
                DemoDocument(
                    document_id="police",
                    name="police.txt",
                    text="The Metropolitan Police Service (MPS) operates in London.",
                )
            ]
        )

        assert len(resolve_service.calls) == 1

        grounded = grounded_documents[0]
        assert grounded.document_id == "police"
        assert grounded.name == "police.txt"
        assert grounded.text.startswith("[DETERMINISTIC_GROUNDING]\n{\n")
        assert '"acronyms": [{"acronym": "MPS"}]' in grounded.text
        assert "\n\n[DOCUMENT]\n" in grounded.text
        assert "The Metropolitan Police Service (MPS) operates in London." in grounded.text


class TestGroundedRagPipeline:

    @pytest.mark.asyncio
    async def test_generates_answer_when_assessment_proceeds(
        self,
        demo_chunk,
        retrieved_chunk,
        grounding_stage,
        chunker,
    ) -> None:
        chunk = demo_chunk(
            chunk_id="doc-1:0",
            document_id="doc-1",
            document_name="doc-1.txt",
            ordinal=0,
            start_offset=0,
            end_offset=10,
            text="grounded",
        )
        retrieved = retrieved_chunk(
            chunk=chunk,
            score=0.95,
        )
        vector_store = _VectorStore(
            responses=[
                (retrieved,),
            ]
        )
        pipeline = GroundedRagPipeline(
            grounding_stage=grounding_stage,
            chunker=chunker,
            vector_store=vector_store,
            evidence_orchestrator=_ProceedOrchestrator(),
        )

        index = await pipeline.index_documents(
            [DemoDocument(document_id="doc-1", name="Doc 1", text="text")]
        )
        result = pipeline.answer_question(index=index, question="What does it mean?")

        assert vector_store.calls == [5]
        assert result.outcome == "answer_with_warning"
        assert result.answer == "MPS stands for Metropolitan Police Service."

    @pytest.mark.asyncio
    async def test_retries_once_before_answering(self,
        demo_chunk,
        retrieved_chunk,
        grounding_stage,
        chunker,
    ) -> None:
        vector_store = _VectorStore(
            responses=[
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
                            text="strong-1",
                        ),
                        score=0.9,
                    ),
                    retrieved_chunk(
                        chunk=demo_chunk(
                            chunk_id="doc-1:2",
                            document_id="doc-1",
                            document_name="doc-1.txt",
                            ordinal=2,
                            start_offset=20,
                            end_offset=30,
                            text="strong-2",
                        ),
                        score=0.85,
                    ),
                ),
            ]
        )
        pipeline = GroundedRagPipeline(
            grounding_stage=grounding_stage,
            chunker=chunker,
            vector_store=vector_store,
            evidence_orchestrator=_RetryThenAnswerOrchestrator(),
        )

        index = await pipeline.index_documents(
            [DemoDocument(document_id="doc-1", name="Doc 1", text="text")]
        )
        result = pipeline.answer_question(index=index, question="What does it mean?")

        assert vector_store.calls == [5, 10]
        assert result.outcome == "answer"
        assert result.answer == "MPS stands for Metropolitan Police Service."

    @pytest.mark.asyncio
    async def test_does_not_generate_answer_when_outcome_is_abstain(self,
        demo_chunk,
        retrieved_chunk,
        grounding_stage,
        chunker,
    ) -> None:
        vector_store = _VectorStore(
            responses=[
                (
                    retrieved_chunk(
                        chunk=demo_chunk(
                            chunk_id="doc-1:0",
                            document_id="doc-1",
                            document_name="doc-1.txt",
                            ordinal=0,
                            start_offset=0,
                            end_offset=10,
                            text='{"chosen_meaning_id": null, "resolution_method": "unresolved"}',
                        ),
                        score=0.2,
                    ),
                ),
            ]
        )
        pipeline = GroundedRagPipeline(
            grounding_stage=grounding_stage,
            chunker=chunker,
            vector_store=vector_store,
            evidence_orchestrator=_AbstainOrchestrator(),
        )

        index = await pipeline.index_documents(
            [DemoDocument(document_id="doc-1", name="Doc 1", text="text")]
        )
        result = pipeline.answer_question(index=index, question="What does it mean?")

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
