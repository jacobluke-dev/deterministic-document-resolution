from __future__ import annotations

from plainera_rag_demo.answering import DemoAnswerGenerator
from plainera_rag_demo.chunking import FixedWindowChunker
from plainera_rag_demo.composition.pipeline import build_baseline_pipeline, build_grounded_pipeline
from plainera_rag_demo.pipelines.baseline import BaselineRagPipeline
from plainera_rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage
from plainera_rag_demo.retrieval import FaissVectorStore
from plainera_rag_demo.settings import RagDemoSettings


class _DummyResolveService:
    pass


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
