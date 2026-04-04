from __future__ import annotations

from plainera_rag_demo.chunking import FixedWindowChunker
from plainera_rag_demo.composition.embedder import build_openai_embedder
from plainera_rag_demo.pipelines.baseline import BaselineRagPipeline
from plainera_rag_demo.retrieval import InMemoryVectorStore
from plainera_rag_demo.settings import RagDemoSettings, rag_demo_settings
from tests.test_plainera_rag_demo.pipelines.test_rag_base import FakeAnswerGenerator


def build_baseline_pipeline(settings: RagDemoSettings = rag_demo_settings) -> BaselineRagPipeline:
    embedder = build_openai_embedder(settings)

    return BaselineRagPipeline(
        chunker=FixedWindowChunker(
            chunk_size=settings.baseline_chunk_size,
            chunk_overlap=settings.baseline_chunk_overlap,
        ),
        vector_store=InMemoryVectorStore(embedder=embedder),
        answer_generator=FakeAnswerGenerator(),
    )
