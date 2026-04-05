from __future__ import annotations

from plainera_rag_demo.answering import DemoAnswerGenerator
from plainera_rag_demo.chunking import FixedWindowChunker
from plainera_rag_demo.composition.embedder import build_openai_embedder
from plainera_rag_demo.pipelines.baseline import BaselineRagPipeline
from plainera_rag_demo.pipelines.grounded import GroundedRagPipeline
from plainera_rag_demo.retrieval import FaissVectorStore
from plainera_rag_demo.settings import RagDemoSettings, rag_demo_settings


def build_baseline_pipeline(settings: RagDemoSettings = rag_demo_settings) -> BaselineRagPipeline:
    """Build the baseline RAG pipeline from package settings.

    The baseline pipeline uses deterministic fixed-window chunking, OpenAI
    embeddings, FAISS-backed retrieval, and a simple demo answer generator.

    Args:
        settings: RAG demo settings providing chunking and embedding
            configuration.

    Returns:
        A configured ``BaselineRagPipeline`` instance ready to index documents
        and answer questions.
    """
    embedder = build_openai_embedder(settings)

    return BaselineRagPipeline(
        chunker=FixedWindowChunker(
            chunk_size=settings.baseline_chunk_size,
            chunk_overlap=settings.baseline_chunk_overlap,
        ),
        vector_store=FaissVectorStore(embedder=embedder),
        answer_generator=DemoAnswerGenerator(),
    )


def build_grounded_pipeline(settings: RagDemoSettings = rag_demo_settings) -> GroundedRagPipeline:
    """Build the grounded RAG pipeline from package settings.

    The grounded pipeline uses deterministic fixed-window chunking, OpenAI
    embeddings, FAISS-backed retrieval, and a simple demo answer generator.

    Args:
        settings: RAG demo settings providing chunking and embedding
            configuration.

    Returns:
        A configured ``groundedRagPipeline`` instance ready to index documents
        and answer questions.
    """
    embedder = build_openai_embedder(settings)

    return GroundedRagPipeline(
        chunker=FixedWindowChunker(
            chunk_size=settings.grounded_chunk_size,
            chunk_overlap=settings.grounded_chunk_overlap,
        ),
        vector_store=FaissVectorStore(embedder=embedder),
        answer_generator=DemoAnswerGenerator(),
    )
