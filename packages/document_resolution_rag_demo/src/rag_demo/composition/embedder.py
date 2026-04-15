from __future__ import annotations

from openai import OpenAI

from rag_demo.embeddings.openai import OpenAIEmbedder
from rag_demo.settings import RagDemoSettings, get_rag_demo_settings


def build_openai_embedder(settings: RagDemoSettings | None = None) -> OpenAIEmbedder:
    """Build an OpenAI-backed embedder from package settings.

    Args:
        settings: RAG demo settings providing the API key, embedding model,
            batch size, and optional embedding dimensions.

    Returns:
        A configured ``OpenAIEmbedder`` instance ready for use in retrieval
        indexing and query embedding.
    """
    settings = settings or get_rag_demo_settings()
    return OpenAIEmbedder(
        client=OpenAI(api_key=settings.openai_api_key),
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        dimensions=settings.embedding_dimensions,
    )
