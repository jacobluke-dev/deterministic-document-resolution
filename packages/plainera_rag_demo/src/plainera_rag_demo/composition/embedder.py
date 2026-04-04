from __future__ import annotations

from openai import OpenAI

from plainera_rag_demo.embeddings.openai import OpenAIEmbedder
from plainera_rag_demo.settings import RagDemoSettings, rag_demo_settings


def build_openai_embedder(settings: RagDemoSettings = rag_demo_settings) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        client=OpenAI(api_key=settings.openai_api_key),
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        dimensions=settings.embedding_dimensions,
    )
