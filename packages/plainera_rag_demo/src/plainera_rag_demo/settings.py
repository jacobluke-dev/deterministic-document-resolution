from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagDemoSettings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    embedding_dimensions: int | None = None
    baseline_chunk_size: int = 1_000
    baseline_chunk_overlap: int = 150
    baseline_top_k: int = 5
    grounded_chunk_size: int = 1_000
    grounded_chunk_overlap: int = 150
    grounded_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_rag_demo_settings() -> RagDemoSettings:
    """Return cached RAG demo settings loaded from environment sources."""
    return RagDemoSettings()  # type: ignore[call-arg]
