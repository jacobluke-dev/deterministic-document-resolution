from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagDemoSettings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    embedding_dimensions: int | None = None
    baseline_chunk_size: int = 1_000
    baseline_chunk_overlap: int = 150
    baseline_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


rag_demo_settings = RagDemoSettings(
    openai_api_key=os.environ["OPENAI_API_KEY"],
)
