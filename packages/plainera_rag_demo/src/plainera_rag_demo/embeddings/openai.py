from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openai import OpenAI

from plainera_rag_demo.common.types import FloatMatrix
from plainera_rag_demo.contracts import Embedder


class OpenAIEmbedder(Embedder):
    """Embed text with the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        dimensions: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> FloatMatrix:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)

        self._validate_inputs(text_list)

        rows: list[list[float]] = []

        for start in range(0, len(text_list), self._batch_size):
            batch = text_list[start : start + self._batch_size]

            if self._dimensions is None:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                    encoding_format="float",
                )
            else:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                    encoding_format="float",
                    dimensions=self._dimensions,
                )

            for item in sorted(response.data, key=lambda item: item.index):
                rows.append(item.embedding)

        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[0] != len(text_list):
            raise ValueError("embedding response row count did not match input row count")

        return matrix

    @staticmethod
    def _validate_inputs(texts: Sequence[str]) -> None:
        for text in texts:
            if not text or not text.strip():
                raise ValueError("embed_texts does not accept empty or whitespace-only strings")
