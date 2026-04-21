from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openai import OpenAI

from rag_demo.common import Embedder
from rag_demo.common.types import FloatMatrix



class OpenAIEmbedder(Embedder):
    """Embed text by calling the OpenAI embeddings API.

    This implementation batches requests to reduce API round-trips and returns
    embeddings as a float32 NumPy matrix suitable for downstream retrieval use.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        dimensions: int | None = None,
    ) -> None:
        """Initialise the embedder.

        Args:
            client: Configured OpenAI client instance.
            model: Embedding model name to use for requests.
            batch_size: Maximum number of texts to embed per API request.
            dimensions: Optional embedding dimension override supported by the
                selected model.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed the supplied texts and return a dense float matrix.

        Args:
            texts: Input texts to embed.

        Returns:
            A float32 matrix with one embedding row per input text. Returns an
            empty ``(0, 0)`` matrix when no texts are supplied.

        Raises:
            ValueError: If any input text is empty or whitespace-only, or if
                the embedding response row count does not match the input row
                count.
        """
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
        """Validate that all embedding inputs contain non-blank text.

        Args:
            texts: Text inputs to validate.

        Raises:
            ValueError: If any text is empty or whitespace-only.
        """
        for text in texts:
            if not text or not text.strip():
                raise ValueError("embed_texts does not accept empty or whitespace-only strings")
