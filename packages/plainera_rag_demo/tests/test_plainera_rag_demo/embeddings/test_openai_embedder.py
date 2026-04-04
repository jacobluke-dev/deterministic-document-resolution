from __future__ import annotations

import numpy as np
import pytest

from plainera_rag_demo.embeddings import OpenAIEmbedder


class _FakeEmbeddingItem:
    def __init__(self, *, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: list[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddingsApi:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        self.calls.append(kwargs)
        batch = kwargs["input"]
        assert isinstance(batch, list)

        data = [
            _FakeEmbeddingItem(
                index=i,
                embedding=[float(len(text)), float(i + 1)],
            )
            for i, text in enumerate(batch)
        ]
        return _FakeEmbeddingResponse(data)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsApi()


class TestOpenAIEmbedder:
    def test_embeds_texts_in_batches_and_returns_float32_matrix(self) -> None:
        client = _FakeOpenAIClient()
        embedder = OpenAIEmbedder(
            client=client,
            model="text-embedding-3-small",
            batch_size=2,
        )

        matrix = embedder.embed_texts(["alpha", "beta", "gamma"])

        assert matrix.dtype == np.float32
        assert matrix.shape == (3, 2)
        assert matrix.tolist() == [
            [5.0, 1.0],
            [4.0, 2.0],
            [5.0, 1.0],
        ]

        assert len(client.embeddings.calls) == 2
        assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"
        assert client.embeddings.calls[0]["input"] == ["alpha", "beta"]
        assert client.embeddings.calls[1]["input"] == ["gamma"]

    def test_rejects_blank_inputs(self) -> None:
        client = _FakeOpenAIClient()
        embedder = OpenAIEmbedder(client=client)

        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedder.embed_texts(["alpha", "   "])

    def test_returns_empty_matrix_for_empty_input(self) -> None:
        client = _FakeOpenAIClient()
        embedder = OpenAIEmbedder(client=client)

        matrix = embedder.embed_texts([])

        assert matrix.dtype == np.float32
        assert matrix.shape == (0, 0)
        assert client.embeddings.calls == []
