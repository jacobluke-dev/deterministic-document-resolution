from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from plainera_rag_demo.common.types import FloatMatrix
from plainera_rag_demo.contracts.interfaces import Embedder, VectorStore, ChunkIndex
from plainera_rag_demo.common.models import DemoChunk, RetrievedChunk


@dataclass(frozen=True, slots=True)
class InMemoryChunkIndex(ChunkIndex):
    """Indexed chunk vectors for retrieval."""

    chunks: tuple[DemoChunk, ...]
    vectors: FloatMatrix


class InMemoryVectorStore(VectorStore):
    """Simple cosine-similarity retriever for demo use and tests."""
    """Simple cosine-similarity retriever for demo use and tests."""

    def __init__(self, *, embedder: Embedder) -> None:
        self._embedder = embedder

    def index_chunks(self, chunks: Sequence[DemoChunk]) -> InMemoryChunkIndex:
        chunk_tuple = tuple(chunks)
        if not chunk_tuple:
            return InMemoryChunkIndex(
                chunks=(),
                vectors=np.zeros((0, 0), dtype=np.float32),
            )

        raw_vectors = self._embedder.embed_texts([chunk.text for chunk in chunk_tuple])
        self._validate_embedding_matrix(raw_vectors, expected_rows=len(chunk_tuple))
        vectors = self._l2_normalise(raw_vectors)

        return InMemoryChunkIndex(
            chunks=chunk_tuple,
            vectors=vectors,
        )

    def retrieve(
        self,
        *,
        index: InMemoryChunkIndex,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if not index.chunks:
            return []

        query_matrix = self._embedder.embed_texts([question])
        self._validate_embedding_matrix(query_matrix, expected_rows=1)
        query_vector = self._l2_normalise(query_matrix)[0]

        scores = index.vectors @ query_vector
        ranked_indices = np.argsort(-scores, kind="stable")[: min(top_k, len(index.chunks))]

        return [
            RetrievedChunk(
                chunk=index.chunks[int(idx)],
                score=float(scores[int(idx)]),
            )
            for idx in ranked_indices
        ]

    @staticmethod
    def _validate_embedding_matrix(matrix: FloatMatrix, *, expected_rows: int) -> None:
        if matrix.ndim != 2:
            raise ValueError("embedder must return a 2D matrix")
        if matrix.shape[0] != expected_rows:
            raise ValueError("embedder returned unexpected number of rows")

    @staticmethod
    def _l2_normalise(matrix: FloatMatrix) -> FloatMatrix:
        if matrix.size == 0:
            return matrix.astype(np.float32, copy=False)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / safe_norms).astype(np.float32, copy=False)
