from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import faiss  # type: ignore[import-untyped]
import numpy as np

from plainera_rag_demo.common import DemoChunk, RetrievedChunk
from plainera_rag_demo.common.types import FloatMatrix
from plainera_rag_demo.contracts.interfaces import ChunkIndex, Embedder, VectorStore


@dataclass(frozen=True, slots=True)
class FaissChunkIndex(ChunkIndex):
    """FAISS-backed chunk index."""

    chunks: tuple[DemoChunk, ...]
    index: faiss.IndexFlatIP


class FaissVectorStore(VectorStore):
    """FAISS cosine-similarity retriever for demo use."""

    def __init__(self, *, embedder: Embedder) -> None:
        self._embedder = embedder

    def index_chunks(self, chunks: Sequence[DemoChunk]) -> FaissChunkIndex:
        chunk_tuple = tuple(chunks)
        if not chunk_tuple:
            return FaissChunkIndex(
                chunks=(),
                index=faiss.IndexFlatIP(0),
            )

        raw_vectors = self._embedder.embed_texts([chunk.text for chunk in chunk_tuple])
        self._validate_embedding_matrix(raw_vectors, expected_rows=len(chunk_tuple))
        vectors = self._l2_normalise(raw_vectors)

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        return FaissChunkIndex(
            chunks=chunk_tuple,
            index=index,
        )

    def retrieve(
        self,
        *,
        index: ChunkIndex,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not isinstance(index, FaissChunkIndex):
            raise TypeError("FaissVectorStore requires a FaissChunkIndex")
        if not index.chunks:
            return []

        query_matrix = self._embedder.embed_texts([question])
        self._validate_embedding_matrix(query_matrix, expected_rows=1)
        query_vector = self._l2_normalise(query_matrix)

        scores, indices = index.index.search(query_vector, min(top_k, len(index.chunks)))

        retrieved: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=index.chunks[int(idx)],
                    score=float(score),
                )
            )
        return retrieved

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
