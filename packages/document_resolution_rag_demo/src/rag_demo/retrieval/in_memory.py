from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rag_demo.common.models import DemoChunk, RetrievedChunk
from rag_demo.common.types import FloatMatrix
from rag_demo.common.interfaces import ChunkIndex, Embedder, VectorStore


@dataclass(frozen=True, slots=True)
class InMemoryChunkIndex(ChunkIndex):
    """Represent an in-memory retrieval index for document chunks.

    Attributes:
        chunks: Chunks indexed in vector row order.
        vectors: L2-normalised embedding matrix aligned to ``chunks``.
    """

    chunks: tuple[DemoChunk, ...]
    vectors: FloatMatrix


class InMemoryVectorStore(VectorStore):
    """Retrieve chunks using an in-memory cosine-similarity search.

    Chunk and query embeddings are L2-normalised so dot-product scoring
    behaves as cosine similarity.
    """

    def __init__(self, *, embedder: Embedder) -> None:
        """Initialise the vector store.

        Args:
            embedder: Embedder used for both chunk indexing and query embedding.
        """
        self._embedder = embedder

    def index_chunks(self, chunks: Sequence[DemoChunk]) -> InMemoryChunkIndex:
        """Build an in-memory index for the supplied chunks.

        Args:
            chunks: Chunks to embed and index.

        Returns:
            An ``InMemoryChunkIndex`` containing the original chunks and their
            aligned embedding matrix.

        Raises:
            ValueError: If the embedder returns an invalid embedding matrix.
        """
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
        index: ChunkIndex,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most similar chunks for a question.

        Args:
            index: Retrieval index to query. Must be an
                ``InMemoryChunkIndex``.
            question: User question to embed and search against the index.
            top_k: Maximum number of results to return.

        Returns:
            Retrieved chunks in descending similarity order.

        Raises:
            ValueError: If ``top_k`` is not positive or if the embedder returns
                an invalid embedding matrix.
            TypeError: If ``index`` is not an ``InMemoryChunkIndex``.
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not isinstance(index, InMemoryChunkIndex):
            raise TypeError("InMemoryVectorStore requires an InMemoryChunkIndex")
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
        """Validate the shape of an embedding matrix.

        Args:
            matrix: Embedding matrix returned by the embedder.
            expected_rows: Expected number of rows in the matrix.

        Raises:
            ValueError: If the matrix is not 2D or does not contain the
                expected number of rows.
        """
        if matrix.ndim != 2:
            raise ValueError("embedder must return a 2D matrix")
        if matrix.shape[0] != expected_rows:
            raise ValueError("embedder returned unexpected number of rows")

    @staticmethod
    def _l2_normalise(matrix: FloatMatrix) -> FloatMatrix:
        """Return an L2-normalised copy of the supplied embedding matrix.

        Zero-norm rows are treated as having norm 1 to avoid division by zero.

        Args:
            matrix: Embedding matrix to normalise.

        Returns:
            A float32 matrix with each row L2-normalised.
        """
        if matrix.size == 0:
            return matrix.astype(np.float32, copy=False)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / safe_norms).astype(np.float32, copy=False)
