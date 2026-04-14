from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import faiss  # type: ignore[import-untyped]
import numpy as np

from rag_demo.common import DemoChunk, RetrievedChunk
from rag_demo.common.types import FloatMatrix
from rag_demo.contracts.interfaces import ChunkIndex, Embedder, VectorStore


@dataclass(frozen=True, slots=True)
class FaissChunkIndex(ChunkIndex):
    """Represent a FAISS-backed retrieval index for document chunks.

    Attributes:
        chunks: Chunks indexed in FAISS row order.
        index: FAISS inner-product index built over the chunk embeddings.
    """

    chunks: tuple[DemoChunk, ...]
    index: faiss.IndexFlatIP


class FaissVectorStore(VectorStore):
    """Retrieve chunks using a FAISS inner-product index over normalised vectors.

    Chunk and query embeddings are L2-normalised before indexing and search so
    inner-product similarity behaves as cosine similarity.
    """

    def __init__(self, *, embedder: Embedder) -> None:
        """Initialise the vector store.

        Args:
            embedder: Embedder used for both chunk indexing and query embedding.
        """
        self._embedder = embedder

    def index_chunks(self, chunks: Sequence[DemoChunk]) -> FaissChunkIndex:
        """Build a FAISS index for the supplied chunks.

        Args:
            chunks: Chunks to embed and add to the retrieval index.

        Returns:
            A ``FaissChunkIndex`` containing the original chunks and the FAISS
            index built over their embeddings.

        Raises:
            ValueError: If the embedder returns an invalid embedding matrix.
        """
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
        """Retrieve the top-k most similar chunks for a question.

        Args:
            index: Retrieval index to query. Must be a ``FaissChunkIndex``.
            question: User question to embed and search against the index.
            top_k: Maximum number of results to return.

        Returns:
            Retrieved chunks in descending similarity order.

        Raises:
            ValueError: If ``top_k`` is not positive or if the embedder returns
                an invalid embedding matrix.
            TypeError: If ``index`` is not a ``FaissChunkIndex``.
        """
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
