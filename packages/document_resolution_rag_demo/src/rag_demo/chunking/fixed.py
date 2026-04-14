from __future__ import annotations

from collections.abc import Sequence

from rag_demo.common.models import DemoChunk, DemoDocument
from rag_demo.contracts.interfaces import Chunker


class FixedWindowChunker(Chunker):
    """Split documents into deterministic fixed-size character chunks.

    Chunks are emitted in input document order and preserve absolute character
    offsets into the source text. Overlap is applied by advancing the chunk
    window by ``chunk_size - chunk_overlap`` characters each iteration.
    """

    def __init__(self, *, chunk_size: int, chunk_overlap: int = 0) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between adjacent
                chunks.

        Raises:
            ValueError: If ``chunk_size`` is not positive, if
                ``chunk_overlap`` is negative, or if ``chunk_overlap`` is not
                smaller than ``chunk_size``.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def chunk_documents(self, documents: Sequence[DemoDocument]) -> list[DemoChunk]:
        """Chunk all supplied documents in deterministic input order.

        Args:
            documents: Documents to chunk.

        Returns:
            A flat list of emitted chunks across all documents.
        """
        chunks: list[DemoChunk] = []

        for document in documents:
            chunks.extend(self._chunk_document(document))

        return chunks

    def _chunk_document(self, document: DemoDocument) -> list[DemoChunk]:
        """Chunk a single document into overlapping character windows.

        Args:
            document: Source document to chunk.

        Returns:
            Chunks for the document. Whitespace-only chunk windows are skipped.
        """
        if not document.text:
            return []

        text = document.text
        chunks: list[DemoChunk] = []
        ordinal = 0
        start = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    DemoChunk(
                        chunk_id=f"{document.document_id}:{ordinal}",
                        document_id=document.document_id,
                        document_name=document.name,
                        ordinal=ordinal,
                        start_offset=start,
                        end_offset=end,
                        text=chunk_text,
                    )
                )

            if end >= len(text):
                break

            start += self._step
            ordinal += 1

        return chunks
