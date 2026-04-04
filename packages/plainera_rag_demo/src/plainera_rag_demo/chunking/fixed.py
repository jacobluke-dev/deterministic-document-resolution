from __future__ import annotations

from collections.abc import Sequence

from plainera_rag_demo.common.models import DemoChunk, DemoDocument
from plainera_rag_demo.contracts.interfaces import Chunker


class FixedWindowChunker(Chunker):
    """Deterministic fixed-size character chunker with overlap."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int = 0) -> None:
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
        chunks: list[DemoChunk] = []

        for document in documents:
            chunks.extend(self._chunk_document(document))

        return chunks

    def _chunk_document(self, document: DemoDocument) -> list[DemoChunk]:
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
