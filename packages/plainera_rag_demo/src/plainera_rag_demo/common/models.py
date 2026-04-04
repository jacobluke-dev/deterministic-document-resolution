from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoDocument:
    """Input document for a demo corpus."""

    document_id: str
    name: str
    text: str


@dataclass(frozen=True, slots=True)
class DemoChunk:
    """Chunk derived from a source document."""

    chunk_id: str
    document_id: str
    document_name: str
    ordinal: int
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    """Chunked corpus ready for retrieval."""

    documents: tuple[DemoDocument, ...]
    chunks: tuple[DemoChunk, ...]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Retrieved chunk plus similarity score."""

    chunk: DemoChunk
    score: float


@dataclass(frozen=True, slots=True)
class BaselineAnswerResult:
    """Baseline RAG answer and the retrieved evidence used to produce it."""

    question: str
    answer: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
