from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoDocument:
    """Represent an input document supplied to the demo corpus.

    Attributes:
        document_id: Stable identifier for the document within the demo corpus.
        name: Human-readable document name.
        text: Full plain-text document content.
    """

    document_id: str
    name: str
    text: str


@dataclass(frozen=True, slots=True)
class DemoChunk:
    """Represent a chunk derived from a source document.

    Attributes:
        chunk_id: Stable identifier for the chunk.
        document_id: Identifier of the source document.
        document_name: Human-readable source document name.
        ordinal: Zero-based chunk position within the source document.
        start_offset: Inclusive character start offset in the source text.
        end_offset: Exclusive character end offset in the source text.
        text: Chunk text content.
    """

    chunk_id: str
    document_id: str
    document_name: str
    ordinal: int
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    """Represent a chunked corpus ready for retrieval.

    Attributes:
        documents: Source documents included in the corpus.
        chunks: Emitted chunks derived from the source documents.
    """

    documents: tuple[DemoDocument, ...]
    chunks: tuple[DemoChunk, ...]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Represent a retrieved chunk and its retrieval score.

    Attributes:
        chunk: Retrieved chunk payload.
        score: Similarity score assigned during retrieval.
    """

    chunk: DemoChunk
    score: float


@dataclass(frozen=True, slots=True)
class BaselineAnswerResult:
    """Represent the baseline RAG answer and supporting retrieval evidence.

    Attributes:
        question: User question supplied to the baseline pipeline.
        answer: Final answer returned by the answer generator.
        retrieved_chunks: Retrieved evidence used to produce the answer.
    """

    question: str
    answer: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
