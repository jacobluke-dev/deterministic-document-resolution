from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from plainera_rag_demo.common.models import DemoChunk, DemoDocument, RetrievedChunk
from plainera_rag_demo.common.types import FloatMatrix


class Chunker(ABC):
    """Chunk documents into retrieval units."""

    @abstractmethod
    def chunk_documents(self, documents: Sequence[DemoDocument]) -> list[DemoChunk]:
        """Return chunks in deterministic input order."""


class ChunkIndex:
    """Marker base class for retrieval index payloads."""


class VectorStore(ABC):
    """Index and retrieve chunks for question answering."""

    @abstractmethod
    def index_chunks(self, chunks: Sequence[DemoChunk]) -> ChunkIndex:
        """Build a retrieval index for the supplied chunks."""

    @abstractmethod
    def retrieve(
        self,
        *,
        index: ChunkIndex,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return top-k retrieved chunks for the question."""


class Embedder(ABC):
    """Generate dense vectors for text inputs."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> FloatMatrix:
        """Return one embedding row per input string."""


class AnswerGenerator(ABC):
    """Generate a final answer from retrieved evidence."""

    @abstractmethod
    def generate_answer(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        """Return a single-turn answer."""


class GroundingStage(ABC):
    """Define deterministic document grounding prior to chunking."""

    @abstractmethod
    async def ground_documents(
        self,
        documents: Sequence[DemoDocument],
    ) -> tuple[DemoDocument, ...]:
        """Return grounded documents ready for chunking."""
