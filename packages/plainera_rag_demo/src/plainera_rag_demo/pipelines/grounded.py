from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from plainera_rag_demo.common import BaselineAnswerResult, DemoDocument, IndexedCorpus
from plainera_rag_demo.contracts import AnswerGenerator, Chunker
from plainera_rag_demo.contracts.interfaces import ChunkIndex, VectorStore, GroundingStage


@dataclass(frozen=True, slots=True)
class GroundedCorpusIndex:
    """Represent an indexed grounded corpus ready for question answering.

    Attributes:
        source_documents: Original source documents provided for indexing.
        grounded_corpus: Chunked corpus built from grounded document text.
        vector_index: Retrieval index built over grounded chunks.
    """

    source_documents: tuple[DemoDocument, ...]
    grounded_corpus: IndexedCorpus
    vector_index: ChunkIndex


class GroundedRagPipeline:
    """Coordinate the grounded RAG control pipeline.

    This pipeline performs deterministic grounding before chunking so that
    selected bindings are injected into the retrieval representation prior to
    embedding and retrieval.
    """

    def __init__(
        self,
        *,
        grounding_stage: GroundingStage,
        chunker: Chunker,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
    ) -> None:
        """Initialise the grounded pipeline.

        Args:
            grounding_stage: Deterministic pre-retrieval grounding stage used to
                transform source documents into grounded documents.
            chunker: Chunking strategy used to split grounded documents.
            vector_store: Retrieval backend used to index and retrieve chunks.
            answer_generator: Answer generator used to produce the final answer
                from retrieved evidence.
        """
        self._grounding_stage = grounding_stage
        self._chunker = chunker
        self._vector_store = vector_store
        self._answer_generator = answer_generator

    async def index_documents(self, documents: Sequence[DemoDocument]) -> GroundedCorpusIndex:
        """Ground documents, chunk them, and build a retrieval index.

        Args:
            documents: Source documents to index.

        Returns:
            A ``GroundedCorpusIndex`` containing the original documents, the
            grounded chunked corpus, and its associated retrieval index.
        """
        source_documents = tuple(documents)
        grounded_documents = tuple(
            await self._grounding_stage.ground_documents(source_documents)
        )
        chunks = tuple(self._chunker.chunk_documents(grounded_documents))
        grounded_corpus = IndexedCorpus(
            documents=grounded_documents,
            chunks=chunks,
        )
        vector_index = self._vector_store.index_chunks(chunks)

        return GroundedCorpusIndex(
            source_documents=source_documents,
            grounded_corpus=grounded_corpus,
            vector_index=vector_index,
        )

    def answer_question(
        self,
        *,
        index: GroundedCorpusIndex,
        question: str,
        top_k: int = 5,
    ) -> BaselineAnswerResult:
        """Retrieve evidence for a question and generate a grounded answer.

        Args:
            index: Indexed grounded corpus to query.
            question: User question to answer.
            top_k: Maximum number of chunks to retrieve.

        Returns:
            A ``BaselineAnswerResult`` containing the final answer and the
            retrieved evidence used to produce it.
        """
        retrieved_chunks = tuple(
            self._vector_store.retrieve(
                index=index.vector_index,
                question=question,
                top_k=top_k,
            )
        )
        answer = self._answer_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return BaselineAnswerResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )
