from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rag_demo.common import BaselineAnswerResult, DemoDocument, IndexedCorpus
from rag_demo.contracts import AnswerGenerator, Chunker
from rag_demo.contracts.interfaces import ChunkIndex, VectorStore


@dataclass(frozen=True, slots=True)
class BaselineCorpusIndex:
    """Represent an indexed baseline corpus ready for question answering.

    Attributes:
        corpus: Chunked corpus payload containing source documents and chunks.
        vector_index: Retrieval index built over the emitted chunks.
    """

    corpus: IndexedCorpus
    vector_index: ChunkIndex


class BaselineRagPipeline:
    """Coordinate the baseline RAG control pipeline.

    This pipeline performs document chunking, vector index construction,
    retrieval, and answer generation without any deterministic grounding or
    glossary binding.
    """

    def __init__(
        self,
        *,
        chunker: Chunker,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
    ) -> None:
        """Initialise the baseline pipeline.

        Args:
            chunker: Chunking strategy used to split input documents.
            vector_store: Retrieval backend used to index and retrieve chunks.
            answer_generator: Answer generator used to produce the final answer
                from retrieved evidence.
        """
        self._chunker = chunker
        self._vector_store = vector_store
        self._answer_generator = answer_generator

    def index_documents(self, documents: Sequence[DemoDocument]) -> BaselineCorpusIndex:
        """Chunk documents and build a retrieval index over the emitted chunks.

        Args:
            documents: Source documents to index.

        Returns:
            A ``BaselineCorpusIndex`` containing the chunked corpus and its
            associated retrieval index.
        """
        document_tuple = tuple(documents)
        chunks = tuple(self._chunker.chunk_documents(document_tuple))
        corpus = IndexedCorpus(
            documents=document_tuple,
            chunks=chunks,
        )
        vector_index = self._vector_store.index_chunks(chunks)

        return BaselineCorpusIndex(
            corpus=corpus,
            vector_index=vector_index,
        )

    def answer_question(
        self,
        *,
        index: BaselineCorpusIndex,
        question: str,
        top_k: int = 5,
    ) -> BaselineAnswerResult:
        """Retrieve evidence for a question and generate a baseline answer.

        Args:
            index: Indexed baseline corpus to query.
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
