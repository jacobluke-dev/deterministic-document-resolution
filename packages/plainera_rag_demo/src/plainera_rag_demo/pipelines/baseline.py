from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from plainera_rag_demo.common import IndexedCorpus, DemoDocument, BaselineAnswerResult
from plainera_rag_demo.contracts import Chunker, AnswerGenerator
from plainera_rag_demo.contracts.interfaces import ChunkIndex, VectorStore



@dataclass(frozen=True, slots=True)
class BaselineCorpusIndex:
    """Indexed baseline corpus ready for question answering."""

    corpus: IndexedCorpus
    vector_index: ChunkIndex


class BaselineRagPipeline:
    """Baseline RAG control pipeline with no grounding logic."""

    def __init__(
        self,
        *,
        chunker: Chunker,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._chunker = chunker
        self._vector_store = vector_store
        self._answer_generator = answer_generator

    def index_documents(self, documents: Sequence[DemoDocument]) -> BaselineCorpusIndex:
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
