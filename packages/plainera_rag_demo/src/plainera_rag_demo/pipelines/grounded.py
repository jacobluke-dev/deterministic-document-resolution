from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from public_api.core.services.resolve_service import ResolveService
from public_api.schemas.resolve import ResolutionMode, ResolveOptions, ResolveRequest, ResolveResponse, ResolveTarget

from plainera_rag_demo.common import BaselineAnswerResult, DemoDocument, IndexedCorpus
from plainera_rag_demo.contracts import AnswerGenerator, Chunker
from plainera_rag_demo.contracts.interfaces import ChunkIndex, GroundingStage, VectorStore


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
        grounded_documents = tuple(await self._grounding_stage.ground_documents(source_documents))
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


class ResolveBackedGroundingStage(GroundingStage):
    """Ground documents using deterministic resolution before retrieval."""

    def __init__(
        self,
        *,
        resolve_service: ResolveService,
    ) -> None:
        """Initialise the grounding stage.

        Args:
            resolve_service: Service used to produce deterministic grounding
                output for a document.
        """
        self._resolve_service = resolve_service

    async def ground_documents(
        self,
        documents: Sequence[DemoDocument],
    ) -> tuple[DemoDocument, ...]:
        """Return grounded documents ready for downstream chunking."""
        grounded_documents: list[DemoDocument] = []

        for document in documents:
            grounded_documents.append(await self._ground_document(document))

        return tuple(grounded_documents)

    @staticmethod
    def _build_grounded_text(*, text: str, resolved: ResolveResponse) -> str:
        """Build grounded retrieval text from resolve output.

        The first pass prepends deterministic JSON context to the source text so the
        downstream chunking and retrieval stages carry that grounding information.
        """
        deterministic_context = resolved.model_dump_json(indent=2)

        return "[DETERMINISTIC_GROUNDING]\n" f"{deterministic_context}\n\n" "[DOCUMENT]\n" f"{text}"

    async def _ground_document(self, document: DemoDocument) -> DemoDocument:
        """Ground a single document."""
        payload = ResolveRequest(
            text=document.text,
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            targets=[
                ResolveTarget.ACRONYMS,
                ResolveTarget.DEFINED_TERMS,
                ResolveTarget.STRUCTURAL_REFERENCES,
            ],
            options=ResolveOptions(
                locale="en-GB",
                window_chars=120,
                max_definitions_per_acronym=5,
                include_glossary_enrichment=True,
                return_occurrences=True,
                min_confidence=0.0,
            ),
        )
        resolved = await self._resolve_service.resolve(payload)

        grounded_text = self._build_grounded_text(
            text=document.text,
            resolved=resolved,
        )

        return DemoDocument(
            document_id=document.document_id,
            name=document.name,
            text=grounded_text,
        )
