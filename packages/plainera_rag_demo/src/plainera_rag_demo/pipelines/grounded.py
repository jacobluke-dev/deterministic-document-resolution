from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from public_api.core.services.resolve_service import ResolveService
from public_api.schemas.resolve import ResolveOptions, ResolveRequest, ResolveTarget, ResolveResponse

from plainera_rag_demo.agentic.orchestrator import SingleAgentEvidenceOrchestrator
from plainera_rag_demo.agentic.types import GroundedAgentAnswerResult
from plainera_rag_demo.common import DemoDocument, IndexedCorpus
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
    embedding and retrieval. After retrieval, it applies a bounded evidence
    orchestration step before final answer generation.
    """

    def __init__(
        self,
        *,
        grounding_stage: GroundingStage,
        chunker: Chunker,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        evidence_orchestrator: SingleAgentEvidenceOrchestrator,
    ) -> None:
        """Initialise the grounded pipeline.

        Args:
            grounding_stage: Deterministic pre-retrieval grounding stage used to
                transform source documents into grounded documents.
            chunker: Chunking strategy used to split grounded documents.
            vector_store: Retrieval backend used to index and retrieve chunks.
            answer_generator: Answer generator used to produce the final answer
                from retrieved evidence.
            evidence_orchestrator: Bounded post-retrieval controller used to
                decide whether to answer, warn, retry once, or abstain.
        """
        self._grounding_stage = grounding_stage
        self._chunker = chunker
        self._vector_store = vector_store
        self._answer_generator = answer_generator
        self._evidence_orchestrator = evidence_orchestrator

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
    ) -> GroundedAgentAnswerResult:
        """Retrieve grounded evidence, assess it, and answer or abstain.

        Args:
            index: Indexed grounded corpus to query.
            question: User question to answer.
            top_k: Maximum number of chunks to retrieve on the first pass.

        Returns:
            A ``GroundedAgentAnswerResult`` containing the final bounded
            outcome, retrieved evidence, and orchestration metadata.
        """
        retrieved_chunks = tuple(
            self._vector_store.retrieve(
                index=index.vector_index,
                question=question,
                top_k=top_k,
            )
        )

        assessment = self._evidence_orchestrator.assess(
            question=question,
            retrieved_chunks=retrieved_chunks,
            has_second_pass_available=True,
        )

        if assessment.action == "retry_once":
            retrieved_chunks = tuple(
                self._vector_store.retrieve(
                    index=index.vector_index,
                    question=question,
                    top_k=max(top_k * 2, top_k + 1),
                )
            )
            assessment = self._evidence_orchestrator.assess(
                question=question,
                retrieved_chunks=retrieved_chunks,
                has_second_pass_available=False,
            )

        if assessment.action == "abstain":
            return GroundedAgentAnswerResult(
                question=question,
                outcome=assessment.outcome,
                answer=None,
                retrieved_chunks=retrieved_chunks,
                assessment=assessment,
            )

        answer = self._answer_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return GroundedAgentAnswerResult(
            question=question,
            outcome=assessment.outcome,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            assessment=assessment,
        )


class ResolveBackedGroundingStage(GroundingStage):
    """Ground documents by prepending deterministic resolve output.

    This stage adapts ``ResolveService`` into the grounded RAG pipeline. For each
    source document, it calls the resolve flow over the raw text, serialises the
    resulting deterministic JSON, and prepends that context to the original
    document text before chunking and retrieval.

    The current implementation uses the full ``ResolveResponse`` JSON as the
    grounded context so the retrieval layer operates over an explicitly enriched
    representation of the source document.
    """

    def __init__(
        self,
        *,
        resolve_service: ResolveService,
    ) -> None:
        """Initialise the grounding stage.

        Args:
            resolve_service: Resolve service used to produce deterministic
                acronym, defined-term, and structural-reference output for each
                document.
        """
        self._resolve_service = resolve_service

    async def ground_documents(
        self,
        documents: Sequence[DemoDocument],
    ) -> tuple[DemoDocument, ...]:
        """Ground each document and return the transformed document set.

        Args:
            documents: Source documents to enrich with deterministic resolve
                output before chunking.

        Returns:
            A tuple of grounded ``DemoDocument`` instances whose ``text`` fields
            contain deterministic context prepended to the original source text.
        """
        grounded_documents: list[DemoDocument] = []

        for document in documents:
            grounded_documents.append(await self._ground_document(document))

        return tuple(grounded_documents)

    @staticmethod
    def _build_grounded_text(*, text: str, resolved: ResolveResponse) -> str:
        """Build the grounded retrieval text for a single document.

        The current grounding representation prepends the full serialized
        ``ResolveResponse`` JSON ahead of the original source text. This makes
        deterministic resolution output visible to downstream chunking,
        embedding, and retrieval without altering the original document body.

        Args:
            text: Original raw document text.
            resolved: Deterministic resolve output produced for the document.

        Returns:
            A single grounded text block containing deterministic context
            followed by the original document text.
        """
        deterministic_context = resolved.model_dump_json(indent=2)

        return "[DETERMINISTIC_GROUNDING]\n" f"{deterministic_context}\n\n" "[DOCUMENT]\n" f"{text}"

    async def _ground_document(self, document: DemoDocument) -> DemoDocument:
        """Ground a single document using the resolve service.

        This method builds a ``ResolveRequest`` from the source document text,
        executes deterministic resolution with the configured targets, converts
        the resulting ``ResolveResponse`` into grounded text, and returns a new
        document carrying that enriched representation.

        Args:
            document: Source document to ground.

        Returns:
            A new ``DemoDocument`` with the same identity metadata as the source
            document and grounded text suitable for downstream chunking.
        """
        payload = ResolveRequest(
            text=document.text,
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
