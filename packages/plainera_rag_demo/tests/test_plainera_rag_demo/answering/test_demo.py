from __future__ import annotations

from typing import Sequence

import pytest
from plainera_rag_demo.common import DemoDocument
from plainera_rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage
from public_api.schemas.resolve import ResolveRequest


class _FakeResolveResponse:
    def model_dump_json(self, *, indent: int | None = None) -> str:
        return '{"acronyms":[{"acronym":"MPS","selected":{"definition":"Metropolitan Police Service"}}]}'


class _FakeResolveService:
    def __init__(self) -> None:
        self.calls: list[ResolveRequest] = []

    async def resolve(self, payload: ResolveRequest) -> _FakeResolveResponse:
        self.calls.append(payload)
        return _FakeResolveResponse()


class _RecordingGroundingStage:
    def __init__(self) -> None:
        self.calls: list[tuple[DemoDocument, ...]] = []

    async def ground_documents(
        self,
        documents: Sequence[DemoDocument],
    ) -> tuple[DemoDocument, ...]:
        document_tuple = tuple(documents)
        self.calls.append(document_tuple)
        return tuple(
            DemoDocument(
                document_id=document.document_id,
                name=document.name,
                text=f"[GROUNDED]\n{document.text}",
            )
            for document in document_tuple
        )


class _RecordingChunker:
    def __init__(self) -> None:
        self.calls: list[tuple[DemoDocument, ...]] = []

    def chunk_documents(self, documents: Sequence[DemoDocument]) -> tuple[object, ...]:
        document_tuple = tuple(documents)
        self.calls.append(document_tuple)
        return ()


class _FakeVectorStore:
    def __init__(self) -> None:
        self.index_calls: list[tuple[object, ...]] = []

    def index_chunks(self, chunks: Sequence[object]) -> object:
        chunk_tuple = tuple(chunks)
        self.index_calls.append(chunk_tuple)
        return object()

    def retrieve(
        self,
        *,
        index: object,
        question: str,
        top_k: int,
    ) -> tuple[object, ...]:
        return ()


class _FakeAnswerGenerator:
    def generate_answer(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[object],
    ) -> str:
        return "answer"


class TestGroundedGroundDocuments:
    @pytest.mark.anyio
    async def test_resolve_backed_grounding_stage_builds_grounded_documents_from_resolve_output(self) -> None:
        resolve_service = _FakeResolveService()
        stage = ResolveBackedGroundingStage(resolve_service=resolve_service)
        document = DemoDocument(
            document_id="doc-1",
            name="sample.txt",
            text="The Metropolitan Police Service (MPS) operates in London.",
        )

        grounded_documents = await stage.ground_documents((document,))

        assert len(resolve_service.calls) == 1
        payload = resolve_service.calls[0]
        assert payload.text == document.text
        assert grounded_documents == (
            DemoDocument(
                document_id="doc-1",
                name="sample.txt",
                text=(
                    "[DETERMINISTIC_GROUNDING]\n"
                    '{"acronyms":[{"acronym":"MPS","selected":{"definition":"Metropolitan Police Service"}}]}\n\n'
                    "[DOCUMENT]\n"
                    "The Metropolitan Police Service (MPS) operates in London."
                ),
            ),
        )


class TestGroundedIndexDocuments:
    @pytest.mark.anyio
    async def test_grounded_pipeline_indexes_grounded_documents_before_chunking(self) -> None:
        grounding_stage = _RecordingGroundingStage()
        chunker = _RecordingChunker()
        vector_store = _FakeVectorStore()
        answer_generator = _FakeAnswerGenerator()
        pipeline = GroundedRagPipeline(
            grounding_stage=grounding_stage,
            chunker=chunker,
            vector_store=vector_store,
            answer_generator=answer_generator,
        )
        document = DemoDocument(
            document_id="doc-1",
            name="sample.txt",
            text="Original source text.",
        )

        index = await pipeline.index_documents((document,))

        assert grounding_stage.calls == [(document,)]
        assert len(chunker.calls) == 1
        grounded_documents = chunker.calls[0]
        assert grounded_documents == (
            DemoDocument(
                document_id="doc-1",
                name="sample.txt",
                text="[GROUNDED]\nOriginal source text.",
            ),
        )
        assert index.source_documents == (document,)
        assert index.grounded_corpus.documents == grounded_documents
