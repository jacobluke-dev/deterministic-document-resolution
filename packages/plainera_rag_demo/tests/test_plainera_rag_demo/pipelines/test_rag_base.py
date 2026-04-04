from __future__ import annotations

import numpy as np

from plainera_rag_demo.chunking import FixedWindowChunker
from plainera_rag_demo.common import RetrievedChunk, DemoDocument
from plainera_rag_demo.contracts import AnswerGenerator, Embedder
from plainera_rag_demo.pipelines import BaselineRagPipeline
from plainera_rag_demo.retrieval import InMemoryVectorStore


class FakeEmbedder(Embedder):
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []

        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    float("alpha" in lower),
                    float("beta" in lower),
                    float("schedule" in lower),
                    float("payment" in lower),
                ]
            )

        return np.asarray(rows, dtype=np.float32)


class FakeAnswerGenerator(AnswerGenerator):
    def generate_answer(
        self,
        *,
        question: str,
        retrieved_chunks: tuple[RetrievedChunk, ...] | list[RetrievedChunk],
    ) -> str:
        if not retrieved_chunks:
            return f"{question} :: no evidence"

        best = retrieved_chunks[0].chunk
        return f"{question} :: {best.document_name} [{best.start_offset}:{best.end_offset}]"


class TestFixedWindowChunker:
    def test_chunks_document_with_overlap(self) -> None:
        chunker = FixedWindowChunker(chunk_size=8, chunk_overlap=3)
        document = DemoDocument(
            document_id="doc-1",
            name="terms.txt",
            text="abcdefghijklmno",
        )

        chunks = chunker.chunk_documents([document])

        assert [chunk.text for chunk in chunks] == [
            "abcdefgh",
            "fghijklm",
            "klmno",
        ]
        assert [chunk.start_offset for chunk in chunks] == [0, 5, 10]
        assert [chunk.end_offset for chunk in chunks] == [8, 13, 15]


class TestBaselineRagPipeline:
    def test_indexes_documents_and_returns_retrieved_chunks_in_score_order(self) -> None:
        pipeline = BaselineRagPipeline(
            chunker=FixedWindowChunker(chunk_size=80),
            vector_store=InMemoryVectorStore(embedder=FakeEmbedder()),
            answer_generator=FakeAnswerGenerator(),
        )
        documents = [
            DemoDocument(
                document_id="doc-1",
                name="msa.txt",
                text="Alpha payment obligations apply to the Supplier.",
            ),
            DemoDocument(
                document_id="doc-2",
                name="schedule.txt",
                text="Schedule beta sets the implementation milestones.",
            ),
        ]

        index = pipeline.index_documents(documents)
        result = pipeline.answer_question(
            index=index,
            question="What does the payment schedule say about alpha?",
            top_k=2,
        )

        assert len(index.corpus.documents) == 2
        assert len(index.corpus.chunks) == 2

        assert result.retrieved_chunks[0].chunk.document_name == "msa.txt"
        assert result.retrieved_chunks[1].chunk.document_name == "schedule.txt"
        assert result.answer == "What does the payment schedule say about alpha? :: msa.txt [0:48]"

    def test_returns_empty_retrieval_when_index_has_no_chunks(self) -> None:
        pipeline = BaselineRagPipeline(
            chunker=FixedWindowChunker(chunk_size=50),
            vector_store=InMemoryVectorStore(embedder=FakeEmbedder()),
            answer_generator=FakeAnswerGenerator(),
        )

        index = pipeline.index_documents(
            [
                DemoDocument(
                    document_id="doc-1",
                    name="empty.txt",
                    text="   ",
                )
            ]
        )
        result = pipeline.answer_question(
            index=index,
            question="Anything there?",
        )

        assert index.corpus.chunks == ()
        assert result.retrieved_chunks == ()
        assert result.answer == "Anything there? :: no evidence"
