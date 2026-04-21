from __future__ import annotations

import numpy as np
from rag_demo.common import DemoDocument, FixedWindowChunker
from rag_demo.retrieval.faiss import FaissVectorStore


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    float("alpha" in lower),
                    float("beta" in lower),
                    float("payment" in lower),
                    float("schedule" in lower),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

class TestFaissVectorStore:
    def test_faiss_vector_store_retrieves_best_chunk_first(self) -> None:
        chunker = FixedWindowChunker(chunk_size=100)
        chunks = chunker.chunk_documents(
            [
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
        )

        store = FaissVectorStore(embedder=FakeEmbedder())
        index = store.index_chunks(chunks)

        results = store.retrieve(
            index=index,
            question="What does the payment schedule say about alpha?",
            top_k=2,
        )

        assert results[0].chunk.document_name == "msa.txt"
        assert results[1].chunk.document_name == "schedule.txt"
