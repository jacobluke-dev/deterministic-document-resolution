from __future__ import annotations

from collections.abc import Sequence

from plainera_rag_demo.common import RetrievedChunk
from plainera_rag_demo.contracts.interfaces import AnswerGenerator


class DemoAnswerGenerator(AnswerGenerator):
    def generate_answer(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        if not retrieved_chunks:
            return f"{question} :: no evidence"

        best = retrieved_chunks[0].chunk
        return f"{question} :: {best.document_name} [{best.start_offset}:{best.end_offset}]"
