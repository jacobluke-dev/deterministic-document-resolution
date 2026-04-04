from __future__ import annotations

from collections.abc import Sequence

from plainera_rag_demo.common import RetrievedChunk
from plainera_rag_demo.contracts.interfaces import AnswerGenerator


class DemoAnswerGenerator(AnswerGenerator):
    """Generate a simple demonstrator answer from the top retrieved chunk.

    This implementation is intentionally lightweight and deterministic. It does
    not call an LLM. Instead, it echoes the highest-ranked retrieved chunk so
    the baseline pipeline can be exercised end to end during early development.
    """

    def generate_answer(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        """Return a simple answer derived from the top retrieved chunk.

        Args:
            question: User question supplied to the pipeline.
            retrieved_chunks: Ranked retrieval results in descending relevance
                order.

        Returns:
            A deterministic string containing the question and either:
                - a no-evidence marker when no chunks were retrieved, or
                - the top chunk's document name and span.
        """
        if not retrieved_chunks:
            return f"{question} :: no evidence"

        best = retrieved_chunks[0].chunk
        return f"{question} :: {best.document_name} [{best.start_offset}:{best.end_offset}]"
