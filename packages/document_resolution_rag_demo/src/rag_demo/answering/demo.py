from __future__ import annotations

from collections.abc import Callable, Sequence

from rag_demo.common import RetrievedChunk
from rag_demo.contracts.interfaces import AnswerGenerator

_BASELINE_SYSTEM_PROMPT = """You are answering questions from retrieved document excerpts in a baseline RAG demo.

Use only the retrieved excerpts provided.
Do not mention missing grounding, deterministic binding, or internal system design.
Answer as directly as possible in 1-3 sentences.

If the retrieved excerpts do not contain enough information, say so briefly.
"""


class DemoAnswerGenerator(AnswerGenerator):
    """Generate a baseline answer from retrieved chunks using plain context inference."""

    def __init__(self, *, model_complete: Callable[[str, str], str]) -> None:
        """Initialise the baseline answer generator.

        Args:
            model_complete: Callable that accepts a system prompt and user prompt
                and returns the model text response.
        """
        self._model_complete = model_complete

    def generate_answer(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        """Return a baseline answer inferred from retrieved chunk text.

        Args:
            question: User question supplied to the pipeline.
            retrieved_chunks: Ranked retrieval results in descending relevance
                order.

        Returns:
            A model-generated answer based only on the retrieved chunk text, or
            a no-evidence marker when nothing was retrieved.
        """
        if not retrieved_chunks:
            return "Insufficient retrieved evidence to answer the question."

        prompt = self._render_user_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
        return self._model_complete(_BASELINE_SYSTEM_PROMPT, prompt).strip()

    @staticmethod
    def _render_user_prompt(
        *,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        """Render retrieved chunks into a compact baseline answering prompt."""
        rendered_chunks: list[str] = []

        for idx, retrieved in enumerate(retrieved_chunks, start=1):
            chunk = retrieved.chunk
            rendered_chunks.append(
                "\n".join(
                    (
                        f"[Chunk {idx}]",
                        f"Document: {chunk.document_name}",
                        f"Span: {chunk.start_offset}:{chunk.end_offset}",
                        "Text:",
                        chunk.text,
                    )
                )
            )

        joined_chunks = "\n\n".join(rendered_chunks)

        return "\n\n".join(
            (
                f"Question: {question}",
                "Retrieved excerpts:",
                joined_chunks,
                "Answer using only the retrieved excerpts.",
            )
        )
