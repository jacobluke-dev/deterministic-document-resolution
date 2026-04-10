from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from plainera_rag_demo.agentic.types import (
    GroundedEvidenceAssessment,
    GroundedEvidenceDocument,
    GroundedEvidencePacket,
)


class GroundedEvidenceReviewer(ABC):
    """Review structured grounded evidence and return a bounded decision.

    Implementations operate over deterministic grounding that has already been
    produced upstream. Reviewers may decide whether the current evidence is
    sufficient, whether a retry is warranted, or whether the pipeline should
    abstain, but they must not perform independent meaning resolution.
    """

    @abstractmethod
    def review(
        self,
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Review a grounded evidence packet and return an assessment.

        Args:
            evidence: Structured grounded evidence assembled from the current
                retrieval result set.
            has_second_pass_available: Whether the pipeline still permits one
                additional retrieval pass before a final decision is required.

        Returns:
            A bounded assessment describing whether to proceed, retry once, or
            abstain, together with user-facing reasoning metadata.
        """


@dataclass(frozen=True, slots=True)
class SingleAgentEvidenceOrchestrator:
    """Assemble grounded evidence and delegate bounded review.

    The orchestrator is responsible for normalizing retrieved chunks into a
    structured evidence packet that combines retrieval metadata with parsed
    deterministic grounding content. It does not decide meanings itself; it
    delegates the final bounded assessment to the configured reviewer.
    """

    reviewer: GroundedEvidenceReviewer

    def assess(
        self,
        *,
        question: str,
        retrieved_chunks: tuple[Any, ...],
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Build a structured evidence packet and delegate review.

        Args:
            question: User question being evaluated against the retrieved
                grounded evidence.
            retrieved_chunks: Retrieval results for the current pass. Each item
                may either be a stored chunk directly or a wrapper that exposes
                the stored chunk via a ``chunk`` attribute.
            has_second_pass_available: Whether the pipeline still permits one
                additional retrieval pass before a final decision must be made.

        Returns:
            The bounded assessment returned by the configured reviewer.
        """
        evidence = self._build_evidence_packet(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
        return self.reviewer.review(
            evidence=evidence,
            has_second_pass_available=has_second_pass_available,
        )

    @staticmethod
    def _looks_like_grounding_fragment(text: str) -> bool:
        """Return whether a chunk looks like a sliced fragment of grounding JSON."""
        markers = (
            '"acronyms"',
            '"defined_terms"',
            '"structural_references"',
            '"orchestration"',
            '"errors"',
            '"model_version"',
            '"resolution_mode"',
        )
        hits = sum(marker in text for marker in markers)
        return hits >= 2 and "[DOCUMENT]" not in text and "[DETERMINISTIC_GROUNDING]" not in text

    @classmethod
    def _best_excerpt_candidate(
        cls,
        items: list[GroundedEvidenceDocument],
    ) -> GroundedEvidenceDocument | None:
        """Return the best non-JSON-like source excerpt for reviewer use."""
        excerpt_items = [
            item
            for item in items
            if item.source_excerpt
            and item.grounding_payload is None
            and not cls._looks_like_grounding_fragment(item.source_excerpt)
        ]
        if not excerpt_items:
            return None

        def sort_key(item: GroundedEvidenceDocument) -> tuple[float, int]:
            score = item.score if item.score is not None else float("-inf")
            span_len = item.chunk_span[1] - item.chunk_span[0]
            return score, span_len

        return max(excerpt_items, key=sort_key)

    @staticmethod
    def _best_grounding_candidate(
        items: list[GroundedEvidenceDocument],
    ) -> GroundedEvidenceDocument | None:
        """Return the highest-value chunk carrying parseable deterministic grounding."""
        grounding_items = [item for item in items if item.grounding_payload is not None]
        if not grounding_items:
            return None

        def sort_key(item: GroundedEvidenceDocument) -> tuple[int, float, int]:
            starts_at_zero = 1 if item.chunk_span[0] == 0 else 0
            score = item.score if item.score is not None else float("-inf")
            span_len = item.chunk_span[1] - item.chunk_span[0]
            return starts_at_zero, score, span_len

        return max(grounding_items, key=sort_key)

    @classmethod
    def _candidate_document(cls, retrieved: Any) -> GroundedEvidenceDocument:
        """Build a candidate evidence document from one retrieved chunk."""
        chunk = cls._inner_chunk(retrieved)
        document_id = str(getattr(chunk, "document_id", ""))
        document_name = str(getattr(chunk, "document_name", ""))
        chunk_id = str(getattr(chunk, "chunk_id", ""))
        start_offset = int(getattr(chunk, "start_offset", 0))
        end_offset = int(getattr(chunk, "end_offset", 0))
        score = getattr(retrieved, "score", None)
        parsed_score = float(score) if isinstance(score, (int, float)) else None
        text = cls._chunk_text(retrieved)

        grounding_payload, source_excerpt = cls._split_grounding_and_document(text)

        return GroundedEvidenceDocument(
            document_id=document_id,
            document_name=document_name,
            chunk_id=chunk_id,
            chunk_span=(start_offset, end_offset),
            score=parsed_score,
            grounding_payload=grounding_payload,
            source_excerpt=source_excerpt,
        )

    @classmethod
    def _select_compact_documents(
        cls,
        candidates: list[GroundedEvidenceDocument],
    ) -> list[GroundedEvidenceDocument]:
        """Reduce overlapping evidence to a compact per-document set."""
        grouped: dict[str, list[GroundedEvidenceDocument]] = {}

        for candidate in candidates:
            grouped.setdefault(candidate.document_id, []).append(candidate)

        selected: list[GroundedEvidenceDocument] = []

        for document_id in sorted(grouped):
            items = grouped[document_id]

            grounding_doc = cls._best_grounding_candidate(items)
            excerpt_doc = cls._best_excerpt_candidate(items)

            if grounding_doc is not None:
                selected.append(grounding_doc)

            if excerpt_doc is not None and (grounding_doc is None or excerpt_doc.chunk_id != grounding_doc.chunk_id):
                selected.append(excerpt_doc)

        return selected

    @classmethod
    def _build_evidence_packet(
        cls,
        *,
        question: str,
        retrieved_chunks: tuple[Any, ...],
    ) -> GroundedEvidencePacket:
        """Convert retrieved chunks into a compact structured grounded evidence packet.

        For each source document, keep at most:
        - one best grounding-bearing chunk, and
        - one best source-excerpt chunk.

        This reduces prompt duplication while preserving both deterministic binding
        context and question-relevant source text.
        """
        candidates = [cls._candidate_document(retrieved) for retrieved in retrieved_chunks]
        documents = cls._select_compact_documents(candidates)

        return GroundedEvidencePacket(
            question=question,
            documents=tuple(documents),
        )

    @staticmethod
    def _inner_chunk(retrieved: Any) -> Any:
        """Return the stored chunk from a retrieval result shape.

        Args:
            retrieved: Retrieval result object, which may either be a stored
                chunk directly or a wrapper exposing the stored chunk via a
                ``chunk`` attribute.

        Returns:
            The underlying stored chunk object used by the rest of the packet
            builder.
        """
        nested = getattr(retrieved, "chunk", None)
        return nested if nested is not None else retrieved

    @classmethod
    def _chunk_text(cls, retrieved: Any) -> str:
        """Extract text from a retrieval result.

        Args:
            retrieved: Retrieval result object or stored chunk.

        Returns:
            The chunk text when available, otherwise an empty string.
        """
        chunk = cls._inner_chunk(retrieved)
        text = getattr(chunk, "text", None)
        return text if isinstance(text, str) else ""

    @staticmethod
    def _split_grounding_and_document(text: str) -> tuple[dict[str, Any] | None, str]:
        """Split grounded text into deterministic payload and source excerpt.

        The grounded representation is expected to follow this shape:

            [DETERMINISTIC_GROUNDING]
            <json>

            [DOCUMENT]
            <source text>

        If the expected markers are not present, the function treats the input
        as plain source text and returns ``None`` for the grounding payload.

        Args:
            text: Full grounded text block or plain source text.

        Returns:
            A tuple containing:
                - the parsed deterministic grounding payload when available, and
                - the source document excerpt.
        """
        grounding_marker = "[DETERMINISTIC_GROUNDING]\n"
        document_marker = "\n\n[DOCUMENT]\n"

        if not text.startswith(grounding_marker):
            return None, text

        marker_index = text.find(document_marker)
        if marker_index == -1:
            return None, text

        grounding_json = text[len(grounding_marker) : marker_index].strip()
        source_excerpt = text[marker_index + len(document_marker) :]

        try:
            payload = json.loads(grounding_json)
        except json.JSONDecodeError:
            return None, source_excerpt

        return payload if isinstance(payload, dict) else None, source_excerpt
