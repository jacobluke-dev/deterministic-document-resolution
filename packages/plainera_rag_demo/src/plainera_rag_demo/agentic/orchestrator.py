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
    """Bounded reviewer that adjudicates over grounded evidence."""

    @abstractmethod
    def review(
        self,
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Review a structured grounded evidence packet and return a decision."""


@dataclass(frozen=True, slots=True)
class StructuredGroundingReviewer(GroundedEvidenceReviewer):
    """Interim reviewer operating over structured grounding payloads.

    This reviewer is intentionally simple. Its purpose is to prove the correct
    orchestration seam: the reviewer receives a structured evidence packet built
    from deterministic grounding output rather than raw chunk-text heuristics.

    It can later be replaced by a model-backed reviewer without changing the
    grounded pipeline contract.
    """

    def review(
        self,
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Review structured grounded evidence and return a bounded decision."""
        docs_with_grounding = tuple(
            document
            for document in evidence.documents
            if document.grounding_payload is not None
        )

        audit_bindings = tuple(dict.fromkeys(doc.document_id for doc in evidence.documents))
        audit_spans = tuple(doc.chunk_span for doc in evidence.documents)

        if docs_with_grounding:
            return GroundedEvidenceAssessment(
                action="proceed",
                outcome="answer_with_warning",
                sufficient_evidence=True,
                ambiguity_detected=False,
                requested_second_pass=False,
                abstain_reason=None,
                warning_reason="Answer supported by structured grounded evidence.",
                reasoning_notes=(
                    f"Retrieved {len(evidence.documents)} grounded chunks.",
                    f"Structured grounding payloads available for {len(docs_with_grounding)} retrieved chunks.",
                    "Reviewer operated over parsed deterministic grounding rather than raw chunk text.",
                ),
                selected_audit_bindings=audit_bindings,
                selected_audit_spans=audit_spans,
            )

        if has_second_pass_available:
            return GroundedEvidenceAssessment(
                action="retry_once",
                outcome="answer_with_warning",
                sufficient_evidence=False,
                ambiguity_detected=False,
                requested_second_pass=True,
                abstain_reason=None,
                warning_reason="Initial retrieval did not include a usable grounding payload.",
                reasoning_notes=(
                    f"Retrieved {len(evidence.documents)} grounded chunks.",
                    "No parseable deterministic grounding payload was available in the current retrieval set.",
                    "One additional retrieval pass was requested.",
                ),
                selected_audit_bindings=audit_bindings,
                selected_audit_spans=audit_spans,
            )

        return GroundedEvidenceAssessment(
            action="abstain",
            outcome="abstain",
            sufficient_evidence=False,
            ambiguity_detected=False,
            requested_second_pass=False,
            abstain_reason="No usable deterministic grounding payload was available after retrieval.",
            warning_reason=None,
            reasoning_notes=(
                f"Retrieved {len(evidence.documents)} grounded chunks.",
                "The reviewer could not find a parseable deterministic grounding payload to support a safe answer.",
            ),
            selected_audit_bindings=audit_bindings,
            selected_audit_spans=audit_spans,
        )


@dataclass(frozen=True, slots=True)
class SingleAgentEvidenceOrchestrator:
    """Build a grounded evidence packet and delegate review to a bounded reviewer."""

    reviewer: GroundedEvidenceReviewer

    def assess(
        self,
        *,
        question: str,
        retrieved_chunks: tuple[Any, ...],
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Build structured grounded evidence and delegate to the reviewer."""
        evidence = self._build_evidence_packet(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
        return self.reviewer.review(
            evidence=evidence,
            has_second_pass_available=has_second_pass_available,
        )

    @classmethod
    def _build_evidence_packet(
        cls,
        *,
        question: str,
        retrieved_chunks: tuple[Any, ...],
    ) -> GroundedEvidencePacket:
        """Convert retrieved chunks into a structured grounded evidence packet."""
        documents: list[GroundedEvidenceDocument] = []

        for retrieved in retrieved_chunks:
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

            documents.append(
                GroundedEvidenceDocument(
                    document_id=document_id,
                    document_name=document_name,
                    chunk_id=chunk_id,
                    chunk_span=(start_offset, end_offset),
                    score=parsed_score,
                    grounding_payload=grounding_payload,
                    source_excerpt=source_excerpt,
                )
            )

        return GroundedEvidencePacket(
            question=question,
            documents=tuple(documents),
        )

    @staticmethod
    def _inner_chunk(retrieved: Any) -> Any:
        """Return the nested chunk object when retrieval wraps the stored chunk."""
        nested = getattr(retrieved, "chunk", None)
        return nested if nested is not None else retrieved

    @classmethod
    def _chunk_text(cls, retrieved: Any) -> str:
        """Extract text from the retrieved chunk shape."""
        chunk = cls._inner_chunk(retrieved)
        text = getattr(chunk, "text", None)
        return text if isinstance(text, str) else ""

    @staticmethod
    def _split_grounding_and_document(text: str) -> tuple[dict[str, Any] | None, str]:
        """Split grounded text into parsed grounding payload and source excerpt."""
        grounding_marker = "[DETERMINISTIC_GROUNDING]\n"
        document_marker = "\n\n[DOCUMENT]\n"

        if not text.startswith(grounding_marker):
            return None, text

        marker_index = text.find(document_marker)
        if marker_index == -1:
            return None, text

        grounding_json = text[len(grounding_marker):marker_index].strip()
        source_excerpt = text[marker_index + len(document_marker):]

        try:
            payload = json.loads(grounding_json)
        except json.JSONDecodeError:
            return None, source_excerpt

        return payload if isinstance(payload, dict) else None, source_excerpt
