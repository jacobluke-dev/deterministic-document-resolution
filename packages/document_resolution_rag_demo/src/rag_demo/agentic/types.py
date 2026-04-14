from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

GroundedAgentOutcome = Literal["answer", "answer_with_warning", "abstain"]
GroundedAgentAction = Literal["proceed", "retry_once", "abstain"]


@dataclass(frozen=True, slots=True)
class GroundedEvidenceDocument:
    """Structured grounded evidence for one retrieved document chunk."""

    document_id: str
    document_name: str
    chunk_id: str
    chunk_span: tuple[int, int]
    score: float | None
    grounding_payload: dict[str, Any] | None
    source_excerpt: str


@dataclass(frozen=True, slots=True)
class GroundedEvidencePacket:
    """Bounded evidence bundle passed to the reviewer."""

    question: str
    documents: tuple[GroundedEvidenceDocument, ...]


@dataclass(frozen=True, slots=True)
class GroundedEvidenceAssessment:
    """Structured evidence assessment for grounded retrieval.

    Args:
        action: Next bounded control action for the pipeline.
        outcome: Final user-visible outcome category.
        sufficient_evidence: Whether the current retrieved evidence is strong
            enough to support answering.
        ambiguity_detected: Whether unresolved ambiguity remains in the current
            evidence set.
        requested_second_pass: Whether the orchestrator is asking for one
            additional retrieval pass.
        answer_text: The user-facing response text to return when ``action`` is not
         ``"abstain"``. Must be ``None`` when ``action`` is ``"abstain"``.
        abstain_reason: Human-readable abstention reason when abstaining.
        warning_reason: Human-readable warning reason when answering with
            caution.
        reasoning_notes: Deterministic reasoning notes suitable for demo UI.
        selected_audit_bindings: Binding keys selected for audit display.
        selected_audit_spans: Span tuples selected for audit display.
    """

    action: GroundedAgentAction
    outcome: GroundedAgentOutcome
    sufficient_evidence: bool
    ambiguity_detected: bool
    requested_second_pass: bool
    answer_text: str | None
    abstain_reason: str | None
    warning_reason: str | None
    reasoning_notes: tuple[str, ...]
    selected_audit_bindings: tuple[str, ...]
    selected_audit_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class GroundedAgentAnswerResult:
    """Final grounded answer result including orchestration metadata.

    Args:
        question: Original user question.
        outcome: Final bounded orchestration outcome.
        answer: Final answer text, or ``None`` when abstaining.
        retrieved_chunks: Retrieved evidence used for the decision.
        assessment: Structured evidence assessment supporting the outcome.
    """

    question: str
    outcome: GroundedAgentOutcome
    answer: str | None
    retrieved_chunks: tuple[Any, ...]
    assessment: GroundedEvidenceAssessment
