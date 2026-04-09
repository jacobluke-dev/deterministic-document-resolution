from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from plainera_rag_demo.agentic.types import (
    GroundedEvidenceAssessment,
    GroundedEvidenceDocument,
    GroundedEvidencePacket,
)


_REVIEWER_SYSTEM_PROMPT = """You are a bounded reviewer for grounded evidence in a regulated-document RAG pipeline.

Deterministic grounding is the source of semantic truth.
You must not invent, infer, override, or independently resolve meanings.
You may only reason over the supplied grounded evidence packet and source excerpts.

Your task is to decide whether the supplied evidence is sufficient to:
- answer
- answer_with_warning
- abstain
- retry_once (only when explicitly allowed)

You must not:
- guess acronym expansions
- invent defined-term meanings
- resolve conflicts by preference or intuition
- rely on general world knowledge in place of supplied evidence

Return strict JSON only.
Do not wrap the JSON in markdown fences.
"""

_REVIEWER_OUTPUT_SCHEMA = {
    "action": "proceed | retry_once | abstain",
    "outcome": "answer | answer_with_warning | abstain",
    "sufficient_evidence": "boolean",
    "ambiguity_detected": "boolean",
    "requested_second_pass": "boolean",
    "abstain_reason": "string | null",
    "warning_reason": "string | null",
    "reasoning_notes": ["string", "..."],
    "selected_audit_bindings": ["string", "..."],
    "selected_audit_spans": [[0, 10], "..."],
}

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
class PromptedGroundingReviewer(GroundedEvidenceReviewer):
    """Model-backed bounded reviewer over structured grounded evidence.

    The model is constrained to reason only over deterministic grounding and
    supplied excerpts. It must not independently resolve meanings.
    """

    model_complete: Callable[[str, str], str]

    def review(
        self,
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Review structured grounded evidence using a bounded prompt.

        Args:
            evidence: Structured grounded evidence assembled from retrieved
                chunks.
            has_second_pass_available: Whether one additional retrieval pass may
                still be requested.

        Returns:
            A bounded assessment produced from validated model output, or a
            conservative fallback when the output is invalid.
        """
        user_prompt = self._render_user_prompt(
            evidence=evidence,
            has_second_pass_available=has_second_pass_available,
        )

        try:
            raw = self.model_complete(_REVIEWER_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            return self._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason=f"Reviewer model call failed: {type(exc).__name__}: {exc}",
            )

        return self._parse_response(
            raw=raw,
            evidence=evidence,
            has_second_pass_available=has_second_pass_available,
        )

    @classmethod
    def _render_user_prompt(
        cls,
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> str:
        """Render the bounded reviewer prompt payload."""
        packet = {
            "question": evidence.question,
            "has_second_pass_available": has_second_pass_available,
            "instructions": {
                "deterministic_grounding_is_semantic_truth": True,
                "do_not_independently_resolve_meanings": True,
                "reason_only_over_supplied_evidence": True,
            },
            "response_schema": _REVIEWER_OUTPUT_SCHEMA,
            "documents": [cls._render_document(document) for document in evidence.documents],
        }

        return json.dumps(packet, indent=2, ensure_ascii=False)

    @staticmethod
    def _render_document(document: GroundedEvidenceDocument) -> dict[str, Any]:
        """Render one evidence document into prompt-safe structured form."""
        grounding_payload = document.grounding_payload or {}
        ambiguity_indicators = PromptedGroundingReviewer._extract_ambiguity_indicators(
            grounding_payload
        )

        return {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "chunk_id": document.chunk_id,
            "chunk_span": list(document.chunk_span),
            "score": document.score,
            "grounding_present": document.grounding_payload is not None,
            "ambiguity_indicators": ambiguity_indicators,
            "grounding_payload": grounding_payload,
            "source_excerpt": document.source_excerpt,
        }

    @staticmethod
    def _extract_ambiguity_indicators(grounding_payload: dict[str, Any]) -> dict[str, Any]:
        """Extract lightweight ambiguity markers from grounding payload."""
        acronyms = grounding_payload.get("acronyms", [])
        defined_terms = grounding_payload.get("defined_terms", [])
        structural_references = grounding_payload.get("structural_references", [])

        def _count_unresolved(items: Any) -> int:
            if not isinstance(items, list):
                return 0

            unresolved = 0
            for item in items:
                if not isinstance(item, dict):
                    continue

                selected = item.get("selected")
                if isinstance(selected, dict):
                    resolution_method = selected.get("resolution_method")
                    chosen_meaning_id = selected.get("chosen_meaning_id")
                    if resolution_method == "unresolved" or chosen_meaning_id is None:
                        unresolved += 1
                    continue

                if item.get("resolution_method") == "unresolved":
                    unresolved += 1

            return unresolved

        return {
            "acronym_count": len(acronyms) if isinstance(acronyms, list) else 0,
            "defined_term_count": len(defined_terms) if isinstance(defined_terms, list) else 0,
            "structural_reference_count": (
                len(structural_references) if isinstance(structural_references, list) else 0
            ),
            "unresolved_acronyms": _count_unresolved(acronyms),
            "unresolved_defined_terms": _count_unresolved(defined_terms),
            "unresolved_structural_references": _count_unresolved(structural_references),
        }

    @classmethod
    def _parse_response(
        cls,
        *,
        raw: str,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
    ) -> GroundedEvidenceAssessment:
        """Parse and validate model JSON output conservatively."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer returned malformed JSON.",
            )

        if not isinstance(payload, dict):
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer returned a non-object JSON payload.",
            )

        action = payload.get("action")
        outcome = payload.get("outcome")
        sufficient_evidence = payload.get("sufficient_evidence")
        ambiguity_detected = payload.get("ambiguity_detected")
        requested_second_pass = payload.get("requested_second_pass")
        abstain_reason = payload.get("abstain_reason")
        warning_reason = payload.get("warning_reason")
        reasoning_notes = payload.get("reasoning_notes")
        selected_audit_bindings = payload.get("selected_audit_bindings")
        selected_audit_spans = payload.get("selected_audit_spans")

        valid_actions = {"proceed", "retry_once", "abstain"}
        valid_outcomes = {"answer", "answer_with_warning", "abstain"}

        if action not in valid_actions or outcome not in valid_outcomes:
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer returned an unsupported action or outcome.",
            )

        if not isinstance(sufficient_evidence, bool) or not isinstance(ambiguity_detected, bool):
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer omitted required boolean fields.",
            )

        if not isinstance(requested_second_pass, bool):
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer omitted requested_second_pass.",
            )

        parsed_notes = cls._parse_reasoning_notes(reasoning_notes)
        parsed_bindings = cls._parse_bindings(selected_audit_bindings)
        parsed_spans = cls._parse_spans(selected_audit_spans)

        if action == "retry_once" and not has_second_pass_available:
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer requested retry when no second pass remained.",
            )

        if action == "abstain" and outcome != "abstain":
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer produced an inconsistent abstain decision.",
            )

        if action == "proceed" and outcome == "abstain":
            return cls._fallback_assessment(
                evidence=evidence,
                has_second_pass_available=has_second_pass_available,
                reason="Reviewer produced an inconsistent proceed decision.",
            )

        return GroundedEvidenceAssessment(
            action=action,
            outcome=outcome,
            sufficient_evidence=sufficient_evidence,
            ambiguity_detected=ambiguity_detected,
            requested_second_pass=requested_second_pass,
            abstain_reason=abstain_reason if isinstance(abstain_reason, str) else None,
            warning_reason=warning_reason if isinstance(warning_reason, str) else None,
            reasoning_notes=parsed_notes,
            selected_audit_bindings=parsed_bindings,
            selected_audit_spans=parsed_spans,
        )

    @staticmethod
    def _parse_reasoning_notes(value: Any) -> tuple[str, ...]:
        """Parse reasoning notes into a clean tuple of strings."""
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str))

    @staticmethod
    def _parse_bindings(value: Any) -> tuple[str, ...]:
        """Parse selected audit bindings into a clean tuple of strings."""
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str))

    @staticmethod
    def _parse_spans(value: Any) -> tuple[tuple[int, int], ...]:
        """Parse selected audit spans into validated integer tuples."""
        if not isinstance(value, list):
            return ()

        spans: list[tuple[int, int]] = []

        for item in value:
            if (
                isinstance(item, list | tuple)
                and len(item) == 2
                and isinstance(item[0], int)
                and isinstance(item[1], int)
            ):
                spans.append((item[0], item[1]))

        return tuple(spans)

    @staticmethod
    def _fallback_assessment(
        *,
        evidence: GroundedEvidencePacket,
        has_second_pass_available: bool,
        reason: str,
    ) -> GroundedEvidenceAssessment:
        """Return a conservative bounded fallback assessment."""
        audit_bindings = tuple(dict.fromkeys(doc.document_id for doc in evidence.documents))
        audit_spans = tuple(doc.chunk_span for doc in evidence.documents)

        if has_second_pass_available:
            return GroundedEvidenceAssessment(
                action="retry_once",
                outcome="answer_with_warning",
                sufficient_evidence=False,
                ambiguity_detected=True,
                requested_second_pass=True,
                abstain_reason=None,
                warning_reason="Reviewer output was invalid; requesting one bounded retry.",
                reasoning_notes=(
                    reason,
                    "Fallback behaviour remained bounded and conservative.",
                ),
                selected_audit_bindings=audit_bindings,
                selected_audit_spans=audit_spans,
            )

        return GroundedEvidenceAssessment(
            action="abstain",
            outcome="abstain",
            sufficient_evidence=False,
            ambiguity_detected=True,
            requested_second_pass=False,
            abstain_reason="Reviewer output was invalid after bounded review.",
            warning_reason=None,
            reasoning_notes=(
                reason,
                "Fallback behaviour remained bounded and conservative.",
            ),
            selected_audit_bindings=audit_bindings,
            selected_audit_spans=audit_spans,
        )


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

    @classmethod
    def _build_evidence_packet(
        cls,
        *,
        question: str,
        retrieved_chunks: tuple[Any, ...],
    ) -> GroundedEvidencePacket:
        """Convert retrieved chunks into a structured grounded evidence packet.

        Each retrieved chunk is normalized into a ``GroundedEvidenceDocument``
        carrying document identity, chunk metadata, an optional parsed grounding
        payload, and the associated source excerpt.

        Args:
            question: User question that the evidence packet is being built for.
            retrieved_chunks: Retrieval results from the current pass.

        Returns:
            A ``GroundedEvidencePacket`` suitable for bounded review.
        """
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
