from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from plainera_rag_demo.agentic.orchestrator import GroundedEvidenceReviewer
from plainera_rag_demo.agentic.reviewer_rendering import (
    extract_ambiguity_indicators,
    summarize_grounding_payload,
)
from plainera_rag_demo.agentic.types import (
    GroundedEvidenceAssessment,
    GroundedEvidenceDocument,
    GroundedEvidencePacket, GroundedAgentOutcome,
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
    "answer_text": "string | null",
    "abstain_reason": "string | null",
    "warning_reason": "string | null",
    "reasoning_notes": ["string", "..."],
    "selected_audit_bindings": ["string", "..."],
    "selected_audit_spans": [[0, 10], "..."],
}

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

    @classmethod
    def _apply_evidence_guards(
        cls,
        *,
        assessment: GroundedEvidenceAssessment,
        evidence: GroundedEvidencePacket,
    ) -> GroundedEvidenceAssessment:
        """Tighten reviewer output using deterministic packet-level conflict checks."""
        conflicting_keys = cls._find_cross_document_binding_conflicts(evidence)

        if not conflicting_keys or assessment.action == "abstain":
            return assessment

        outcome: GroundedAgentOutcome = assessment.outcome
        warning_reason = assessment.warning_reason
        ambiguity_detected = True

        if outcome == "answer":
            outcome = "answer_with_warning"

        if warning_reason is None:
            joined = ", ".join(conflicting_keys)
            warning_reason = (
                f"Cross-document evidence contains conflicting deterministic bindings for: {joined}."
            )

        return GroundedEvidenceAssessment(
            action=assessment.action,
            outcome=outcome,
            sufficient_evidence=assessment.sufficient_evidence,
            ambiguity_detected=ambiguity_detected,
            requested_second_pass=assessment.requested_second_pass,
            answer_text=assessment.answer_text,
            abstain_reason=assessment.abstain_reason,
            warning_reason=warning_reason,
            reasoning_notes=assessment.reasoning_notes,
            selected_audit_bindings=assessment.selected_audit_bindings,
            selected_audit_spans=assessment.selected_audit_spans,
        )

    @staticmethod
    def _find_cross_document_binding_conflicts(
        evidence: GroundedEvidencePacket,
    ) -> tuple[str, ...]:
        """Return binding keys whose selected meanings conflict across documents."""
        bindings_by_key: dict[str, set[str]] = {}

        for document in evidence.documents:
            payload = document.grounding_payload
            if not isinstance(payload, dict):
                continue

            acronyms = payload.get("acronyms")
            if isinstance(acronyms, list):
                for item in acronyms:
                    if not isinstance(item, dict):
                        continue
                    key = item.get("acronym")
                    selected = item.get("selected")
                    if not isinstance(key, str) or not isinstance(selected, dict):
                        continue
                    definition = selected.get("definition")
                    if isinstance(definition, str) and definition.strip():
                        bindings_by_key.setdefault(f"acronym:{key}", set()).add(definition.strip())

            defined_terms = payload.get("defined_terms")
            if isinstance(defined_terms, list):
                for item in defined_terms:
                    if not isinstance(item, dict):
                        continue
                    key = item.get("normalized_key")
                    chosen_span = item.get("chosen_definition_span")
                    if not isinstance(key, str) or not isinstance(chosen_span, dict):
                        continue
                    definition = chosen_span.get("text")
                    if isinstance(definition, str) and definition.strip():
                        bindings_by_key.setdefault(f"defined_term:{key}", set()).add(definition.strip())

        conflicts = [key for key, values in bindings_by_key.items() if len(values) > 1]
        return tuple(sorted(conflicts))

    @staticmethod
    def _render_document(document: GroundedEvidenceDocument) -> dict[str, Any]:
        """Render one evidence document into prompt-safe structured form."""
        grounding_payload = document.grounding_payload or {}
        grounding_summary = summarize_grounding_payload(grounding_payload)
        ambiguity_indicators = extract_ambiguity_indicators(
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
            "grounding_summary": grounding_summary,
            "source_excerpt": document.source_excerpt[:500],
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
        answer_text = payload.get("answer_text")
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

        if action == "abstain":
            if outcome != "abstain":
                return cls._fallback_assessment(
                    evidence=evidence,
                    has_second_pass_available=has_second_pass_available,
                    reason="Reviewer produced an inconsistent abstain decision.",
                )
            if answer_text is not None:
                return cls._fallback_assessment(
                    evidence=evidence,
                    has_second_pass_available=has_second_pass_available,
                    reason="Reviewer returned answer_text while abstaining.",
                )
        else:
            if outcome == "abstain":
                return cls._fallback_assessment(
                    evidence=evidence,
                    has_second_pass_available=has_second_pass_available,
                    reason="Reviewer produced an inconsistent proceed decision.",
                )
            if not isinstance(answer_text, str) or not answer_text.strip():
                return cls._fallback_assessment(
                    evidence=evidence,
                    has_second_pass_available=has_second_pass_available,
                    reason="Reviewer omitted answer_text for a non-abstain outcome.",
                )

        assessment = GroundedEvidenceAssessment(
            action=action,
            outcome=outcome,
            sufficient_evidence=sufficient_evidence,
            ambiguity_detected=ambiguity_detected,
            requested_second_pass=requested_second_pass,
            answer_text=answer_text.strip() if isinstance(answer_text, str) else None,
            abstain_reason=abstain_reason if isinstance(abstain_reason, str) else None,
            warning_reason=warning_reason if isinstance(warning_reason, str) else None,
            reasoning_notes=parsed_notes,
            selected_audit_bindings=parsed_bindings,
            selected_audit_spans=parsed_spans,
        )
        return cls._apply_evidence_guards(assessment=assessment, evidence=evidence)

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
                answer_text=None,
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
            answer_text=None,
            abstain_reason="Reviewer output was invalid after bounded review.",
            warning_reason=None,
            reasoning_notes=(
                reason,
                "Fallback behaviour remained bounded and conservative.",
            ),
            selected_audit_bindings=audit_bindings,
            selected_audit_spans=audit_spans,
        )
