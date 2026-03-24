from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

type PipelineKey = str

PIPELINE_ACRONYMS: Final[PipelineKey] = "acronyms"
PIPELINE_DEFINED_TERMS: Final[PipelineKey] = "defined_terms"
PIPELINE_STRUCTURAL_REFERENCES: Final[PipelineKey] = "structural_references"


@dataclass(frozen=True, slots=True)
class OrchestrationExecutionOptions:
    """Execution controls for top-level orchestration flow.

    Args:
        partial_success: When true, per-pipeline failures are captured in
            orchestration state and do not abort the whole flow.
    """

    partial_success: bool = True

@dataclass(frozen=True, slots=True)
class OrchestrationRequest:
    """Top-level input for orchestration-driven pipeline selection.

    Args:
        text: Source document text to process.
        targets: Requested pipeline keys to run.
        pipeline_options: Optional per-pipeline execution options keyed by
            pipeline name.
        execution_options: Top-level orchestration execution controls.
    """

    text: str
    targets: tuple[PipelineKey, ...]
    pipeline_options: Mapping[PipelineKey, Mapping[str, object]] = field(default_factory=dict)
    execution_options: OrchestrationExecutionOptions = field(
        default_factory=OrchestrationExecutionOptions
    )


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Single pipeline request derived from orchestration input.

    Args:
        text: Source document text to process.
        options: Pipeline-specific execution options for the selected runner.
    """

    text: str
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Agnostic result returned by a top-level pipeline runner.

    Args:
        pipeline: Stable pipeline key that produced the result.
        payload: Pipeline-native result payload preserved without orchestration-layer reshaping.
        metadata: Optional orchestration-facing metadata associated with the run.
    """

    pipeline: PipelineKey
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)


class PipelineRunner(ABC):
    """Abstract base class for a top-level pipeline runner."""

    key: PipelineKey

    @abstractmethod
    def run(self, request: PipelineRequest) -> PipelineRunResult:
        """Execute the pipeline for a single request.

        Args:
            request: Pipeline-specific request derived from orchestration input.

        Returns:
            Opaque top-level result for the executed pipeline.
        """
