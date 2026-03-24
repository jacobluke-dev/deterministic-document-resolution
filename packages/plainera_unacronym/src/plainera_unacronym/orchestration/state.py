from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter

from .interface import PipelineKey, PipelineRunResult


@dataclass(frozen=True, slots=True)
class OrchestrationPipelineError:
    """Structured per-pipeline orchestration failure.

    Args:
        pipeline: Stable pipeline key for the failed pipeline.
        error_type: Exception or failure type name captured by orchestration.
        message: Human-readable failure message.
        details: Optional structured failure details preserved without
            pipeline-specific interpretation.
    """

    pipeline: PipelineKey
    code: str
    message: str
    error_type: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrchestrationMetadata:
    """Top-level orchestration execution metadata.

    Args:
        started_at_monotonic: Monotonic timestamp captured when orchestration
            state is created.
        finished_at_monotonic: Monotonic timestamp captured when orchestration
            state is finalised, if available.
    """

    started_at_monotonic: float
    finished_at_monotonic: float | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.finished_at_monotonic is None:
            return None
        return int((self.finished_at_monotonic - self.started_at_monotonic) * 1000)


@dataclass(slots=True)
class OrchestrationState:
    """Shared pipeline-agnostic orchestration state for deterministic accumulation.

    Args:
        requested_targets: Pipeline keys resolved for execution in canonical
            orchestration order.
        completed_targets: Pipeline keys recorded as successful, in
            deterministic accumulation order.
        failed_targets: Pipeline keys recorded as failed, in deterministic
            accumulation order.
        results_by_pipeline: Successful per-pipeline results keyed by stable
            pipeline key.
        errors_by_pipeline: Structured per-pipeline failures keyed by stable
            pipeline key.
        metadata: Top-level orchestration execution metadata.
    """

    requested_targets: tuple[PipelineKey, ...]
    completed_targets: tuple[PipelineKey, ...] = ()
    failed_targets: tuple[PipelineKey, ...] = ()
    results_by_pipeline: dict[PipelineKey, PipelineRunResult] = field(default_factory=dict)
    errors_by_pipeline: dict[PipelineKey, OrchestrationPipelineError] = field(default_factory=dict)
    metadata: OrchestrationMetadata = field(
        default_factory=lambda: OrchestrationMetadata(started_at_monotonic=perf_counter())
    )

    @classmethod
    def from_requested_targets(
        cls,
        targets: tuple[PipelineKey, ...],
    ) -> OrchestrationState:
        """Create orchestration state from resolved requested targets.

        Args:
            targets: Requested pipeline keys in canonical orchestration order.

        Returns:
            A new orchestration state initialised with the requested targets.
        """
        return cls(requested_targets=targets)

    def record_success(self, result: PipelineRunResult) -> None:
        """Record a successful pipeline result.

        Args:
            result: Top-level pipeline result to store.

        Raises:
            ValueError: If the pipeline was not requested or has already been
                recorded as completed or failed.
        """
        pipeline = result.pipeline
        if pipeline not in self.requested_targets:
            raise ValueError(f"Cannot record success for unrequested pipeline: {pipeline!r}")
        if pipeline in self.completed_targets:
            raise ValueError(f"Pipeline already recorded as completed: {pipeline!r}")
        if pipeline in self.failed_targets:
            raise ValueError(f"Pipeline already recorded as failed: {pipeline!r}")

        self.results_by_pipeline[pipeline] = result
        self.completed_targets = (*self.completed_targets, pipeline)

    def record_failure(
        self,
        error: OrchestrationPipelineError,
    ) -> None:
        """Record a failed pipeline outcome.

        Args:
            error: Structured per-pipeline failure to store.

        Raises:
            ValueError: If the pipeline was not requested or has already been
                recorded as failed or completed.
        """
        pipeline = error.pipeline
        if pipeline not in self.requested_targets:
            raise ValueError(f"Cannot record failure for unrequested pipeline: {pipeline!r}")
        if pipeline in self.failed_targets:
            raise ValueError(f"Pipeline already recorded as failed: {pipeline!r}")
        if pipeline in self.completed_targets:
            raise ValueError(f"Pipeline already recorded as completed: {pipeline!r}")

        self.errors_by_pipeline[pipeline] = error
        self.failed_targets = (*self.failed_targets, pipeline)

    def finish(self) -> None:
        """Mark orchestration state as finished.

        Raises:
            ValueError: If the orchestration state has already been finished.
        """
        if self.metadata.finished_at_monotonic is not None:
            raise ValueError("Orchestration state already finished")

        self.metadata = OrchestrationMetadata(
            started_at_monotonic=self.metadata.started_at_monotonic,
            finished_at_monotonic=perf_counter(),
        )
