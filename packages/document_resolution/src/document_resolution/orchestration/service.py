from __future__ import annotations

from dataclasses import dataclass

import anyio

from document_resolution.orchestration.interface import (
    OrchestrationRequest,
    PipelineKey,
    PipelineRequest,
    PipelineRunResult,
)
from document_resolution.orchestration.registry import PipelineRegistry
from document_resolution.orchestration.state import (
    OrchestrationPipelineError,
    OrchestrationState,
    PipelineErrorCode,
)


@dataclass(frozen=True, slots=True)
class PipelineExecutionOutcome:
    """Captured pipeline outcome paired with its registry-order index."""

    index: int
    pipeline: PipelineKey
    result: PipelineRunResult | None = None
    error: OrchestrationPipelineError | None = None


def _classify_pipeline_exception(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return PipelineErrorCode.PIPELINE_TIMEOUT
    if isinstance(exc, ValueError):
        return PipelineErrorCode.PIPELINE_INVALID_OPTIONS
    return PipelineErrorCode.PIPELINE_EXECUTION_FAILED


async def run_selected_pipelines(
    registry: PipelineRegistry,
    request: OrchestrationRequest,
) -> OrchestrationState:
    """Run the requested pipelines concurrently and accumulate orchestration state.

    Pipelines are executed independently and outcomes are recorded in
    deterministic registry resolution order rather than completion order.

    Args:
        registry: Registry containing available pipeline runners.
        request: Top-level orchestration request.

    Returns:
        Orchestration state populated with requested targets, per-pipeline
        results, per-pipeline failures, and execution metadata.

    Raises:
        Exception: Re-raises a pipeline execution error when partial-success
            mode is disabled.
    """
    runners = registry.resolve(request.targets)
    requested_targets = tuple(runner.key for runner in runners)
    state = OrchestrationState.from_requested_targets(requested_targets)

    collected: list[PipelineExecutionOutcome] = []

    async def _run_one(index: int) -> None:
        runner = runners[index]
        pipeline_request = PipelineRequest(
            text=request.text,
            options=request.pipeline_options.get(runner.key, {}),
        )

        try:
            result = await anyio.to_thread.run_sync(runner.run, pipeline_request)
        except Exception as exc:
            if not request.execution_options.partial_success:
                raise

            collected.append(
                PipelineExecutionOutcome(
                    index=index,
                    pipeline=runner.key,
                    error=OrchestrationPipelineError(
                        pipeline=runner.key,
                        code=PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                        message=str(exc) or "Pipeline execution failed.",
                        error_type=type(exc).__name__,
                    ),
                )
            )
            return

        collected.append(
            PipelineExecutionOutcome(
                index=index,
                pipeline=runner.key,
                result=result,
            )
        )

    async with anyio.create_task_group() as tg:
        for index in range(len(runners)):
            tg.start_soon(_run_one, index)

    ordered = sorted(collected, key=lambda item: item.index)

    for outcome in ordered:
        if outcome.result is not None:
            state.record_success(outcome.result)
            continue

        if outcome.error is not None:
            state.record_failure(outcome.error)
            continue

        raise ValueError(f"Pipeline outcome for {outcome.pipeline!r} had neither result nor error")

    state.finish()
    return state
