from __future__ import annotations

from dataclasses import dataclass

import anyio

from plainera_unacronym.orchestration.interface import (
    OrchestrationRequest,
    PipelineRequest,
    PipelineRunResult,
)
from plainera_unacronym.orchestration.registry import PipelineRegistry


@dataclass(frozen=True, slots=True)
class _PipelineExecution:
    """Captured pipeline result paired with its registry-order index."""

    index: int
    result: PipelineRunResult


async def run_selected_pipelines(
    registry: PipelineRegistry,
    request: OrchestrationRequest,
) -> tuple[PipelineRunResult, ...]:
    """Run the requested pipelines concurrently.

    Pipelines are executed concurrently, but results are returned in
    deterministic registry resolution order rather than completion order.

    Args:
        registry: Registry containing available pipeline runners.
        request: Top-level orchestration request.

    Returns:
        Pipeline results ordered by registry resolution order.
    """
    runners = registry.resolve(request.targets)
    collected: list[_PipelineExecution] = []

    async def _run_one(index: int) -> None:
        runner = runners[index]
        pipeline_request = PipelineRequest(
            text=request.text,
            options=request.pipeline_options.get(runner.key, {}),
        )
        result = await anyio.to_thread.run_sync(runner.run, pipeline_request)
        collected.append(_PipelineExecution(index=index, result=result))

    async with anyio.create_task_group() as tg:
        for index in range(len(runners)):
            tg.start_soon(_run_one, index)

    ordered = sorted(collected, key=lambda item: item.index)
    return tuple(item.result for item in ordered)
