from typing import Any

import anyio

from plainera_unacronym.orchestration import PipelineRegistry
from plainera_unacronym.orchestration.interface import (
    OrchestrationRequest,
    PIPELINE_ACRONYMS,
    PipelineKey,
    PipelineRunResult,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
)
from plainera_unacronym.orchestration.service import PipelineExecutionOutcome
from plainera_unacronym.orchestration.state import (
    OrchestrationPipelineError,
    OrchestrationState, PipelineErrorCode,
)

from public_api.core.pipelines import (AcronymPipelineExecutor,
                                       DefinedTermsPipelineExecutor,
                                       StructuralPipelineExecutor)
from public_api.core.services import ResolveError
from public_api.db.repos import GlossaryRepository
from public_api.schemas.resolve import ResolveOptions, ResolutionMode


class Orchestrator:
    def __init__(
        self,
        *,
        pipeline_registry: PipelineRegistry,
        glossary_repo: GlossaryRepository,
        request_timeout_ms: int,
        tier2_model: Any | None,
    ) -> None:
        self._pipeline_registry = pipeline_registry
        self._executors = {
            PIPELINE_ACRONYMS: AcronymPipelineExecutor(
                pipeline_registry=pipeline_registry,
                glossary_repo=glossary_repo,
                request_timeout_ms=request_timeout_ms,
                tier2_model=tier2_model,
            ),
            PIPELINE_DEFINED_TERMS: DefinedTermsPipelineExecutor(
                pipeline_registry=pipeline_registry,
                request_timeout_ms=request_timeout_ms,
            ),
            PIPELINE_STRUCTURAL_REFERENCES: StructuralPipelineExecutor(
                pipeline_registry=pipeline_registry,
                request_timeout_ms=request_timeout_ms,
            ),
        }
        self._glossary_repo = glossary_repo
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)
        self._tier2_model = tier2_model


    def _registry_order_targets(
        self,
        targets: tuple[PipelineKey, ...],
    ) -> tuple[PipelineKey, ...]:
        return tuple(runner.key for runner in self._pipeline_registry.resolve(targets))

    @staticmethod
    def _map_pipeline_exception(
        pipeline: PipelineKey,
        exc: Exception,
    ) -> OrchestrationPipelineError:
        if isinstance(exc, ResolveError):
            code = PipelineErrorCode.PIPELINE_TIMEOUT if exc.message == "Resolution timed out." else PipelineErrorCode.PIPELINE_EXECUTION_FAILED
            return OrchestrationPipelineError(
                pipeline=pipeline,
                code=code,
                message=exc.message,
                error_type=type(exc).__name__,
                details=exc.details or {},
            )

        if isinstance(exc, TimeoutError):
            code = PipelineErrorCode.PIPELINE_TIMEOUT
        elif isinstance(exc, ValueError):
            code = PipelineErrorCode.PIPELINE_INVALID_OPTIONS
        else:
            code = PipelineErrorCode.PIPELINE_EXECUTION_FAILED

        return OrchestrationPipelineError(
            pipeline=pipeline,
            code=code,
            message=str(exc) or "Pipeline execution failed.",
            error_type=type(exc).__name__,
            details={},
        )

    async def _execute_pipeline(
        self,
        *,
        pipeline: PipelineKey,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        executor = self._executors.get(pipeline)
        if executor is None:
            raise ValueError(f"No executor configured for pipeline {pipeline!r}.")

        return await executor.execute(
            request=request,
            opts=opts,
            lang=lang,
            resolution_mode=resolution_mode,
        )

    async def execute_orchestration_request(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> OrchestrationState:
        requested_targets = self._registry_order_targets(request.targets)
        state = OrchestrationState.from_requested_targets(requested_targets)

        collected: list[PipelineExecutionOutcome] = []

        async def _run_one(index: int, pipeline: PipelineKey) -> None:
            try:
                result = await self._execute_pipeline(
                    pipeline=pipeline,
                    request=request,
                    opts=opts,
                    lang=lang,
                    resolution_mode=resolution_mode,
                )
            except Exception as exc:
                if not request.execution_options.partial_success:
                    raise

                collected.append(
                    PipelineExecutionOutcome(
                        index=index,
                        pipeline=pipeline,
                        error=self._map_pipeline_exception(pipeline, exc),
                    )
                )
                return

            collected.append(
                PipelineExecutionOutcome(
                    index=index,
                    pipeline=pipeline,
                    result=result,
                )
            )

        async with anyio.create_task_group() as tg:
            for index, pipeline in enumerate(requested_targets):
                tg.start_soon(_run_one, index, pipeline)

        for outcome in sorted(collected, key=lambda item: item.index):
            if outcome.result is not None:
                state.record_success(outcome.result)
            elif outcome.error is not None:
                state.record_failure(outcome.error)
            else:
                raise ValueError(f"Pipeline outcome for {outcome.pipeline!r} had neither result nor error")

        state.finish()
        return state
