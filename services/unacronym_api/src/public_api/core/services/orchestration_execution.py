import asyncio
from typing import Any

import anyio

from plainera_unacronym.nlp.common.types import ExtractionResult, AcronymDetectorResult
from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.orchestration import PipelineRegistry
from plainera_unacronym.orchestration.interface import OrchestrationRequest, PIPELINE_ACRONYMS, PipelineRunResult, \
    PipelineKey
from plainera_unacronym.orchestration.service import run_selected_pipelines
from plainera_unacronym.orchestration.state import OrchestrationState, OrchestrationPipelineError
from public_api.core.auth.chunking import shift_blocks, merge_blocks, make_chunks
from public_api.core.services import ResolveError
from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import map_pipeline_to_blocks
from public_api.core.settings import app_settings
from public_api.db.repos import GlossaryRepository
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolutionMode
from fastapi import status


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
        self._glossary_repo = glossary_repo
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)
        self._tier2_model = tier2_model

    @staticmethod
    def _should_chunk_acronyms(
        *,
        text: str,
        targets: tuple[PipelineKey, ...],
    ) -> bool:
        """Return True when the acronym pipeline should use chunked execution."""
        return (
            PIPELINE_ACRONYMS in targets
            and bool(app_settings.CHUNKING_ENABLED)
            and len(text) > int(app_settings.CHUNK_THRESHOLD_CHARS)
        )

    @staticmethod
    def _pipeline_error_from_resolve_error(exc: ResolveError) -> OrchestrationPipelineError:
        """Map a chunked acronym resolve error into orchestration failure shape."""
        code = "PIPELINE_TIMEOUT" if exc.message == "Resolution timed out." else "PIPELINE_EXECUTION_FAILED"
        return OrchestrationPipelineError(
            pipeline=PIPELINE_ACRONYMS,
            code=code,
            message=exc.message,
            error_type=type(exc).__name__,
            details=exc.details or {},
        )

    def _registry_order_targets(
        self,
        targets: tuple[PipelineKey, ...],
    ) -> tuple[PipelineKey, ...]:
        """Resolve requested targets into deterministic registry order."""
        return tuple(runner.key for runner in self._pipeline_registry.resolve(targets))

    async def _run_with_optional_chunked_acronyms(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> OrchestrationState:
        """Execute requested pipelines, using chunked execution for acronyms when needed."""
        requested_targets = self._registry_order_targets(request.targets)
        state = OrchestrationState.from_requested_targets(requested_targets)

        acronym_result: PipelineRunResult | None = None
        acronym_error: OrchestrationPipelineError | None = None

        non_acronym_targets = tuple(t for t in request.targets if t != PIPELINE_ACRONYMS)
        non_acronym_state: OrchestrationState | None = None

        if self._should_chunk_acronyms(text=request.text, targets=request.targets):
            try:
                acronym_result = await self._run_chunked_acronym_pipeline(
                    text=request.text,
                    opts=opts,
                    lang=lang,
                    resolution_mode=resolution_mode,
                )
            except ResolveError as exc:
                if not request.execution_options.partial_success:
                    raise
                acronym_error = self._pipeline_error_from_resolve_error(exc)
        elif PIPELINE_ACRONYMS in request.targets:
            acronym_request = OrchestrationRequest(
                text=request.text,
                targets=(PIPELINE_ACRONYMS,),
                pipeline_options={
                    PIPELINE_ACRONYMS: request.pipeline_options.get(PIPELINE_ACRONYMS, {}),
                },
                execution_options=request.execution_options,
            )
            acronym_state = await run_selected_pipelines(self._pipeline_registry, acronym_request)
            for target in acronym_state.completed_targets:
                state.record_success(acronym_state.results_by_pipeline[target])
            for target in acronym_state.failed_targets:
                state.record_failure(acronym_state.errors_by_pipeline[target])

        if non_acronym_targets:
            non_acronym_request = OrchestrationRequest(
                text=request.text,
                targets=non_acronym_targets,
                pipeline_options={
                    key: value
                    for key, value in request.pipeline_options.items()
                    if key in non_acronym_targets
                },
                execution_options=request.execution_options,
            )
            non_acronym_state = await run_selected_pipelines(self._pipeline_registry, non_acronym_request)

        if acronym_result is not None:
            state.record_success(acronym_result)
        elif acronym_error is not None:
            state.record_failure(acronym_error)

        if non_acronym_state is not None:
            for target in requested_targets:
                if target in state.completed_targets or target in state.failed_targets:
                    continue
                if target in non_acronym_state.results_by_pipeline:
                    state.record_success(non_acronym_state.results_by_pipeline[target])
                    continue
                if target in non_acronym_state.errors_by_pipeline:
                    state.record_failure(non_acronym_state.errors_by_pipeline[target])
                    continue

        state.finish()
        return state

    async def _run_chunked_acronym_pipeline(
        self,
        *,
        text: str,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        """Run the acronym pipeline in chunked mode and return a single pipeline result."""
        chunks = make_chunks(
            text,
            chunk_size=int(app_settings.CHUNK_SIZE_CHARS),
            overlap=int(app_settings.CHUNK_OVERLAP_CHARS),
        )

        all_blocks: list[list[dict[str, Any]]] = []

        for chunk in chunks:
            try:
                det_res, extr = await self._run_pipeline(chunk.text, opts)
            except asyncio.TimeoutError as exc:
                raise ResolveError(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution timed out.",
                    details={
                        "timeout_ms": int(self._timeout_s * 1000),
                        "chunk": {"start": chunk.start, "end": chunk.end},
                    },
                ) from exc
            except Exception as exc:
                raise ResolveError(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution failed.",
                    details={
                        "reason": str(exc),
                        "chunk": {"start": chunk.start, "end": chunk.end},
                    },
                ) from exc

            blocks = map_pipeline_to_blocks(
                det_res=det_res,
                extr=extr,
                opts=opts,
                lang=lang,
                glossary_repo=self._glossary_repo,
            )
            all_blocks.append(shift_blocks(blocks, chunk.start))

        merged = merge_blocks(all_blocks)
        merged = attach_resolution_metadata(
            blocks=merged,
            opts=opts,
            resolution_mode=resolution_mode,
            glossary_repo=self._glossary_repo,
        )

        return PipelineRunResult(
            pipeline=PIPELINE_ACRONYMS,
            payload=merged,
        )

    async def _run_pipeline(self, text: str, opts: ResolveOptions) -> tuple[AcronymDetectorResult, ExtractionResult]:
        """Execute the acronym detection and extraction pipeline with a timeout."""
        return await asyncio.wait_for(
            anyio.to_thread.run_sync(
                lambda: detect_and_extract(
                    text,
                    det_cfg=None,
                    ext_cfg=None,
                    tier2_model=self._tier2_model,
                    window_left=int(opts.window_chars),
                    window_right=int(opts.window_chars),
                    return_reports=False,
                    trace=False,
                    return_state=False,
                )
            ),
            timeout=self._timeout_s,
        )

    async def execute_orchestration_request(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> OrchestrationState:
        return await self._run_with_optional_chunked_acronyms(
            request=request,
            opts=opts,
            lang=lang,
            resolution_mode=resolution_mode,
        )
