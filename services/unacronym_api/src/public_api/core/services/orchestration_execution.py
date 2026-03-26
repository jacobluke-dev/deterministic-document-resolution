import asyncio
from collections.abc import Mapping
from typing import Any

import anyio
from fastapi import status

from plainera_unacronym.nlp.common.types import AcronymDetectorResult, ExtractionResult
from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.nlp.extraction.defined_terms.execute import detect_and_resolve_terms
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult
from plainera_unacronym.nlp.extraction.structural.execute import detect_and_resolve_structural_references
from plainera_unacronym.orchestration import PipelineRegistry
from plainera_unacronym.orchestration.interface import (
    OrchestrationRequest,
    PIPELINE_ACRONYMS,
    PipelineKey,
    PipelineRequest,
    PipelineRunResult, PIPELINE_DEFINED_TERMS, PIPELINE_STRUCTURAL_REFERENCES,
)
from plainera_unacronym.orchestration.service import PipelineExecutionOutcome
from plainera_unacronym.orchestration.state import (
    OrchestrationPipelineError,
    OrchestrationState,
)
from public_api.core.processing.acronym_chunking import make_chunks, merge_blocks, shift_blocks
from public_api.core.processing.defined_term_chunking import merge_defined_term_results
from public_api.core.processing.structural_chunking import merge_structural_reference_results
from public_api.core.services import ResolveError
from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import map_pipeline_to_blocks
from public_api.db.repos import GlossaryRepository
from public_api.schemas.error import ErrorCode
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
        self._glossary_repo = glossary_repo
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)
        self._tier2_model = tier2_model

    @staticmethod
    def _bool_option(
        options: Mapping[str, object],
        key: str,
        default: bool,
    ) -> bool:
        value = options.get(key, default)
        return value if isinstance(value, bool) else default

    @staticmethod
    def _int_option(
        options: Mapping[str, object],
        key: str,
        default: int,
    ) -> int:
        value = options.get(key, default)
        return value if isinstance(value, int) else default

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
            code = "PIPELINE_TIMEOUT" if exc.message == "Resolution timed out." else "PIPELINE_EXECUTION_FAILED"
            return OrchestrationPipelineError(
                pipeline=pipeline,
                code=code,
                message=exc.message,
                error_type=type(exc).__name__,
                details=exc.details or {},
            )

        if isinstance(exc, TimeoutError):
            code = "PIPELINE_TIMEOUT"
        elif isinstance(exc, ValueError):
            code = "PIPELINE_INVALID_OPTIONS"
        else:
            code = "PIPELINE_EXECUTION_FAILED"

        return OrchestrationPipelineError(
            pipeline=pipeline,
            code=code,
            message=str(exc) or "Pipeline execution failed.",
            error_type=type(exc).__name__,
            details={},
        )

    @staticmethod
    def _pipeline_options(
        request: OrchestrationRequest,
        pipeline: PipelineKey,
    ) -> Mapping[str, object]:
        return request.pipeline_options.get(pipeline, {})

    @classmethod
    def _should_chunk_pipeline(
        cls,
        *,
        request: OrchestrationRequest,
        pipeline: PipelineKey,
    ) -> bool:
        if pipeline not in request.targets:
            return False

        options = request.pipeline_options.get(pipeline, {})
        chunking_enabled = cls._bool_option(options, "chunking_enabled", False)
        chunk_threshold_chars = cls._int_option(options, "chunk_threshold_chars", 0)

        return chunking_enabled and len(request.text) > chunk_threshold_chars

    async def _run_acronym_pipeline_chunk(
        self,
        *,
        text: str,
        options: Mapping[str, object],
    ) -> tuple[AcronymDetectorResult, ExtractionResult]:
        return await asyncio.wait_for(
            anyio.to_thread.run_sync(
                lambda: detect_and_extract(
                    text,
                    det_cfg=options.get("det_cfg"),
                    ext_cfg=options.get("ext_cfg"),
                    tier2_model=options.get("tier2_model", self._tier2_model),
                    window_left=self._int_option(options, "window_left", 320),
                    window_right=self._int_option(options, "window_right", 280),
                    return_reports=self._bool_option(options, "return_reports", False),
                    trace=self._bool_option(options, "trace", False),
                    return_state=self._bool_option(options, "return_state", False),
                    trace_filter=options.get("trace_filter"),
                )
            ),
            timeout=self._timeout_s,
        )

    async def _run_chunked_acronym_pipeline(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        options = self._pipeline_options(request, PIPELINE_ACRONYMS)
        chunk_size_chars = self._int_option(options, "chunk_size_chars", max(1, len(request.text)))
        chunk_overlap_chars = self._int_option(options, "chunk_overlap_chars", 0)

        chunks = make_chunks(
            request.text,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
        )

        all_blocks: list[list[dict[str, Any]]] = []

        for chunk in chunks:
            try:
                det_res, extr = await self._run_acronym_pipeline_chunk(
                    text=chunk.text,
                    options=options,
                )
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

    async def _run_defined_term_pipeline_chunk(
        self,
        *,
        text: str,
        options: Mapping[str, object],
    ):
        return await asyncio.wait_for(
            anyio.to_thread.run_sync(
                lambda: detect_and_resolve_terms(
                    text,
                    det_cfg=options.get("det_cfg"),
                    ext_cfg=options.get("ext_cfg"),
                    return_reports=self._bool_option(options, "return_reports", False),
                    trace=self._bool_option(options, "trace", False),
                    return_state=self._bool_option(options, "return_state", False),
                    trace_filter=options.get("trace_filter"),
                )
            ),
            timeout=self._timeout_s,
        )

    async def _run_chunked_defined_term_pipeline(
        self,
        *,
        request: OrchestrationRequest,
    ) -> PipelineRunResult:
        options = self._pipeline_options(request, PIPELINE_DEFINED_TERMS)
        chunk_size_chars = self._int_option(options, "chunk_size_chars", max(1, len(request.text)))
        chunk_overlap_chars = self._int_option(options, "chunk_overlap_chars", 0)

        chunks = make_chunks(
            request.text,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
        )

        chunk_payloads: list[tuple[int, TermResolutionResult]] = []
        for chunk in chunks:
            try:
                payload = await self._run_defined_term_pipeline_chunk(
                    text=chunk.text,
                    options=options,
                )
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

            chunk_payloads.append((chunk.start, payload))

        merged = merge_defined_term_results(chunk_payloads)

        return PipelineRunResult(
            pipeline=PIPELINE_DEFINED_TERMS,
            payload=merged,
        )

    async def _run_structural_reference_pipeline_chunk(
        self,
        *,
        text: str,
        options: Mapping[str, object],
    ):
        return await asyncio.wait_for(
            anyio.to_thread.run_sync(
                lambda: detect_and_resolve_structural_references(
                    text,
                    det_cfg=options.get("det_cfg"),
                    ext_cfg=options.get("ext_cfg"),
                    return_reports=self._bool_option(options, "return_reports", False),
                    return_state=self._bool_option(options, "return_state", False),
                )
            ),
            timeout=self._timeout_s,
        )

    async def _run_chunked_structural_reference_pipeline(
        self,
        *,
        request: OrchestrationRequest,
    ) -> PipelineRunResult:
        options = self._pipeline_options(request, PIPELINE_STRUCTURAL_REFERENCES)
        chunk_size_chars = self._int_option(options, "chunk_size_chars", max(1, len(request.text)))
        chunk_overlap_chars = self._int_option(options, "chunk_overlap_chars", 0)

        chunks = make_chunks(
            request.text,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
        )

        chunk_payloads = []
        for chunk in chunks:
            try:
                payload = await self._run_structural_reference_pipeline_chunk(
                    text=chunk.text,
                    options=options,
                )
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

            chunk_payloads.append((chunk.start, payload))

        merged = merge_structural_reference_results(chunk_payloads)

        return PipelineRunResult(
            pipeline=PIPELINE_STRUCTURAL_REFERENCES,
            payload=merged,
        )


    async def _execute_pipeline_direct(
        self,
        *,
        pipeline: PipelineKey,
        request: OrchestrationRequest,
    ) -> PipelineRunResult:
        runner = self._pipeline_registry.get(pipeline)
        pipeline_request = PipelineRequest(
            text=request.text,
            options=dict(self._pipeline_options(request, pipeline)),
        )
        return await anyio.to_thread.run_sync(runner.run, pipeline_request)

    async def _execute_pipeline_chunked(
        self,
        *,
        pipeline: PipelineKey,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        if pipeline == PIPELINE_ACRONYMS:
            return await self._run_chunked_acronym_pipeline(
                request=request,
                opts=opts,
                lang=lang,
                resolution_mode=resolution_mode,
            )

        if pipeline == PIPELINE_DEFINED_TERMS:
            return await self._run_chunked_defined_term_pipeline(
                request=request,
            )

        if pipeline == PIPELINE_STRUCTURAL_REFERENCES:
            return await self._run_chunked_structural_reference_pipeline(
                request=request,
            )

        raise ValueError(f"Chunked execution not implemented for pipeline {pipeline!r}.")

    async def _execute_pipeline(
        self,
        *,
        pipeline: PipelineKey,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        if self._should_chunk_pipeline(request=request, pipeline=pipeline):
            return await self._execute_pipeline_chunked(
                pipeline=pipeline,
                request=request,
                opts=opts,
                lang=lang,
                resolution_mode=resolution_mode,
            )

        return await self._execute_pipeline_direct(
            pipeline=pipeline,
            request=request,
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
