from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import anyio
from document_resolution.nlp.common.types import AcronymPipelineResult
from document_resolution.orchestration import PipelineRegistry
from document_resolution.orchestration.interface import (
    OrchestrationRequest,
    PipelineKey,
    PipelineRequest,
    PipelineRunResult,
)
from fastapi import status

from public_api.core.errors import ResolveError
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolutionMode, ResolveOptions


@dataclass(frozen=True, slots=True)
class Chunk:
    """Character-based chunk produced for chunked pipeline execution.

    Attributes:
        start: Inclusive start offset in the source text.
        end: Exclusive end offset in the source text.
        text: Exact substring covering ``text[start:end]``.
    """
    start: int
    end: int
    text: str


class BasePipelineExecutor(ABC):
    """Shared execution mechanics for orchestration pipeline executors.

    This base class centralizes option lookup, chunking decisions, direct runner
    dispatch, timeout handling, and standard chunk-level error construction.
    Concrete executors implement only pipeline-specific chunked execution.
    """

    key: PipelineKey

    def __init__(
        self,
        *,
        pipeline_registry: PipelineRegistry,
        request_timeout_ms: int,
    ) -> None:
        """Store shared execution dependencies for a pipeline executor.

        Args:
            pipeline_registry: Registry used to resolve the configured pipeline
                runner for direct execution.
            request_timeout_ms: Timeout budget in milliseconds for blocking work
                executed via worker threads.
        """
        self._pipeline_registry = pipeline_registry
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)
        self._pipeline_registry = pipeline_registry
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)

    @staticmethod
    def make_chunks(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
        """Split `text` into overlapping windows suitable for chunked processing.

        Chunks are returned in document order and use Python slice semantics, where
        each chunk covers the half-open interval ``[start, end)``. Consecutive
        chunks overlap by ``overlap`` characters to reduce boundary misses for
        entities that straddle chunk edges.

        Args:
            text: Full input text to chunk.
            chunk_size: Maximum number of characters per chunk. Must be greater than
                zero.
            overlap: Number of overlapping characters between adjacent chunks. Must
                satisfy ``0 <= overlap < chunk_size``.

        Returns:
            A list of ``Chunk`` objects in ascending order of ``start``. For empty
            input, returns a single empty chunk at ``[0, 0)``.

        Raises:
            ValueError: If ``chunk_size <= 0``, ``overlap < 0``, or
                ``overlap >= chunk_size``.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")

        n = len(text)
        if n == 0:
            return [Chunk(start=0, end=0, text="")]

        step = chunk_size - overlap
        chunks: list[Chunk] = []

        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(Chunk(start=start, end=end, text=text[start:end]))
            if end >= n:
                break
            start += step

        return chunks

    @staticmethod
    def _bool_option(
        options: Mapping[str, object],
        key: str,
        default: bool,
    ) -> bool:
        """Read a boolean option with type-safe fallback semantics.

        Args:
            options: Raw pipeline options mapping.
            key: Option key to read.
            default: Value to return when the key is missing or not a bool.

        Returns:
            The boolean option value when present and correctly typed; otherwise the
            supplied default.
        """
        value = options.get(key, default)
        return value if isinstance(value, bool) else default

    @staticmethod
    def _int_option(
        options: Mapping[str, object],
        key: str,
        default: int,
    ) -> int:
        """Read an integer option with type-safe fallback semantics.

        Args:
            options: Raw pipeline options mapping.
            key: Option key to read.
            default: Value to return when the key is missing or not an int.

        Returns:
            The integer option value when present and correctly typed; otherwise the
            supplied default.
        """
        value = options.get(key, default)
        return value if isinstance(value, int) else default

    def _pipeline_options(
        self,
        request: OrchestrationRequest,
    ) -> Mapping[str, object]:
        """Return the option mapping configured for this executor's pipeline key.

        Args:
            request: Top-level orchestration request.

        Returns:
            The per-pipeline options mapping for this executor, or an empty mapping
            when no options were provided for the key.
        """
        return request.pipeline_options.get(self.key, {})

    def _should_chunk(
        self,
        *,
        request: OrchestrationRequest,
    ) -> bool:
        """Return whether this request should use chunked execution.

        Chunking is enabled only when the executor's key is included in the
        requested targets, chunking is enabled in pipeline options, and the input
        text length exceeds the configured threshold.

        Args:
            request: Top-level orchestration request.

        Returns:
            True when chunked execution should be used for this pipeline; otherwise
            False.
        """
        if self.key not in request.targets:
            return False

        options = self._pipeline_options(request)
        chunking_enabled = self._bool_option(options, "chunking_enabled", False)
        chunk_threshold_chars = self._int_option(options, "chunk_threshold_chars", 0)

        return chunking_enabled and len(request.text) > chunk_threshold_chars

    async def _execute_direct(
        self,
        *,
        request: OrchestrationRequest,
    ) -> PipelineRunResult:
        """Execute the configured pipeline runner directly on the full input text.

        Args:
            request: Top-level orchestration request.

        Returns:
            The runner's ``PipelineRunResult`` for the full input text.
        """
        runner = self._pipeline_registry.get(self.key)
        pipeline_request = PipelineRequest(
            text=request.text,
            options=dict(self._pipeline_options(request)),
        )
        return await anyio.to_thread.run_sync(runner.run, pipeline_request)

    async def _run_sync_with_timeout(
        self,
        func: Any,
    ) -> Any | AcronymPipelineResult:
        """Run blocking work in a worker thread with the executor timeout applied.

        Args:
            func: Zero-argument callable containing blocking work.

        Returns:
            The callable's return value.

        Raises:
            TimeoutError: If execution exceeds the configured timeout budget.
        """
        return await asyncio.wait_for(
            anyio.to_thread.run_sync(func),
            timeout=self._timeout_s,
        )

    def _chunk_timeout_error(
        self,
        *,
        chunk_start: int,
        chunk_end: int,
    ) -> ResolveError:
        """Build a standardized timeout error for a single chunk failure.

        Args:
            chunk_start: Inclusive start offset of the timed-out chunk.
            chunk_end: Exclusive end offset of the timed-out chunk.

        Returns:
            A ``ResolveError`` describing a chunk-level timeout.
        """
        return ResolveError(
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            code=ErrorCode.SERVICE_UNAVAILABLE,
            message="Resolution timed out.",
            details={
                "timeout_ms": int(self._timeout_s * 1000),
                "chunk": {"start": chunk_start, "end": chunk_end},
            },
        )

    @staticmethod
    def _chunk_failure_error(
        *,
        chunk_start: int,
        chunk_end: int,
        exc: Exception,
    ) -> ResolveError:
        """Build a standardized execution failure error for a single chunk.

        Args:
            chunk_start: Inclusive start offset of the failed chunk.
            chunk_end: Exclusive end offset of the failed chunk.
            exc: Underlying exception raised while processing the chunk.

        Returns:
            A ``ResolveError`` describing a chunk-level execution failure.
        """
        return ResolveError(
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            code=ErrorCode.SERVICE_UNAVAILABLE,
            message="Resolution failed.",
            details={
                "reason": str(exc),
                "chunk": {"start": chunk_start, "end": chunk_end},
            },
        )

    async def execute(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        """Execute the pipeline via direct or chunked mode as configured.

        Args:
            request: Top-level orchestration request.
            opts: Resolved API options for the request.
            lang: Language hint for downstream processing.
            resolution_mode: Resolution mode requested by the caller.

        Returns:
            The pipeline execution result from either the direct or chunked path.
        """
        if self._should_chunk(request=request):
            return await self._execute_chunked(
                request=request,
                opts=opts,
                lang=lang,
                resolution_mode=resolution_mode,
            )
        return await self._execute_direct(request=request)

    @abstractmethod
    async def _execute_chunked(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions | None,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        """Run this pipeline via its chunked execution path."""
