from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import anyio
from fastapi import status

from plainera_unacronym.orchestration import PipelineRegistry
from plainera_unacronym.orchestration.interface import (
    OrchestrationRequest,
    PipelineKey,
    PipelineRequest,
    PipelineRunResult,
)
from public_api.core.services import ResolveError
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolutionMode


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    end: int
    text: str


class BasePipelineExecutor(ABC):
    """Shared execution mechanics for orchestration pipeline executors."""

    key: PipelineKey

    def __init__(
        self,
        *,
        pipeline_registry: PipelineRegistry,
        request_timeout_ms: int,
    ) -> None:
        self._pipeline_registry = pipeline_registry
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)

    @staticmethod
    def make_chunks(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
        """
            Split `text` into overlapping windows suitable for chunked processing.

            Chunks are returned in order and use Python-slice semantics:
            each chunk covers the half-open interval [start, end), where `end` is exclusive.

            The next chunk starts at `previous_start + (chunk_size - overlap)`, ensuring an
            overlap region of `overlap` characters between consecutive chunks. Overlap is
            used to avoid missing matches (e.g. acronyms/definitions) that straddle chunk
            boundaries.

            Args:
                text: Full input text to chunk.
                chunk_size: Maximum number of characters per chunk. Must be > 0.
                overlap: Number of characters of overlap between consecutive chunks.
                    Must satisfy 0 <= overlap < chunk_size.

            Returns:
                A list of `Chunk` objects in ascending order of `start`. For empty input,
                returns a single chunk with start=end=0.

            Raises:
                ValueError: If `chunk_size <= 0`, `overlap < 0`, or `overlap >= chunk_size`.

            Notes:
                - This function does not attempt to align chunks to word/sentence boundaries.
                  It is purely character-based for determinism.
                - Coverage is complete: concatenating chunk ranges covers [0, len(text)]
                  with possible overlaps but no gaps.
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

    def _pipeline_options(
        self,
        request: OrchestrationRequest,
    ) -> Mapping[str, object]:
        return request.pipeline_options.get(self.key, {})


    def _should_chunk(
        self,
        *,
        request: OrchestrationRequest,
    ) -> bool:
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
        runner = self._pipeline_registry.get(self.key)
        pipeline_request = PipelineRequest(
            text=request.text,
            options=dict(self._pipeline_options(request)),
        )
        return await anyio.to_thread.run_sync(runner.run, pipeline_request)

    async def _run_sync_with_timeout(
        self,
        func: Any,
    ) -> Any:
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
