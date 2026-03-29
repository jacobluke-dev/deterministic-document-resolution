import asyncio
from collections.abc import Mapping

from plainera_unacronym.nlp.extraction.structural.execute import detect_and_resolve_structural_references
from plainera_unacronym.nlp.extraction.structural.types import StructuralReferenceResolutionResult
from plainera_unacronym.orchestration import PipelineRegistry
from plainera_unacronym.orchestration.interface import (OrchestrationRequest,
                                                        PipelineRunResult,
                                                        PIPELINE_STRUCTURAL_REFERENCES)
from public_api.core.pipelines.base import BasePipelineExecutor
from public_api.core.processing.structural_chunking import merge_structural_reference_results
from public_api.schemas.resolve import ResolveOptions, ResolutionMode


class StructuralPipelineExecutor(BasePipelineExecutor):
    """Execute the structural-reference pipeline via direct or chunked paths."""

    key = PIPELINE_STRUCTURAL_REFERENCES

    def __init__(
        self,
        *,
        pipeline_registry: PipelineRegistry,
        request_timeout_ms: int,
    ):
        """Initialise shared dependencies for structural pipeline execution.

        Args:
            pipeline_registry: Registry used for direct non-chunked execution.
            request_timeout_ms: Timeout budget in milliseconds for blocking work
                executed via worker threads.
        """
        super().__init__(
            pipeline_registry=pipeline_registry,
            request_timeout_ms=request_timeout_ms,
        )

    async def _run_structural_reference_pipeline_chunk(
        self,
        *,
        text: str,
        options: Mapping[str, object],
    ):
        """Run structural-reference detection and resolution for a single chunk.

        Args:
            text: Chunk text to process.
            options: Per-pipeline options controlling detector/extractor
                configuration and report/state return flags.

        Returns:
            The chunk-level payload returned by
            ``detect_and_resolve_structural_references``.
        """
        return await self._run_sync_with_timeout(
            lambda: detect_and_resolve_structural_references(
                text,
                det_cfg=options.get("det_cfg"),
                ext_cfg=options.get("ext_cfg"),
                return_reports=self._bool_option(options, "return_reports", False),
                return_state=self._bool_option(options, "return_state", False),
            )
        )

    async def _execute_chunked(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str | None = None,
        resolution_mode: ResolutionMode | None = None,
    ) -> PipelineRunResult:
        """Execute the structural-reference pipeline over overlapping text chunks.

        Each chunk is processed independently, then merged back into a single
        document-level ``StructuralReferenceResolutionResult`` using the original
        chunk start offsets.

        Args:
            request: Top-level orchestration request.
            opts: Resolved API options for the request. Present for interface
                consistency; not currently used by this executor.
            lang: Optional language hint. Present for interface consistency; not
                currently used by this executor.
            resolution_mode: Optional resolution mode. Present for interface
                consistency; not currently used by this executor.

        Returns:
            A ``PipelineRunResult`` whose payload is the merged structural-reference
            result.

        Raises:
            ResolveError: If any chunk times out or fails during execution.
        """
        options = self._pipeline_options(request)
        chunk_size_chars = self._int_option(options, "chunk_size_chars", max(1, len(request.text)))
        chunk_overlap_chars = self._int_option(options, "chunk_overlap_chars", 0)

        chunks = self.make_chunks(
            request.text,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
        )

        chunk_payloads: list[tuple[int, StructuralReferenceResolutionResult]] = []
        for chunk in chunks:
            try:
                payload = await self._run_structural_reference_pipeline_chunk(
                    text=chunk.text,
                    options=options,
                )
            except asyncio.TimeoutError as exc:
                raise self._chunk_timeout_error(
                    chunk_start=chunk.start,
                    chunk_end=chunk.end,
                ) from exc
            except Exception as exc:
                raise self._chunk_failure_error(
                    chunk_start=chunk.start,
                    chunk_end=chunk.end,
                    exc=exc,
                ) from exc

            chunk_payloads.append((chunk.start, payload))

        merged = merge_structural_reference_results(chunk_payloads)

        return PipelineRunResult(
            pipeline=PIPELINE_STRUCTURAL_REFERENCES,
            payload=merged,
        )
