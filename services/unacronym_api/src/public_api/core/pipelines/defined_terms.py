import asyncio
from collections.abc import Mapping

from plainera_unacronym.nlp.extraction.defined_terms.execute import detect_and_resolve_terms
from plainera_unacronym.nlp.extraction.defined_terms.types import TermResolutionResult
from plainera_unacronym.orchestration import PipelineRegistry
from plainera_unacronym.orchestration.interface import (OrchestrationRequest,
                                                        PipelineRunResult,
                                                        PIPELINE_DEFINED_TERMS)
from public_api.core.pipelines.base import BasePipelineExecutor
from public_api.core.processing.defined_term_chunking import merge_defined_term_results
from public_api.schemas.resolve import ResolveOptions, ResolutionMode


class DefinedTermsPipelineExecutor(BasePipelineExecutor):
    """Execute the defined-term pipeline via direct or chunked orchestration paths."""

    key = PIPELINE_DEFINED_TERMS

    def __init__(
        self,
        *,
        pipeline_registry: PipelineRegistry,
        request_timeout_ms: int,
    ):
        """Initialise shared dependencies for defined-term pipeline execution.

        Args:
            pipeline_registry: Registry used for direct non-chunked execution.
            request_timeout_ms: Timeout budget in milliseconds for blocking work
                executed via worker threads.
        """
        super().__init__(
            pipeline_registry=pipeline_registry,
            request_timeout_ms=request_timeout_ms,
        )

    async def _run_defined_term_pipeline_chunk(
        self,
        *,
        text: str,
        options: Mapping[str, object],
    ):
        """Run defined-term detection and resolution for a single text chunk.

        Args:
            text: Chunk text to process.
            options: Per-pipeline options controlling detector/extractor
                configuration, trace flags, and report/state returns.

        Returns:
            The chunk-level payload returned by ``detect_and_resolve_terms``.
        """
        return await self._run_sync_with_timeout(
            lambda: detect_and_resolve_terms(
                text,
                det_cfg=options.get("det_cfg"),
                ext_cfg=options.get("ext_cfg"),
                return_reports=self._bool_option(options, "return_reports", False),
                trace=self._bool_option(options, "trace", False),
                return_state=self._bool_option(options, "return_state", False),
                trace_filter=options.get("trace_filter"),
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
        """Execute the defined-term pipeline over overlapping text chunks.

        Each chunk is processed independently, then merged back into a single
        document-level ``TermResolutionResult`` using the original chunk start
        offsets.

        Args:
            request: Top-level orchestration request.
            opts: Resolved API options for the request. Present for interface
                consistency; not currently used by this executor.
            lang: Optional language hint. Present for interface consistency; not
                currently used by this executor.
            resolution_mode: Optional resolution mode. Present for interface
                consistency; not currently used by this executor.

        Returns:
            A ``PipelineRunResult`` whose payload is the merged defined-term result.

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

        chunk_payloads: list[tuple[int, TermResolutionResult]] = []
        for chunk in chunks:
            try:
                payload = await self._run_defined_term_pipeline_chunk(
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

        merged = merge_defined_term_results(chunk_payloads)

        return PipelineRunResult(
            pipeline=PIPELINE_DEFINED_TERMS,
            payload=merged,
        )
