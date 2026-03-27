import asyncio
from collections.abc import Mapping
from typing import Any

from plainera_unacronym.nlp.common.types import AcronymDetectorResult, ExtractionResult
from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.orchestration import PIPELINE_ACRONYMS, PipelineRegistry, PipelineRunResult
from plainera_unacronym.orchestration.interface import OrchestrationRequest
from public_api.core.pipelines.base import BasePipelineExecutor
from public_api.core.processing.acronym_chunking import (
    merge_acronym_blocks,
    shift_acronym_blocks,
)
from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import map_acronym_pipeline_to_blocks
from public_api.db.repos import GlossaryRepository
from public_api.schemas.resolve import ResolveOptions, ResolutionMode


class AcronymPipelineExecutor(BasePipelineExecutor):
    key = PIPELINE_ACRONYMS

    def __init__(
        self,
        *,
        pipeline_registry: PipelineRegistry,
        glossary_repo: GlossaryRepository,
        request_timeout_ms: int,
        tier2_model: Any | None,
    ) -> None:
        super().__init__(
            pipeline_registry=pipeline_registry,
            request_timeout_ms=request_timeout_ms,
        )
        self._glossary_repo = glossary_repo
        self._tier2_model = tier2_model

    async def _run_chunk(
        self,
        *,
        text: str,
        options: Mapping[str, object],
    ) -> tuple[AcronymDetectorResult, ExtractionResult]:
        return await self._run_sync_with_timeout(
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
        )

    async def _execute_chunked(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        options = self._pipeline_options(request)
        chunk_size_chars = self._int_option(options, "chunk_size_chars", max(1, len(request.text)))
        chunk_overlap_chars = self._int_option(options, "chunk_overlap_chars", 0)

        chunks = self.make_chunks(
            request.text,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
        )

        all_blocks: list[list[dict[str, Any]]] = []

        for chunk in chunks:
            try:
                det_res, extr = await self._run_chunk(
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

            blocks = map_acronym_pipeline_to_blocks(
                det_res=det_res,
                extr=extr,
                opts=opts,
                lang=lang,
                glossary_repo=self._glossary_repo,
            )
            all_blocks.append(shift_acronym_blocks(blocks, chunk.start))

        merged = merge_acronym_blocks(all_blocks)
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
