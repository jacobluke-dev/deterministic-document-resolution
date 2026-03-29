from __future__ import annotations

import asyncio

import pytest

from plainera_unacronym.orchestration.interface import OrchestrationRequest, PipelineRunResult
from public_api.core.pipelines.structural import StructuralPipelineExecutor
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolutionMode
from tests.test_public_api.core.pipelines.conftest import fake_registry


class TestStructuralPipelineExecutor:
    @pytest.fixture
    def resolve_options(self) -> ResolveOptions:
        return ResolveOptions(
            locale="en-GB",
            window_chars=120,
            max_definitions_per_acronym=5,
            include_glossary_enrichment=True,
            return_occurrences=True,
            min_confidence=0.0,
        )

    @pytest.fixture
    def executor(self, fake_registry):
        return StructuralPipelineExecutor(
            pipeline_registry=fake_registry(None),
            request_timeout_ms=250,
        )

    @pytest.mark.anyio
    async def test_run_structural_reference_pipeline_chunk_passes_expected_options(
        self,
        executor,
        _patch,
        monkeypatch,
    ):
        calls: dict[str, object] = {}

        def fake_detect_and_resolve_structural_references(
            text,
            *,
            det_cfg,
            ext_cfg,
            return_reports,
            return_state,
        ):
            calls.update(
                {
                    "text": text,
                    "det_cfg": det_cfg,
                    "ext_cfg": ext_cfg,
                    "return_reports": return_reports,
                    "return_state": return_state,
                }
            )
            return "structural-result"

        async def fake_run_sync_with_timeout(func):
            return func()

        _patch(
            StructuralPipelineExecutor._run_structural_reference_pipeline_chunk,
            detect_and_resolve_structural_references=fake_detect_and_resolve_structural_references,
        )
        monkeypatch.setattr(executor, "_run_sync_with_timeout", fake_run_sync_with_timeout)

        out = await executor._run_structural_reference_pipeline_chunk(
            text="Section 2 and Schedule 1",
            options={
                "det_cfg": "det-cfg",
                "ext_cfg": "ext-cfg",
                "return_reports": True,
                "return_state": True,
            },
        )

        assert out == "structural-result"
        assert calls == {
            "text": "Section 2 and Schedule 1",
            "det_cfg": "det-cfg",
            "ext_cfg": "ext-cfg",
            "return_reports": True,
            "return_state": True,
        }

    @pytest.mark.anyio
    async def test_run_structural_reference_pipeline_chunk_uses_default_flags(
        self,
        executor,
        _patch,
        monkeypatch,
    ):
        calls: dict[str, object] = {}

        def fake_detect_and_resolve_structural_references(
            text,
            *,
            det_cfg,
            ext_cfg,
            return_reports,
            return_state,
        ):
            calls.update(
                {
                    "text": text,
                    "det_cfg": det_cfg,
                    "ext_cfg": ext_cfg,
                    "return_reports": return_reports,
                    "return_state": return_state,
                }
            )
            return "structural-result"

        async def fake_run_sync_with_timeout(func):
            return func()

        _patch(
            StructuralPipelineExecutor._run_structural_reference_pipeline_chunk,
            detect_and_resolve_structural_references=fake_detect_and_resolve_structural_references,
        )
        monkeypatch.setattr(executor, "_run_sync_with_timeout", fake_run_sync_with_timeout)

        out = await executor._run_structural_reference_pipeline_chunk(
            text="Section 2 and Schedule 1",
            options={},
        )

        assert out == "structural-result"
        assert calls == {
            "text": "Section 2 and Schedule 1",
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": False,
            "return_state": False,
        }

    @pytest.mark.anyio
    async def test_execute_chunked_merges_chunk_payloads(
        self,
        executor,
        resolve_options,
        monkeypatch,
        _patch,
    ):
        request = OrchestrationRequest(
            text="abcdefghij",
            targets=("structural_references",),
            pipeline_options={
                "structural_references": {
                    "chunk_size_chars": 6,
                    "chunk_overlap_chars": 2,
                }
            },
        )

        seen_chunks: list[tuple[str, dict[str, object]]] = []

        async def fake_run_structural_reference_pipeline_chunk(*, text: str, options):
            seen_chunks.append((text, dict(options)))
            return f"payload:{text}"

        def fake_merge_structural_reference_results(chunk_payloads):
            assert chunk_payloads == [
                (0, "payload:abcdef"),
                (4, "payload:efghij"),
            ]
            return "merged-structural-result"

        monkeypatch.setattr(
            executor,
            "_run_structural_reference_pipeline_chunk",
            fake_run_structural_reference_pipeline_chunk,
        )
        _patch(
            StructuralPipelineExecutor._execute_chunked,
            merge_structural_reference_results=fake_merge_structural_reference_results,
        )

        out = await executor._execute_chunked(
            request=request,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )

        assert seen_chunks == [
            ("abcdef", {"chunk_size_chars": 6, "chunk_overlap_chars": 2}),
            ("efghij", {"chunk_size_chars": 6, "chunk_overlap_chars": 2}),
        ]
        assert out == PipelineRunResult(
            pipeline="structural_references",
            payload="merged-structural-result",
        )

    @pytest.mark.anyio
    async def test_execute_chunked_raises_timeout_error_for_chunk_timeout(
        self,
        executor,
        resolve_options,
        monkeypatch,
    ):
        request = OrchestrationRequest(
            text="abcdefghij",
            targets=("structural_references",),
            pipeline_options={
                "structural_references": {
                    "chunk_size_chars": 6,
                    "chunk_overlap_chars": 2,
                }
            },
        )

        async def fake_run_structural_reference_pipeline_chunk(*, text: str, options):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(
            executor,
            "_run_structural_reference_pipeline_chunk",
            fake_run_structural_reference_pipeline_chunk,
        )

        with pytest.raises(Exception) as exc:
            await executor._execute_chunked(
                request=request,
                opts=resolve_options,
                lang="en",
                resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            )

        err = exc.value
        assert err.http_status == 503
        assert err.code == ErrorCode.SERVICE_UNAVAILABLE
        assert err.message == "Resolution timed out."
        assert err.details == {
            "timeout_ms": 250,
            "chunk": {"start": 0, "end": 6},
        }

    @pytest.mark.anyio
    async def test_execute_chunked_raises_failure_error_for_chunk_exception(
        self,
        executor,
        resolve_options,
        monkeypatch,
    ):
        request = OrchestrationRequest(
            text="abcdefghij",
            targets=("structural_references",),
            pipeline_options={
                "structural_references": {
                    "chunk_size_chars": 6,
                    "chunk_overlap_chars": 2,
                }
            },
        )

        async def fake_run_structural_reference_pipeline_chunk(*, text: str, options):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            executor,
            "_run_structural_reference_pipeline_chunk",
            fake_run_structural_reference_pipeline_chunk,
        )

        with pytest.raises(Exception) as exc:
            await executor._execute_chunked(
                request=request,
                opts=resolve_options,
                lang="en",
                resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            )

        err = exc.value
        assert err.http_status == 503
        assert err.code == ErrorCode.SERVICE_UNAVAILABLE
        assert err.message == "Resolution failed."
        assert err.details == {
            "reason": "boom",
            "chunk": {"start": 0, "end": 6},
        }
