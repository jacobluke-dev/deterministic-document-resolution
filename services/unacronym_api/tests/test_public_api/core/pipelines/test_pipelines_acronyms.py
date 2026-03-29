from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from plainera_unacronym.orchestration.interface import OrchestrationRequest, PipelineRunResult
from public_api.core.pipelines.acronyms import AcronymPipelineExecutor
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolutionMode, ResolveOptions


class TestAcronymPipelineExecutor:
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
        return AcronymPipelineExecutor(
            pipeline_registry=fake_registry(None),
            glossary_repo=SimpleNamespace(),
            request_timeout_ms=250,
            tier2_model="default-tier2-model",
        )

    @pytest.mark.anyio
    async def test_run_chunk_passes_expected_options_to_detect_and_extract(self, executor, _patch, monkeypatch):
        calls: dict[str, object] = {}

        def fake_detect_and_extract(
            text,
            *,
            det_cfg,
            ext_cfg,
            tier2_model,
            window_left,
            window_right,
            return_reports,
            trace,
            return_state,
            trace_filter,
        ):
            calls.update(
                {
                    "text": text,
                    "det_cfg": det_cfg,
                    "ext_cfg": ext_cfg,
                    "tier2_model": tier2_model,
                    "window_left": window_left,
                    "window_right": window_right,
                    "return_reports": return_reports,
                    "trace": trace,
                    "return_state": return_state,
                    "trace_filter": trace_filter,
                }
            )
            return "det", "extr"

        async def fake_run_sync_with_timeout(func):
            return func()

        _patch(AcronymPipelineExecutor._run_chunk, detect_and_extract=fake_detect_and_extract)
        monkeypatch.setattr(executor, "_run_sync_with_timeout", fake_run_sync_with_timeout)

        out = await executor._run_chunk(
            text="ABC text",
            options={
                "det_cfg": "det-cfg",
                "ext_cfg": "ext-cfg",
                "tier2_model": "override-tier2",
                "window_left": 111,
                "window_right": 222,
                "return_reports": True,
                "trace": True,
                "return_state": True,
                "trace_filter": "only-acronyms",
            },
        )

        assert out == ("det", "extr")
        assert calls == {
            "text": "ABC text",
            "det_cfg": "det-cfg",
            "ext_cfg": "ext-cfg",
            "tier2_model": "override-tier2",
            "window_left": 111,
            "window_right": 222,
            "return_reports": True,
            "trace": True,
            "return_state": True,
            "trace_filter": "only-acronyms",
        }

    @pytest.mark.anyio
    async def test_run_chunk_uses_default_tier2_model_and_window_defaults(self, executor, _patch, monkeypatch):
        calls: dict[str, object] = {}

        def fake_detect_and_extract(
            text,
            *,
            det_cfg,
            ext_cfg,
            tier2_model,
            window_left,
            window_right,
            return_reports,
            trace,
            return_state,
            trace_filter,
        ):
            calls.update(
                {
                    "tier2_model": tier2_model,
                    "window_left": window_left,
                    "window_right": window_right,
                    "return_reports": return_reports,
                    "trace": trace,
                    "return_state": return_state,
                    "trace_filter": trace_filter,
                }
            )
            return "det", "extr"

        async def fake_run_sync_with_timeout(func):
            return func()

        _patch(AcronymPipelineExecutor._run_chunk, detect_and_extract=fake_detect_and_extract)
        monkeypatch.setattr(executor, "_run_sync_with_timeout", fake_run_sync_with_timeout)

        await executor._run_chunk(
            text="ABC text",
            options={},
        )

        assert calls == {
            "tier2_model": "default-tier2-model",
            "window_left": 320,
            "window_right": 280,
            "return_reports": False,
            "trace": False,
            "return_state": False,
            "trace_filter": None,
        }

    @pytest.mark.anyio
    async def test_execute_chunked_maps_shifts_merges_and_attaches_metadata(
        self,
        executor,
        resolve_options,
        monkeypatch,
        _patch,
    ):
        request = OrchestrationRequest(
            text="abcdefghij",
            targets=("acronyms",),
            pipeline_options={
                "acronyms": {
                    "chunk_size_chars": 6,
                    "chunk_overlap_chars": 2,
                }
            },
        )

        chunks_seen: list[tuple[str, dict[str, object]]] = []
        shifted_seen: list[tuple[list[dict[str, object]], int]] = []

        async def fake_run_chunk(*, text: str, options):
            chunks_seen.append((text, dict(options)))
            return f"det:{text}", f"extr:{text}"

        def fake_map_acronym_pipeline_to_blocks(*, det_res, extr, opts, lang, glossary_repo):
            return [{"chunk": det_res, "lang": lang}]

        def fake_shift_acronym_blocks(blocks, delta):
            shifted = [{"chunk": blocks[0]["chunk"], "shift": delta}]
            shifted_seen.append((blocks, delta))
            return shifted

        def fake_merge_acronym_blocks(all_blocks):
            assert all_blocks == [
                [{"chunk": "det:abcdef", "shift": 0}],
                [{"chunk": "det:efghij", "shift": 4}],
            ]
            return [{"merged": True}]

        def fake_attach_resolution_metadata(*, blocks, opts, resolution_mode, glossary_repo):
            assert blocks == [{"merged": True}]
            assert opts is resolve_options
            assert resolution_mode == ResolutionMode.DOMAIN_PRIORITY
            return [{"final": True}]

        monkeypatch.setattr(executor, "_run_chunk", fake_run_chunk)
        _patch(
            AcronymPipelineExecutor._execute_chunked,
            map_acronym_pipeline_to_blocks=fake_map_acronym_pipeline_to_blocks,
            shift_acronym_blocks=fake_shift_acronym_blocks,
            merge_acronym_blocks=fake_merge_acronym_blocks,
            attach_resolution_metadata=fake_attach_resolution_metadata,
        )

        out = await executor._execute_chunked(
            request=request,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )

        assert chunks_seen == [
            ("abcdef", {"chunk_size_chars": 6, "chunk_overlap_chars": 2}),
            ("efghij", {"chunk_size_chars": 6, "chunk_overlap_chars": 2}),
        ]
        assert shifted_seen == [
            ([{"chunk": "det:abcdef", "lang": "en"}], 0),
            ([{"chunk": "det:efghij", "lang": "en"}], 4),
        ]
        assert out == PipelineRunResult(
            pipeline="acronyms",
            payload=[{"final": True}],
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
            targets=("acronyms",),
            pipeline_options={
                "acronyms": {
                    "chunk_size_chars": 6,
                    "chunk_overlap_chars": 2,
                }
            },
        )

        async def fake_run_chunk(*, text: str, options):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(executor, "_run_chunk", fake_run_chunk)

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
            targets=("acronyms",),
            pipeline_options={
                "acronyms": {
                    "chunk_size_chars": 6,
                    "chunk_overlap_chars": 2,
                }
            },
        )

        async def fake_run_chunk(*, text: str, options):
            raise RuntimeError("boom")

        monkeypatch.setattr(executor, "_run_chunk", fake_run_chunk)

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
