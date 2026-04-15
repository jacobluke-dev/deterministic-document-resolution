from __future__ import annotations

import time

import pytest
from document_resolution.orchestration.interface import (
    OrchestrationRequest,
    PipelineRequest,
    PipelineRunResult,
)
from public_api.core.errors import ResolveError
from public_api.core.pipelines.base import BasePipelineExecutor
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolutionMode, ResolveOptions


class _FakeRunner:
    def __init__(self, result: PipelineRunResult):
        self.result = result
        self.calls: list[PipelineRequest] = []

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        self.calls.append(request)
        return self.result


class _TestExecutor(BasePipelineExecutor):
    key = "defined_terms"

    def __init__(self, *, pipeline_registry, request_timeout_ms: int = 100):
        super().__init__(
            pipeline_registry=pipeline_registry,
            request_timeout_ms=request_timeout_ms,
        )
        self.chunked_calls: list[dict[str, object]] = []

    async def _execute_chunked(
        self,
        *,
        request: OrchestrationRequest,
        opts: ResolveOptions | None,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> PipelineRunResult:
        self.chunked_calls.append(
            {
                "request": request,
                "opts": opts,
                "lang": lang,
                "resolution_mode": resolution_mode,
            }
        )
        return PipelineRunResult(
            pipeline=self.key,
            payload={"mode": "chunked"},
        )

class TestMakeChunks:
    def test_make_chunks_basic_overlap(self):
        text = "x" * 100

        chunks = BasePipelineExecutor.make_chunks(
            text,
            chunk_size=30,
            overlap=10,
        )

        assert [(c.start, c.end) for c in chunks] == [
            (0, 30),
            (20, 50),
            (40, 70),
            (60, 90),
            (80, 100),
        ]
        assert [c.text for c in chunks] == [
            text[0:30],
            text[20:50],
            text[40:70],
            text[60:90],
            text[80:100],
        ]

    def test_make_chunks_returns_single_empty_chunk_for_empty_text(self):
        chunks = BasePipelineExecutor.make_chunks("", chunk_size=30, overlap=10)

        assert [(c.start, c.end, c.text) for c in chunks] == [(0, 0, "")]

    def test_make_chunks_returns_single_chunk_when_text_shorter_than_chunk_size(self):
        text = "hello"

        chunks = BasePipelineExecutor.make_chunks(text, chunk_size=30, overlap=10)

        assert [(c.start, c.end, c.text) for c in chunks] == [(0, 5, "hello")]

    @pytest.mark.parametrize(
        ("chunk_size", "overlap", "message"),
        [
            (0, 0, "chunk_size must be > 0"),
            (10, -1, "overlap must be >= 0"),
            (10, 10, "overlap must be < chunk_size"),
            (10, 11, "overlap must be < chunk_size"),
        ],
    )
    def test_make_chunks_raises_for_invalid_arguments(self, chunk_size, overlap, message):
        with pytest.raises(ValueError, match=message):
            BasePipelineExecutor.make_chunks("abcdef", chunk_size=chunk_size, overlap=overlap)

class TestBasePipelineExecutorHelpers:
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

    def test_bool_option_returns_value_when_bool(self):
        assert BasePipelineExecutor._bool_option({"x": True}, "x", False) is True

    def test_bool_option_falls_back_when_not_bool(self):
        assert BasePipelineExecutor._bool_option({"x": "yes"}, "x", False) is False

    def test_int_option_returns_value_when_int(self):
        assert BasePipelineExecutor._int_option({"x": 7}, "x", 3) == 7

    def test_int_option_falls_back_when_not_int(self):
        assert BasePipelineExecutor._int_option({"x": "7"}, "x", 3) == 3

    def test_pipeline_options_returns_mapping_for_executor_key(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="abc",
            targets=("defined_terms",),
            pipeline_options={"defined_terms": {"chunking_enabled": True}},
        )

        assert executor._pipeline_options(request) == {"chunking_enabled": True}

    def test_pipeline_options_returns_empty_mapping_when_missing(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="abc",
            targets=("defined_terms",),
            pipeline_options={},
        )

        assert executor._pipeline_options(request) == {}

    def test_should_chunk_false_when_pipeline_not_requested(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="x" * 500,
            targets=("acronyms",),
            pipeline_options={
                "defined_terms": {
                    "chunking_enabled": True,
                    "chunk_threshold_chars": 100,
                }
            },
        )

        assert executor._should_chunk(request=request) is False

    def test_should_chunk_false_when_chunking_disabled(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="x" * 500,
            targets=("defined_terms",),
            pipeline_options={
                "defined_terms": {
                    "chunking_enabled": False,
                    "chunk_threshold_chars": 100,
                }
            },
        )

        assert executor._should_chunk(request=request) is False

    def test_should_chunk_false_when_text_does_not_exceed_threshold(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="x" * 100,
            targets=("defined_terms",),
            pipeline_options={
                "defined_terms": {
                    "chunking_enabled": True,
                    "chunk_threshold_chars": 100,
                }
            },
        )

        assert executor._should_chunk(request=request) is False

    def test_should_chunk_true_when_enabled_and_text_exceeds_threshold(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="x" * 101,
            targets=("defined_terms",),
            pipeline_options={
                "defined_terms": {
                    "chunking_enabled": True,
                    "chunk_threshold_chars": 100,
                }
            },
        )

        assert executor._should_chunk(request=request) is True

    @pytest.mark.anyio
    async def test_execute_direct_runs_registry_runner_with_pipeline_options(self, fake_registry):
        expected = PipelineRunResult(
            pipeline="defined_terms",
            payload={"ok": True},
        )
        runner = _FakeRunner(expected)
        executor = _TestExecutor(pipeline_registry=fake_registry(runner))
        request = OrchestrationRequest(
            text="example text",
            targets=("defined_terms",),
            pipeline_options={"defined_terms": {"trace": True}},
        )

        out = await executor._execute_direct(request=request)

        assert out is expected
        assert len(runner.calls) == 1
        assert runner.calls[0] == PipelineRequest(
            text="example text",
            options={"trace": True},
        )

    @pytest.mark.anyio
    async def test_run_sync_with_timeout_returns_result(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None), request_timeout_ms=50)

        out = await executor._run_sync_with_timeout(lambda: 123)

        assert out == 123

    @pytest.mark.anyio
    async def test_run_sync_with_timeout_raises_timeout_error(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None), request_timeout_ms=10)

        def slow():
            time.sleep(0.05)

        with pytest.raises(TimeoutError):
            await executor._run_sync_with_timeout(slow)

    def test_chunk_timeout_error_has_expected_shape(self, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None), request_timeout_ms=250)

        err = executor._chunk_timeout_error(chunk_start=10, chunk_end=20)

        assert isinstance(err, ResolveError)
        assert err.http_status == 503
        assert err.code == ErrorCode.SERVICE_UNAVAILABLE
        assert err.message == "Resolution timed out."
        assert err.details == {
            "timeout_ms": 250,
            "chunk": {"start": 10, "end": 20},
        }

    def test_chunk_failure_error_has_expected_shape(self):
        err = BasePipelineExecutor._chunk_failure_error(
            chunk_start=10,
            chunk_end=20,
            exc=RuntimeError("boom"),
        )

        assert isinstance(err, ResolveError)
        assert err.http_status == 503
        assert err.code == ErrorCode.SERVICE_UNAVAILABLE
        assert err.message == "Resolution failed."
        assert err.details == {
            "reason": "boom",
            "chunk": {"start": 10, "end": 20},
        }

    @pytest.mark.anyio
    async def test_execute_uses_chunked_path_when_should_chunk_is_true(self, resolve_options, fake_registry):
        executor = _TestExecutor(pipeline_registry=fake_registry(None))
        request = OrchestrationRequest(
            text="x" * 101,
            targets=("defined_terms",),
            pipeline_options={
                "defined_terms": {
                    "chunking_enabled": True,
                    "chunk_threshold_chars": 100,
                }
            },
        )

        out = await executor.execute(
            request=request,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )

        assert out.payload == {"mode": "chunked"}
        assert len(executor.chunked_calls) == 1
        assert executor.chunked_calls[0]["request"] == request

    @pytest.mark.anyio
    async def test_execute_uses_direct_path_when_should_chunk_is_false(self, resolve_options, fake_registry):
        expected = PipelineRunResult(
            pipeline="defined_terms",
            payload={"mode": "direct"},
        )
        runner = _FakeRunner(expected)
        executor = _TestExecutor(pipeline_registry=fake_registry(runner))
        request = OrchestrationRequest(
            text="short text",
            targets=("defined_terms",),
            pipeline_options={
                "defined_terms": {
                    "chunking_enabled": True,
                    "chunk_threshold_chars": 100,
                }
            },
        )

        out = await executor.execute(
            request=request,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )

        assert out is expected
        assert executor.chunked_calls == []
        assert len(runner.calls) == 1
