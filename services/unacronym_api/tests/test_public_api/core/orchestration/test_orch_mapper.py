from __future__ import annotations

from types import SimpleNamespace

import pytest

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineRunResult,
)
from plainera_unacronym.orchestration.state import (
    OrchestrationPipelineError,
    OrchestrationState,
    PipelineErrorCode,
)
from public_api.core.orchestration.mapper import _resolve_pipeline_payload, map_orchestration_state, compose_sections

from public_api.schemas.resolve import ResolveOptions, ResolutionMode


class _DummyResult:
    pass


class TestResolvePipelinePayload:
    def test_returns_list_payload_as_is(self):
        payload = [{"x": 1}]

        out = _resolve_pipeline_payload(
            payload,
            result_type=_DummyResult,
            error_message="bad payload",
        )

        assert out is payload

    def test_returns_direct_result_payload(self):
        payload = _DummyResult()

        out = _resolve_pipeline_payload(
            payload,
            result_type=_DummyResult,
            error_message="bad payload",
        )

        assert out is payload

    def test_returns_resolution_from_tuple_payload(self):
        resolution = _DummyResult()
        payload = ("detector", resolution)

        out = _resolve_pipeline_payload(
            payload,
            result_type=_DummyResult,
            error_message="bad payload",
        )

        assert out is resolution

    def test_raises_for_tuple_with_wrong_resolution_type(self):
        payload = ("detector", object())

        with pytest.raises(ValueError, match="bad payload"):
            _resolve_pipeline_payload(
                payload,
                result_type=_DummyResult,
                error_message="bad payload",
            )

    def test_raises_for_unsupported_payload_shape(self):
        with pytest.raises(ValueError, match="bad payload"):
            _resolve_pipeline_payload(
                object(),
                result_type=_DummyResult,
                error_message="bad payload",
            )


class TestMapOrchestrationState:
    def test_maps_meta_and_errors_in_failed_target_order(self):
        state = OrchestrationState.from_requested_targets(
            (
                PIPELINE_ACRONYMS,
                PIPELINE_DEFINED_TERMS,
                PIPELINE_STRUCTURAL_REFERENCES,
            )
        )
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_ACRONYMS,
                payload=[],
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline=PIPELINE_STRUCTURAL_REFERENCES,
                code=PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                message="structural boom",
                error_type="RuntimeError",
                details={},
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline=PIPELINE_DEFINED_TERMS,
                code=PipelineErrorCode.PIPELINE_INVALID_OPTIONS,
                message="bad options",
                error_type="ValueError",
                details={},
            )
        )

        meta, errors = map_orchestration_state(state)

        assert meta.requested == [
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
        ]
        assert meta.completed == [PIPELINE_ACRONYMS]
        assert meta.failed == [
            PIPELINE_STRUCTURAL_REFERENCES,
            PIPELINE_DEFINED_TERMS,
        ]
        assert [(e.pipeline, e.code, e.message) for e in errors] == [
            (
                PIPELINE_STRUCTURAL_REFERENCES,
                PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                "structural boom",
            ),
            (
                PIPELINE_DEFINED_TERMS,
                PipelineErrorCode.PIPELINE_INVALID_OPTIONS,
                "bad options",
            ),
        ]


class TestComposeSections:
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

    def test_returns_empty_sections_when_no_pipelines_completed(self, resolve_options):
        state = OrchestrationState.from_requested_targets(())

        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=SimpleNamespace(),
        )

        assert out == {
            "acronyms": [],
            "defined_terms": [],
            "structural_references": [],
        }

    def test_passes_through_prebuilt_acronym_blocks(self, resolve_options):
        state = OrchestrationState.from_requested_targets((PIPELINE_ACRONYMS,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_ACRONYMS,
                payload=[{"acronym": "MPS"}],
            )
        )

        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=SimpleNamespace(),
        )

        assert out["acronyms"] == [{"acronym": "MPS"}]
        assert out["defined_terms"] == []
        assert out["structural_references"] == []

    def test_maps_acronym_tuple_payload_and_attaches_metadata(self, resolve_options, _patch):
        state = OrchestrationState.from_requested_targets((PIPELINE_ACRONYMS,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_ACRONYMS,
                payload=("detector_result", "extraction_result"),
            )
        )

        calls: dict[str, object] = {}

        def fake_map_acronym_pipeline_to_blocks(*, det_res, extr, opts, lang, glossary_repo):
            calls["map"] = {
                "det_res": det_res,
                "extr": extr,
                "opts": opts,
                "lang": lang,
                "glossary_repo": glossary_repo,
            }
            return [{"acronym": "MPS"}]

        def fake_attach_resolution_metadata(*, blocks, opts, resolution_mode, glossary_repo):
            calls["attach"] = {
                "blocks": blocks,
                "opts": opts,
                "resolution_mode": resolution_mode,
                "glossary_repo": glossary_repo,
            }
            return [{"acronym": "MPS", "selected": {"definition": "Metropolitan Police Service"}}]

        _patch(
            compose_sections,
            map_acronym_pipeline_to_blocks=fake_map_acronym_pipeline_to_blocks,
            attach_resolution_metadata=fake_attach_resolution_metadata,
        )

        glossary_repo = SimpleNamespace()
        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=glossary_repo,
        )

        assert out["acronyms"] == [
            {"acronym": "MPS", "selected": {"definition": "Metropolitan Police Service"}}
        ]
        assert calls["map"] == {
            "det_res": "detector_result",
            "extr": "extraction_result",
            "opts": resolve_options,
            "lang": "en",
            "glossary_repo": glossary_repo,
        }
        assert calls["attach"] == {
            "blocks": [{"acronym": "MPS"}],
            "opts": resolve_options,
            "resolution_mode": ResolutionMode.DOMAIN_PRIORITY,
            "glossary_repo": glossary_repo,
        }

    def test_maps_defined_term_resolution_payload(self, resolve_options, _patch):
        state = OrchestrationState.from_requested_targets((PIPELINE_DEFINED_TERMS,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_DEFINED_TERMS,
                payload="term_result",
            )
        )

        def fake_resolve_pipeline_payload(payload, *, result_type, error_message):
            assert payload == "term_result"
            return "resolved_term_result"

        def fake_map_defined_term_blocks(resolved):
            assert resolved == "resolved_term_result"
            return [{"term": "Services"}]

        _patch(
            compose_sections,
            _resolve_pipeline_payload=fake_resolve_pipeline_payload,
            map_defined_term_blocks=fake_map_defined_term_blocks,
        )

        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=SimpleNamespace(),
        )

        assert out["defined_terms"] == [{"term": "Services"}]
        assert out["acronyms"] == []
        assert out["structural_references"] == []

    def test_passes_through_prebuilt_defined_term_blocks(self, resolve_options, _patch):
        state = OrchestrationState.from_requested_targets((PIPELINE_DEFINED_TERMS,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_DEFINED_TERMS,
                payload="term_result",
            )
        )

        def fake_resolve_pipeline_payload(payload, *, result_type, error_message):
            return [{"term": "Services"}]

        _patch(
            compose_sections,
            _resolve_pipeline_payload=fake_resolve_pipeline_payload,
        )

        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=SimpleNamespace(),
        )

        assert out["defined_terms"] == [{"term": "Services"}]

    def test_maps_structural_resolution_payload(self, resolve_options, _patch):
        state = OrchestrationState.from_requested_targets((PIPELINE_STRUCTURAL_REFERENCES,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_STRUCTURAL_REFERENCES,
                payload="structural_result",
            )
        )

        def fake_resolve_pipeline_payload(payload, *, result_type, error_message):
            assert payload == "structural_result"
            return "resolved_structural_result"

        def fake_map_structural_blocks(resolved):
            assert resolved == "resolved_structural_result"
            return [{"kind": "Section", "label": "2"}]

        _patch(
            compose_sections,
            _resolve_pipeline_payload=fake_resolve_pipeline_payload,
            map_structural_blocks=fake_map_structural_blocks,
        )

        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=SimpleNamespace(),
        )

        assert out["structural_references"] == [{"kind": "Section", "label": "2"}]
        assert out["acronyms"] == []
        assert out["defined_terms"] == []

    def test_passes_through_prebuilt_structural_blocks(self, resolve_options, _patch):
        state = OrchestrationState.from_requested_targets((PIPELINE_STRUCTURAL_REFERENCES,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_STRUCTURAL_REFERENCES,
                payload="structural_result",
            )
        )

        def fake_resolve_pipeline_payload(payload, *, result_type, error_message):
            return [{"kind": "Schedule", "label": "1"}]

        _patch(
            compose_sections,
            _resolve_pipeline_payload=fake_resolve_pipeline_payload,
        )

        out = compose_sections(
            state,
            opts=resolve_options,
            lang="en",
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
            glossary_repo=SimpleNamespace(),
        )

        assert out["structural_references"] == [{"kind": "Schedule", "label": "1"}]

    def test_raises_for_unsupported_acronym_payload_shape(self, resolve_options):
        state = OrchestrationState.from_requested_targets((PIPELINE_ACRONYMS,))
        state.record_success(
            PipelineRunResult(
                pipeline=PIPELINE_ACRONYMS,
                payload="bad payload",
            )
        )

        with pytest.raises(ValueError, match="Unsupported acronym pipeline payload shape."):
            compose_sections(
                state,
                opts=resolve_options,
                lang="en",
                resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
                glossary_repo=SimpleNamespace(),
            )


    def test_maps_all_success_metadata_and_no_errors(self) -> None:
        state = OrchestrationState.from_requested_targets(("acronyms", "defined_terms"))
        state.record_success(PipelineRunResult(pipeline="acronyms", payload={}, metadata={}))
        state.record_success(PipelineRunResult(pipeline="defined_terms", payload={}, metadata={}))
        state.finish()

        meta, errors = map_orchestration_state(state)

        assert meta.requested == ["acronyms", "defined_terms"]
        assert meta.completed == ["acronyms", "defined_terms"]
        assert meta.failed == []
        assert errors == []

    def test_maps_partial_success_metadata_and_errors_in_failed_order(self) -> None:
        state = OrchestrationState.from_requested_targets(
            ("acronyms", "defined_terms", "structural_references")
        )
        state.record_success(PipelineRunResult(pipeline="acronyms", payload={}, metadata={}))
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="defined_terms",
                code=PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                message="boom",
                error_type="RuntimeError",
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="structural_references",
                code=PipelineErrorCode.PIPELINE_TIMEOUT,
                message="timed out",
                error_type="TimeoutError",
            )
        )
        state.finish()

        meta, errors = map_orchestration_state(state)

        assert meta.requested == ["acronyms", "defined_terms", "structural_references"]
        assert meta.completed == ["acronyms"]
        assert meta.failed == ["defined_terms", "structural_references"]
        assert [error.pipeline for error in errors] == [
            "defined_terms",
            "structural_references",
        ]
        assert [error.code for error in errors] == [
            PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
            PipelineErrorCode.PIPELINE_TIMEOUT,
        ]

    def test_maps_all_failure_metadata(self) -> None:
        state = OrchestrationState.from_requested_targets(("acronyms", "defined_terms"))
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="acronyms",
                code=PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
                message="boom",
                error_type="RuntimeError",
            )
        )
        state.record_failure(
            OrchestrationPipelineError(
                pipeline="defined_terms",
                code=PipelineErrorCode.PIPELINE_INVALID_OPTIONS,
                message="bad config",
                error_type="ValueError",
            )
        )
        state.finish()

        meta, errors = map_orchestration_state(state)

        assert meta.requested == ["acronyms", "defined_terms"]
        assert meta.completed == []
        assert meta.failed == ["acronyms", "defined_terms"]
        assert [error.code for error in errors] == [
            PipelineErrorCode.PIPELINE_EXECUTION_FAILED,
            PipelineErrorCode.PIPELINE_INVALID_OPTIONS,
        ]
