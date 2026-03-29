from __future__ import annotations

from types import SimpleNamespace

import pytest

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
)

from public_api.schemas.resolve import ResolveOptions, ResolveRequest

from public_api.core.orchestration.request_builder import (
    _base_pipeline_options,
    _build_acronym_options,
    _build_defined_term_options,
    _build_structural_reference_options,
    _default_chunking_options,
    build_orchestration_request,
    _chunking_options,
)


class TestChunkingOptions:
    def test_chunking_options_normalizes_values(self):
        out = _chunking_options(
            enabled=1,
            threshold_chars=500,
            chunk_size_chars=250,
            chunk_overlap_chars=25,
        )

        assert out == {
            "chunking_enabled": True,
            "chunk_threshold_chars": 500,
            "chunk_size_chars": 250,
            "chunk_overlap_chars": 25,
        }

    def test_default_chunking_options_reads_app_settings(self, _patch):
        _patch(
            _default_chunking_options,
            app_settings=SimpleNamespace(
                CHUNKING_ENABLED=True,
                CHUNK_THRESHOLD_CHARS=1000,
                CHUNK_SIZE_CHARS=400,
                CHUNK_OVERLAP_CHARS=40,
            ),
        )

        out = _default_chunking_options()

        assert out == {
            "chunking_enabled": True,
            "chunk_threshold_chars": 1000,
            "chunk_size_chars": 400,
            "chunk_overlap_chars": 40,
        }


class TestPipelineOptionBuilders:
    @pytest.fixture
    def resolve_options(self) -> ResolveOptions:
        return ResolveOptions(
            locale="en-GB",
            window_chars=240,
            max_definitions_per_acronym=5,
            include_glossary_enrichment=True,
            return_occurrences=True,
            min_confidence=0.0,
        )

    def test_base_pipeline_options_returns_expected_defaults(self):
        assert _base_pipeline_options() == {
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": False,
            "return_state": False,
            "trace": False,
            "trace_filter": None,
        }

    def test_build_acronym_options_uses_payload_options_and_tier2_model(
        self,
        resolve_options,
        _patch,
    ):
        _patch(
            _build_acronym_options,
            _default_chunking_options=lambda: {
                "chunking_enabled": True,
                "chunk_threshold_chars": 1000,
                "chunk_size_chars": 400,
                "chunk_overlap_chars": 40,
            },
        )

        payload = ResolveRequest(
            text="Alpha Beta Charlie (ABC).",
            options=resolve_options,
        )
        tier2_model = object()

        out = _build_acronym_options(payload, tier2_model=tier2_model)

        assert out == {
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": False,
            "return_state": False,
            "trace": False,
            "trace_filter": None,
            "window_left": 240,
            "window_right": 240,
            "tier2_model": tier2_model,
            "chunking_enabled": True,
            "chunk_threshold_chars": 1000,
            "chunk_size_chars": 400,
            "chunk_overlap_chars": 40,
        }

    def test_build_acronym_options_uses_default_resolve_options_when_missing(self, _patch):
        _patch(
            _build_acronym_options,
            _default_chunking_options=lambda: {
                "chunking_enabled": False,
                "chunk_threshold_chars": 800,
                "chunk_size_chars": 300,
                "chunk_overlap_chars": 30,
            },
        )

        payload = ResolveRequest(
            text="Alpha Beta Charlie (ABC).",
            options=None,
        )

        out = _build_acronym_options(payload, tier2_model=None)

        assert out["window_left"] == 120
        assert out["window_right"] == 120
        assert out["tier2_model"] is None
        assert out["chunking_enabled"] is False
        assert out["chunk_threshold_chars"] == 800
        assert out["chunk_size_chars"] == 300
        assert out["chunk_overlap_chars"] == 30

    def test_build_defined_term_options_uses_base_and_chunking(self, _patch):
        _patch(
            _build_defined_term_options,
            _default_chunking_options=lambda: {
                "chunking_enabled": True,
                "chunk_threshold_chars": 900,
                "chunk_size_chars": 350,
                "chunk_overlap_chars": 35,
            },
        )

        payload = ResolveRequest(text='"Services" means the services.')

        out = _build_defined_term_options(payload)

        assert out == {
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": False,
            "return_state": False,
            "trace": False,
            "trace_filter": None,
            "chunking_enabled": True,
            "chunk_threshold_chars": 900,
            "chunk_size_chars": 350,
            "chunk_overlap_chars": 35,
        }

    def test_build_structural_reference_options_uses_base_and_chunking(self, _patch):
        _patch(
            _build_structural_reference_options,
            _default_chunking_options=lambda: {
                "chunking_enabled": True,
                "chunk_threshold_chars": 700,
                "chunk_size_chars": 280,
                "chunk_overlap_chars": 28,
            },
        )

        payload = ResolveRequest(text="See Section 2 and Schedule 1.")

        out = _build_structural_reference_options(payload)

        assert out == {
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": False,
            "return_state": False,
            "trace": False,
            "trace_filter": None,
            "chunking_enabled": True,
            "chunk_threshold_chars": 700,
            "chunk_size_chars": 280,
            "chunk_overlap_chars": 28,
        }


class TestBuildOrchestrationRequest:
    def test_builds_request_for_selected_targets_only(self, _patch):
        _patch(
            build_orchestration_request,
            _build_acronym_options=lambda payload, *, tier2_model: {
                "kind": "acronyms",
                "tier2_model": tier2_model,
            },
            _build_defined_term_options=lambda payload: {
                "kind": "defined_terms",
            },
            _build_structural_reference_options=lambda payload: {
                "kind": "structural_references",
            },
        )

        payload = ResolveRequest(text="Example text.")
        tier2_model = object()

        out = build_orchestration_request(
            payload,
            targets=(PIPELINE_ACRONYMS, PIPELINE_STRUCTURAL_REFERENCES),
            tier2_model=tier2_model,
        )

        assert out.text == "Example text."
        assert out.targets == (PIPELINE_ACRONYMS, PIPELINE_STRUCTURAL_REFERENCES)
        assert out.pipeline_options == {
            PIPELINE_ACRONYMS: {
                "kind": "acronyms",
                "tier2_model": tier2_model,
            },
            PIPELINE_STRUCTURAL_REFERENCES: {
                "kind": "structural_references",
            },
        }

    def test_builds_request_for_all_supported_targets(self, _patch):
        _patch(
            build_orchestration_request,
            _build_acronym_options=lambda payload, *, tier2_model: {"kind": "acronyms"},
            _build_defined_term_options=lambda payload: {"kind": "defined_terms"},
            _build_structural_reference_options=lambda payload: {"kind": "structural_references"},
        )

        payload = ResolveRequest(text="Example text.")

        out = build_orchestration_request(
            payload,
            targets=(
                PIPELINE_ACRONYMS,
                PIPELINE_DEFINED_TERMS,
                PIPELINE_STRUCTURAL_REFERENCES,
            ),
            tier2_model=None,
        )

        assert out.pipeline_options == {
            PIPELINE_ACRONYMS: {"kind": "acronyms"},
            PIPELINE_DEFINED_TERMS: {"kind": "defined_terms"},
            PIPELINE_STRUCTURAL_REFERENCES: {"kind": "structural_references"},
        }

    def test_builds_request_with_empty_pipeline_options_when_no_targets(self):
        payload = ResolveRequest(text="Example text.")

        out = build_orchestration_request(
            payload,
            targets=(),
            tier2_model=None,
        )

        assert out.text == "Example text."
        assert out.targets == ()
        assert out.pipeline_options == {}
