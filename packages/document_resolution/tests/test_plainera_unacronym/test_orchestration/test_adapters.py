from __future__ import annotations

from document_resolution.orchestration.adapters import (
    AcronymPipelineRunner,
    DefinedTermsPipelineRunner,
    StructuralReferencesPipelineRunner,
)
from document_resolution.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineRequest,
)


def test_acronym_pipeline_runner_run_passes_expected_kwargs(_patch) -> None:
    calls: list[dict[str, object]] = []

    def fake_detect_and_extract(text: str, **kwargs: object) -> tuple[str, str]:
        calls.append({"text": text, **kwargs})
        return "detector", "extraction"

    _patch(AcronymPipelineRunner.run, detect_and_extract=fake_detect_and_extract)

    runner = AcronymPipelineRunner()
    result = runner.run(
        PipelineRequest(
            text="Example text",
            options={
                "det_cfg": "det-cfg",
                "ext_cfg": "ext-cfg",
                "tier2_model": "tier2",
                "window_left": 400,
                "window_right": 260,
                "return_reports": True,
                "trace": True,
                "return_state": True,
                "trace_filter": r"^(API)$",
            },
        )
    )

    assert result.pipeline == PIPELINE_ACRONYMS
    assert result.payload == ("detector", "extraction")
    assert calls == [
        {
            "text": "Example text",
            "det_cfg": None,
            "ext_cfg": None,
            "tier2_model": "tier2",
            "window_left": 400,
            "window_right": 260,
            "return_reports": True,
            "trace": True,
            "return_state": True,
            "trace_filter": r"^(API)$",
        }
    ]


def test_acronym_pipeline_runner_run_uses_default_windows_for_invalid_values(
    _patch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_detect_and_extract(text: str, **kwargs: object) -> tuple[str, str]:
        calls.append({"text": text, **kwargs})
        return "detector", "extraction"

    _patch(AcronymPipelineRunner.run, detect_and_extract=fake_detect_and_extract)

    runner = AcronymPipelineRunner()
    runner.run(
        PipelineRequest(
            text="Example text",
            options={
                "window_left": "bad",
                "window_right": None,
            },
        )
    )

    assert calls == [
        {
            "text": "Example text",
            "det_cfg": None,
            "ext_cfg": None,
            "tier2_model": None,
            "window_left": 320,
            "window_right": 280,
            "return_reports": False,
            "trace": False,
            "return_state": False,
            "trace_filter": None,
        }
    ]


def test_defined_terms_pipeline_runner_run_passes_expected_kwargs(_patch) -> None:
    calls: list[dict[str, object]] = []

    def fake_detect_and_resolve_terms(
        text: str,
        **kwargs: object,
    ) -> tuple[str, str]:
        calls.append({"text": text, **kwargs})
        return "detector", "resolution"

    _patch(
        DefinedTermsPipelineRunner.run,
        detect_and_resolve_terms=fake_detect_and_resolve_terms,
    )

    runner = DefinedTermsPipelineRunner()
    result = runner.run(
        PipelineRequest(
            text="Defined term text",
            options={
                "det_cfg": "det-cfg",
                "ext_cfg": "ext-cfg",
                "return_reports": True,
                "trace": True,
                "return_state": True,
                "trace_filter": r"^(Agreement)$",
            },
        )
    )

    assert result.pipeline == PIPELINE_DEFINED_TERMS
    assert result.payload == ("detector", "resolution")
    assert calls == [
        {
            "text": "Defined term text",
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": True,
            "trace": True,
            "return_state": True,
            "trace_filter": r"^(Agreement)$",
        }
    ]


def test_structural_references_pipeline_runner_run_passes_expected_kwargs(
    _patch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_detect_and_resolve_structural_references(
        text: str,
        **kwargs: object,
    ) -> tuple[str, str]:
        calls.append({"text": text, **kwargs})
        return "detector", "resolution"

    _patch(
        StructuralReferencesPipelineRunner.run,
        detect_and_resolve_structural_references=(
            fake_detect_and_resolve_structural_references
        ),
    )

    runner = StructuralReferencesPipelineRunner()
    result = runner.run(
        PipelineRequest(
            text="Structural reference text",
            options={
                "det_cfg": "det-cfg",
                "ext_cfg": "ext-cfg",
                "return_reports": True,
                "return_state": True,
            },
        )
    )

    assert result.pipeline == PIPELINE_STRUCTURAL_REFERENCES
    assert result.payload == ("detector", "resolution")
    assert calls == [
        {
            "text": "Structural reference text",
            "det_cfg": None,
            "ext_cfg": None,
            "return_reports": True,
            "return_state": True,
        }
    ]

def test_acronym_pipeline_runner_run_discards_invalid_config_types(_patch) -> None:
    calls: list[dict[str, object]] = []

    def fake_detect_and_extract(text: str, **kwargs: object) -> tuple[str, str]:
        calls.append({"text": text, **kwargs})
        return "detector", "extraction"

    _patch(AcronymPipelineRunner.run, detect_and_extract=fake_detect_and_extract)

    runner = AcronymPipelineRunner()
    runner.run(
        PipelineRequest(
            text="Example text",
            options={
                "det_cfg": "bad",
                "ext_cfg": 123,
            },
        )
    )

    assert calls == [
        {
            "text": "Example text",
            "det_cfg": None,
            "ext_cfg": None,
            "tier2_model": None,
            "window_left": 320,
            "window_right": 280,
            "return_reports": False,
            "trace": False,
            "return_state": False,
            "trace_filter": None,
        }
    ]
