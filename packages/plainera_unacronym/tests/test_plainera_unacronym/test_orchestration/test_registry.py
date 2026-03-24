from __future__ import annotations
import pytest

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineRequest,
    PipelineRunResult,
    PipelineRunner,
)
from plainera_unacronym.orchestration.registry import (
    DuplicatePipelineKeyError,
    PipelineRegistry,
    UnknownPipelineKeyError,
)


class _StubRunner(PipelineRunner):
    def __init__(self, key: str) -> None:
        self.key = key

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        return PipelineRunResult(
            pipeline=self.key,
            payload={"text": request.text},
        )


def test_pipeline_registry_register_and_get_returns_registered_runner() -> None:
    registry = PipelineRegistry()
    runner = _StubRunner(PIPELINE_ACRONYMS)

    registry.register(runner)

    assert registry.get(PIPELINE_ACRONYMS) is runner
    assert registry.keys() == (PIPELINE_ACRONYMS,)


def test_pipeline_registry_register_raises_for_duplicate_key() -> None:
    registry = PipelineRegistry()
    registry.register(_StubRunner(PIPELINE_ACRONYMS))

    with pytest.raises(DuplicatePipelineKeyError):
        registry.register(_StubRunner(PIPELINE_ACRONYMS))


def test_pipeline_registry_get_raises_for_unknown_key() -> None:
    registry = PipelineRegistry()

    with pytest.raises(UnknownPipelineKeyError):
        registry.get(PIPELINE_DEFINED_TERMS)


def test_pipeline_registry_resolve_returns_requested_runners_in_registry_order() -> None:
    registry = PipelineRegistry()
    registry.register(_StubRunner(PIPELINE_ACRONYMS))
    registry.register(_StubRunner(PIPELINE_DEFINED_TERMS))
    registry.register(_StubRunner(PIPELINE_STRUCTURAL_REFERENCES))

    resolved = registry.resolve(
        (PIPELINE_STRUCTURAL_REFERENCES, PIPELINE_ACRONYMS),
    )

    assert tuple(runner.key for runner in resolved) == (
        PIPELINE_ACRONYMS,
        PIPELINE_STRUCTURAL_REFERENCES,
    )


def test_pipeline_registry_resolve_returns_empty_tuple_for_no_targets() -> None:
    registry = PipelineRegistry()
    registry.register(_StubRunner(PIPELINE_ACRONYMS))

    resolved = registry.resolve(())

    assert resolved == ()


def test_pipeline_registry_resolve_raises_for_unknown_target() -> None:
    registry = PipelineRegistry()
    registry.register(_StubRunner(PIPELINE_ACRONYMS))

    with pytest.raises(UnknownPipelineKeyError):
        registry.resolve((PIPELINE_ACRONYMS, "not_real"))
