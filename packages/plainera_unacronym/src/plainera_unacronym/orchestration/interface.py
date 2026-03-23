from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

type PipelineKey = str

PIPELINE_ACRONYMS: Final[PipelineKey] = "acronyms"
PIPELINE_DEFINED_TERMS: Final[PipelineKey] = "defined_terms"
PIPELINE_STRUCTURAL_REFERENCES: Final[PipelineKey] = "structural_references"


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Opaque request passed into a top-level pipeline runner."""

    text: str
    config: object | None = None


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Opaque result returned by a top-level pipeline runner."""

    pipeline: PipelineKey
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)


class PipelineRunner(Protocol):
    """Protocol for orchestration-layer pipeline execution."""

    @property
    def key(self) -> PipelineKey:
        """Stable pipeline key."""

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        """Execute a single top-level pipeline."""
