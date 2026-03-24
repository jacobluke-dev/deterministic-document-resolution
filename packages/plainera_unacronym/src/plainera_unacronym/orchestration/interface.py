from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

type PipelineKey = str

PIPELINE_ACRONYMS: Final[PipelineKey] = "acronyms"
PIPELINE_DEFINED_TERMS: Final[PipelineKey] = "defined_terms"
PIPELINE_STRUCTURAL_REFERENCES: Final[PipelineKey] = "structural_references"


@dataclass(frozen=True, slots=True)
class OrchestrationRequest:
    """Top-level orchestration input."""

    text: str
    targets: tuple[PipelineKey, ...]
    pipeline_options: Mapping[PipelineKey, Mapping[str, object]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Single pipeline request derived from orchestration input."""

    text: str
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Opaque top-level pipeline result."""

    pipeline: PipelineKey
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)


class PipelineRunner(ABC):
    """Abstract base class for a top-level pipeline runner."""

    key: PipelineKey

    @abstractmethod
    def run(self, request: PipelineRequest) -> PipelineRunResult:
        """Execute the pipeline for a single request."""
