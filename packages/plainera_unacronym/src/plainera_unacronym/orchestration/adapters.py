from __future__ import annotations

from dataclasses import dataclass

from plainera_unacronym.orchestration.interface import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineRequest,
    PipelineRunResult, PipelineKey,
)


@dataclass(frozen=True, slots=True)
class AcronymPipelineRunner:
    key: PipelineKey = PIPELINE_ACRONYMS

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        payload = ...  # call acronym top-level execute here
        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class DefinedTermsPipelineRunner:
    key: PipelineKey = PIPELINE_DEFINED_TERMS

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        payload = ...  # call defined-terms top-level execute here
        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class StructuralReferencesPipelineRunner:
    key: PipelineKey = PIPELINE_STRUCTURAL_REFERENCES

    def run(self, request: PipelineRequest) -> PipelineRunResult:
        payload = ...  # call structural top-level execute here
        return PipelineRunResult(
            pipeline=self.key,
            payload=payload,
        )
