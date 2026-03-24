from plainera_unacronym.orchestration.adapters import (
    AcronymPipelineRunner,
    DefinedTermsPipelineRunner,
    StructuralReferencesPipelineRunner,
)
from plainera_unacronym.orchestration.registry import PipelineRegistry


def build_pipeline_registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register(AcronymPipelineRunner())
    registry.register(DefinedTermsPipelineRunner())
    registry.register(StructuralReferencesPipelineRunner())
    return registry
