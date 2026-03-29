from plainera_unacronym.orchestration.adapters import (
    AcronymPipelineRunner,
    DefinedTermsPipelineRunner,
    StructuralReferencesPipelineRunner,
)
from plainera_unacronym.orchestration.registry import PipelineRegistry


def build_pipeline_registry() -> PipelineRegistry:
    """Build the default orchestration pipeline registry.

        Registers the plainera_unacronym pipeline runners used by the API-facing
        orchestration flow in deterministic execution order.

        Returns:
            A ``PipelineRegistry`` populated with the acronym, defined-terms, and
            structural-reference pipeline runners.
    """
    registry = PipelineRegistry()
    registry.register(AcronymPipelineRunner())
    registry.register(DefinedTermsPipelineRunner())
    registry.register(StructuralReferencesPipelineRunner())
    return registry
