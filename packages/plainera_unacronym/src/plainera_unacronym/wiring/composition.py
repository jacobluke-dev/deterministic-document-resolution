from plainera_core.db_manager.sink_bootstrap import build_sink_from_env
from plainera_core.db_manager.sink_factory import register_sink
from plainera_unacronym.db.models.logger import PackageLogger
from plainera_unacronym.orchestration.adapters import (
    AcronymPipelineRunner,
    DefinedTermsPipelineRunner,
    StructuralReferencesPipelineRunner,
)
from plainera_unacronym.orchestration.registry import PipelineRegistry

register_sink("package_logger", PackageLogger, default_logger_type="package")
sink = build_sink_from_env("package_logger")




def build_pipeline_registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register(AcronymPipelineRunner())
    registry.register(DefinedTermsPipelineRunner())
    registry.register(StructuralReferencesPipelineRunner())
    return registry
