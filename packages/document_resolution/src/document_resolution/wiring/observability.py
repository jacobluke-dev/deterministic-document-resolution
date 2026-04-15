from document_resolution.db.models.logger import PackageLogger
from document_resolution_core.db_manager.sink_factory import build_sink_from_env, register_sink

register_sink("package_logger", PackageLogger, default_logger_type="package")
sink = build_sink_from_env("package_logger")
