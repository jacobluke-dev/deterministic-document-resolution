from plainera_core.db_manager.sink_bootstrap import build_sink_from_env
from plainera_core.db_manager.sink_factory import register_sink
from public_api.db.models import Logger

register_sink("api_logger", Logger, default_logger_type="api")
sink = build_sink_from_env("api_logger")
