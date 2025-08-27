import json, logging
from datetime import datetime, timezone
from typing import Any, Protocol, Optional
from .levels import LogLevel, STD_LEVEL
from .context import request_id_var
from .redact import scrub

logger = logging.getLogger("plainera")

class DBSink(Protocol):
    async def enqueue(self, payload: dict[str, Any]) -> None: ...

def emit(event: str,
         level: LogLevel = LogLevel.INFO,
         logger_type: str = "decorator",
         db_sink: Optional[DBSink] = None,
         **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.name.lower(),
        "event": event,
        "logger_type": logger_type,
        "request_id": request_id_var.get(),
        **scrub(fields),
    }
    logging.log(STD_LEVEL[level], json.dumps(payload, default=str))
    # Fire-and-forget enqueue if provided
    if db_sink:
        try:
            # do not await here; caller decides concurrency model
            _ = db_sink.enqueue(payload)
        except Exception:  # never block or crash on sink errors
            logging.log(STD_LEVEL[LogLevel.ERROR], json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "error",
                "event": "db_sink_enqueue_failed",
                "logger_type": "system",
                "request_id": request_id_var.get(),
            }))
