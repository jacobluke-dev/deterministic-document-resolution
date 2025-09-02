from datetime import datetime, timezone
from typing import Any

_LEVEL_NAME_TO_CODE = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

def logger_model_map(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Map a raw logging payload into a normalized database model format.
    """
    level_name: str = str(payload.get("level", "info")).lower()
    level_code = _LEVEL_NAME_TO_CODE.get(level_name, 20)

    # timestamp from emit is ISO string; ensure tz-aware datetime
    ts_raw = payload.get("timestamp")
    date_time = (
        datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else datetime.now(timezone.utc)
    )

    return {
        "level_code": level_code,
        "level_name": level_name,
        "event": payload.get("event", ""),
        "logger_type": payload.get("logger_type", "decorator"),
        "function_name": payload.get("function"),
        "request_id": payload.get("request_id"),
        "duration_ms": payload.get("duration_ms"),

        # free-form summary; fall back to event name
        "info": payload.get("result") or payload.get("info") or payload.get("event"),

        # args your decorator already attaches; kw args only if you add them
        "arguments": payload.get("args"),
        "keyword_arguments": payload.get("kwargs"),

        # http-ish extras if present
        "path": payload.get("path"),
        "method": payload.get("method"),
        "status": payload.get("status"),
        "bytes": payload.get("bytes"),
        "client_ip": payload.get("client_ip"),
        "key_id": payload.get("key_id"),
        "date_time": date_time,
    }
