from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect as sqla_inspect

from plainera_core.db_manager.config import MapperFn

_LEVEL_NAME_TO_CODE = {"debug":10, "info":20, "warning":30, "error":40, "critical":50}
_CODE_TO_LEVEL_NAME = {v:k for k,v in _LEVEL_NAME_TO_CODE.items()}

def _level_name(raw: Any) -> str:
    if isinstance(raw, int):
        return _CODE_TO_LEVEL_NAME.get(raw, "info")
    s = str(raw or "info").lower()
    return s if s in _LEVEL_NAME_TO_CODE else "info"

def _parse_ts(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _sanitize_status(x: Any) -> int | None:
    try:
        v = int(x)
        return v if 100 <= v <= 599 else None
    except Exception:
        return None


def make_logger_mapper(
    model: type[Any],             # or: type[Base] if you can import your Base
    *,
    default_logger_type: str = "decorator",
) -> MapperFn:
    # Collect column names from the mapped class
    cols = {c.key for c in sqla_inspect(model).columns}

    def map_(payload: dict[str, Any]) -> dict[str, Any]:
        lvl_name = _level_name(payload.get("level"))
        out = {
            "level_code": _LEVEL_NAME_TO_CODE[lvl_name],
            "level_name": lvl_name,
            "event": payload.get("event", ""),
            "logger_type": payload.get("logger_type", default_logger_type),
            "function_name": payload.get("function"),
            "request_id": payload.get("request_id"),
            "duration_ms": payload.get("duration_ms"),
            "info": payload.get("result") or payload.get("info") or payload.get("event"),
            "arguments": payload.get("args"),
            "keyword_arguments": payload.get("kwargs"),
            "date_time": _parse_ts(payload.get("timestamp")),
        }
        # not all loggers have these cols
        for key in {"path", "method", "bytes", "client_ip", "key_id"} & cols:
            out[key] = payload.get(key)
        return {k: v for k, v in out.items() if k in cols}
    return map_
