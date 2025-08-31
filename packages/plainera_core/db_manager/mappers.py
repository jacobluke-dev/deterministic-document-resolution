from typing import Any

KNOWN = {
    "timestamp","level","event","logger_type","request_id",
    "function","args","duration_ms","result","error",
}

def default_map(payload: dict[str, Any]) -> dict[str, Any]:
    row = {k: payload.get(k) for k in KNOWN}
    row["extra"] = {k: v for k, v in payload.items() if k not in KNOWN}
    return row
