import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect as sqla_inspect

from document_resolution_core.db_manager.config import MapperFn

_LEVEL_NAME_TO_CODE = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}
_CODE_TO_LEVEL_NAME = {v: k for k, v in _LEVEL_NAME_TO_CODE.items()}


def _level_name(raw: Any) -> str:
    """Normalize a logging level to a lowercase name.

    Args:
        raw: An integer code (e.g., 10, 20) or a string name
             (e.g., ``"DEBUG"``, ``"warning"``), or any object convertible to string.

    Returns:
        str: One of the known level names (e.g., ``"debug"``, ``"info"``,
             ``"warning"``, ``"error"``, ``"critical"``). Defaults to ``"info"`` if unrecognized.
    """
    if isinstance(raw, int):
        return _CODE_TO_LEVEL_NAME.get(raw, "info")
    s = str(raw or "info").lower()
    return s if s in _LEVEL_NAME_TO_CODE else "info"


def _parse_ts(val: Any) -> datetime:
    """Parse a timestamp-like value and return a timezone-aware UTC
    ``datetime``.

    Args:
        ts: ISO-8601 string (e.g., ``"2025-09-17T12:34:56Z"`` or with an offset),
            a ``datetime`` (naive or aware), or any other value.

    Returns:
        datetime: A timezone-aware UTC ``datetime`` for strings and fallback;
        for aware ``datetime`` inputs, the original object is returned unchanged.
    """
    # already a datetime?
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)

    # string?
    if isinstance(val, str):
        s = val.strip()
        # Reject bare dates: default to now()
        if "T" not in s:
            return datetime.now(timezone.utc)
        try:
            # Normalize 'Z' to '+00:00' for fromisoformat
            s_norm = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s_norm)
        except Exception:
            return datetime.now(timezone.utc)
        # Make UTC-aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # everything else → now()
    return datetime.now(timezone.utc)


def _sanitize_status(x: Any) -> int | None:
    """Normalize a value to a valid HTTP status code or return ``None``.

    Accepts ints, numeric strings, and whole-number floats (e.g., ``200.0``)
    that fall within the HTTP status range ``100..599`` (inclusive). Rejects
    booleans, non-finite floats (``nan``, ``inf``), and non-integer floats
    (e.g., ``200.1``). Any parsing/validation failure results in ``None``.

    Args:
        x: Candidate status value. May be an ``int``, ``float``, string, etc.

    Returns:
        ``int``: The validated status code in the range ``100..599``.
        ``None``: If the value cannot be interpreted as a valid status.

    Notes:
        * Booleans are explicitly rejected (even though ``bool`` is a subclass of ``int``).
        * Floats must be finite and represent an integer (``x.is_integer()``).
        * Strings are parsed via ``int(x)``; non-numeric strings are rejected.
    """
    try:
        if isinstance(x, bool):
            return None  # avoid True/False (1/0)
        if isinstance(x, float) and (not math.isfinite(x) or not x.is_integer()):
                return None
        v = int(x)
        return v if 100 <= v <= 599 else None
    except Exception:
        return None


def make_logger_mapper(
    model: type[Any],
    *,
    default_logger_type: str = "decorator",
) -> MapperFn:
    """The returned callable accepts a logging payload (dict) and produces a
    dict of column values filtered to the columns declared on ``model``. It
    normalizes level information (``level_name``/``level_code``), chooses a
    final ``info`` field (prefers ``result`` then ``info`` then ``event``),
    converts the timestamp to a timezone-aware UTC ``date_time``, and includes
    optional HTTP-style fields only if the model defines them.

    Args:
        model: A SQLAlchemy mapped class whose columns are discovered via
            ``sqlalchemy.inspection.inspect(model).columns``.
        default_logger_type: Fallback value for the ``logger_type`` column when
            the payload does not provide one. Defaults to ``"decorator"``.

    Returns:
        MapperFn: A function ``map_(payload: dict[str, Any]) -> dict[str, Any]``
        that produces a row ready for insertion (e.g., ``insert(model).values(**row)``).

    Notes:
        - Unknown payload keys are dropped.
        - Only columns present on ``model`` are returned in the mapped dict.
    """
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
