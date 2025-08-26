from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

SENSITIVE_HEADERS = {"authorization", "x-api-key"}

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key in {"msg", "args", "levelname", "name", "exc_text", "exc_info", "stack_info"}:
                continue
            if key.startswith("_cache"):
                continue
            payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger()
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return logger
