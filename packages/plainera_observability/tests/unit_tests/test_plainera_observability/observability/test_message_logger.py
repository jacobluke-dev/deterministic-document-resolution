import json
import logging

from observability.observability.levels import LogLevel
from observability.observability.message_logger import message_logger


def _last_json(caplog):
    rec = caplog.records[-1]
    return json.loads(rec.getMessage()), rec

def test_message_logger_emits(caplog):
    caplog.set_level(logging.INFO, logger="plainera")
    message_logger("custom_event", level=LogLevel.WARNING, args={"authorization": "abc"})
    p, rec = _last_json(caplog)
    assert p["event"] == "custom_event"
    assert p["level"] == "warning"
    assert p["logger_type"] == "inline"
    assert p["args"]["authorization"] == "[REDACTED]"
    assert p["function"] == "test_message_logger_emits"
