import json
import logging
import pytest

from observability.observability.emit import emit
from observability.observability.levels import LogLevel, STD_LEVEL

def _last_json(caplog) -> tuple[dict, logging.LogRecord]:
    assert caplog.records, "no records captured"
    rec = caplog.records[-1]
    return json.loads(rec.msg), rec

@pytest.mark.parametrize("lvl", [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.MESSAGE])
def test_emit_level_payload_and_logrecord(caplog, lvl):
    caplog.set_level(logging.DEBUG)  # capture everything
    emit("probe", level=lvl, logger_type="test")
    payload, rec = _last_json(caplog)

    # payload fields
    assert payload["event"] == "probe"
    assert payload["level"] == lvl.name.lower()
    assert payload["logger_type"] == "test"

    # actual log record severity used by logging.log(...)
    assert rec.levelno == STD_LEVEL[lvl]

def test_decorator_defaults_to_info_level(caplog):
    caplog.set_level(logging.DEBUG)
    from observability.observability.decorator import logger

    @logger("work", arg_names=["x"])
    def work(x: int) -> int:
        return x * 2

    assert work(3) == 6
    payload, rec = _last_json(caplog)
    assert payload["event"] == "work"
    assert payload["level"] == "info"             # default in decorator
    assert rec.levelno == logging.INFO
