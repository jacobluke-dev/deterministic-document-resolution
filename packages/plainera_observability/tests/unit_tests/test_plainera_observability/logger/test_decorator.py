import json
import logging

from observability.logger.context import request_id_var
from observability.logger.decorator import logger
from observability.logger.levels import STD_LEVEL, LogLevel


def _last_json(caplog) -> dict:
    assert caplog.records, "no logs captured"
    return json.loads(caplog.records[-1].msg)

def test_sync_decorator_emits_payload(capture_sink):
    sink = capture_sink()

    @logger("compute", arg_names=["x", "y"], logger_type="service", db_sink=sink)
    def compute(x: int, y: int) -> int:
        return x + y

    request_id_var.set("rid-123")
    assert compute(2, 3) == 5

    assert sink.items, "no payload captured"
    p = sink.items[-1]

    assert p["event"] == "compute"
    assert p["function"] == "compute"
    assert p["logger_type"] == "service"
    assert p["request_id"] == "rid-123"
    assert p["args"] == {"x": 2, "y": 3}
    assert isinstance(p["duration_ms"], int)


def test_redaction(capture_sink):
    sink = capture_sink()

    @logger("store", arg_names=["token", "data"], redact=["token"], db_sink=sink)
    def store(token: str, data: dict) -> None:
        pass

    store(token="secret123", data={"ok": 1, "Authorization": "Bearer abc"})

    p = sink.items[-1]
    assert p["args"]["token"] == "[REDACTED]"
    assert p["args"]["data"]["Authorization"] == "[REDACTED]"


def test_db_sink_is_called(capture_sink):
    sink = capture_sink()

    @logger("persist", arg_names=["x"], db_sink=sink)
    def persist(x: int) -> None:
        pass

    persist(7)

    # sink got exactly one payload
    assert len(sink.items) == 1
    p = sink.items[0]

    assert p["event"] == "persist"
    assert p["function"] == "persist"
    assert p["args"] == {"x": 7}
    assert isinstance(p["duration_ms"], int)


def test_std_level_mapping():
    assert STD_LEVEL[LogLevel.DEBUG]   == logging.DEBUG
    assert STD_LEVEL[LogLevel.INFO]    == logging.INFO
    assert STD_LEVEL[LogLevel.WARNING] == logging.WARNING
    assert STD_LEVEL[LogLevel.ERROR]   == logging.ERROR
    # MESSAGE is an alias of INFO
    assert STD_LEVEL[LogLevel.MESSAGE] == logging.INFO


def test_log_result_included_and_truncated(caplog):
    caplog.set_level(logging.INFO, logger="plainera")
    from observability.logger.decorator import logger

    @logger("calc", arg_names=["x"], log_result=True, result_max_len=20)
    def calc(x): return {"ok": True, "data": "x"*100}

    calc(1)
    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "calc"
    assert payload["args"] == {"x": 1}
    assert payload["result"].endswith(" chars)")
