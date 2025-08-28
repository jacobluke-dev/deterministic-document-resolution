import json
import logging
import pytest

from plainera_observability.observability.decorator import logger
from plainera_observability.observability.context import request_id_var
from plainera_observability.observability.levels import LogLevel, STD_LEVEL


def _last_json(caplog) -> dict:
    assert caplog.records, "no logs captured"
    return json.loads(caplog.records[-1].msg)

def test_sync_decorator_emits_json(caplog):
    caplog.set_level(logging.INFO)

    @logger("compute", arg_names=["x", "y"], logger_type="service")
    def compute(x: int, y: int) -> int:
        return x + y

    request_id_var.set("rid-123")
    assert compute(2, 3) == 5

    p = _last_json(caplog)
    assert p["event"] == "compute"
    assert p["function"] == "compute"
    assert p["logger_type"] == "service"
    assert p["request_id"] == "rid-123"
    assert p["args"] == {"x": 2, "y": 3}
    assert isinstance(p["duration_ms"], int)

@pytest.mark.asyncio
async def test_async_decorator_emits_json(caplog):
    caplog.set_level(logging.INFO)

    @logger("fetch", arg_names=["q"], logger_type="service")
    async def fetch(q: str) -> str:
        return q.upper()

    request_id_var.set("rid-xyz")
    out = await fetch("abc")
    assert out == "ABC"

    p = _last_json(caplog)
    assert p["event"] == "fetch"
    assert p["function"] == "fetch"
    assert p["request_id"] == "rid-xyz"
    assert p["args"] == {"q": "abc"}
    assert isinstance(p["duration_ms"], int)

def test_redaction(caplog):
    caplog.set_level(logging.INFO)

    @logger("store", arg_names=["token", "data"], redact=["token"])
    def store(token: str, data: dict) -> None:
        pass

    store(token="secret123", data={"ok": 1, "Authorization": "Bearer abc"})
    p = _last_json(caplog)
    # explicit redact list applies
    assert p["args"]["token"] == "[REDACTED]"
    # central redactor should scrub nested sensitive keys (case-insensitive)
    assert p["args"]["data"]["Authorization"] == "[REDACTED]"

def test_db_sink_is_called(caplog):
    caplog.set_level(logging.INFO)

    class DummySink:
        def __init__(self): self.items = []
        # sync method on purpose (avoid coroutine-not-awaited warnings)
        def enqueue(self, payload): self.items.append(payload)

    sink = DummySink()

    @logger("persist", arg_names=["x"], db_sink=sink)
    def persist(x: int) -> None:
        pass

    persist(7)
    # still logs to stdout
    p = _last_json(caplog)
    assert p["event"] == "persist"
    # and we enqueued once to the sink
    assert len(sink.items) == 1
    assert sink.items[0]["event"] == "persist"


def test_std_level_mapping():
    assert STD_LEVEL[LogLevel.DEBUG]   == logging.DEBUG
    assert STD_LEVEL[LogLevel.INFO]    == logging.INFO
    assert STD_LEVEL[LogLevel.WARNING] == logging.WARNING
    assert STD_LEVEL[LogLevel.ERROR]   == logging.ERROR
    # MESSAGE is an alias of INFO
    assert STD_LEVEL[LogLevel.MESSAGE] == logging.INFO


def test_log_result_included_and_truncated(caplog):
    caplog.set_level(logging.INFO)
    from plainera_observability.observability.decorator import logger

    @logger("calc", arg_names=["x"], log_result=True, result_max_len=20)
    def calc(x): return {"ok": True, "data": "x"*100}

    calc(1)
    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "calc"
    assert payload["args"] == {"x": 1}
    assert payload["result"].endswith(" chars)")
