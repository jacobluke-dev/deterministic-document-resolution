import json
import logging

import pytest
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
    caplog.set_level(logging.INFO, logger="document_resolution")
    from observability.logger.decorator import logger

    @logger("calc", arg_names=["x"], log_result=True, result_max_len=20)
    def calc(x): return {"ok": True, "data": "x"*100}

    calc(1)
    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "calc"
    assert payload["args"] == {"x": 1}
    assert payload["result"].endswith(" chars)")


class _CaptureSink:
    def __init__(self):
        self.items = []

    def enqueue(self, payload):
        self.items.append(payload)


def _last(caplog) -> dict:
    assert caplog.records
    return json.loads(caplog.records[-1].msg)


def test_log_before_emits_pre_call_then_success(caplog):
    caplog.set_level(logging.INFO, logger="document_resolution")

    @logger("do", arg_names=["x"], log_before=True)
    def do(x: int) -> int:
        return x + 1

    assert do(1) == 2

    # Should have two logs: pre-call, then success finalise
    assert len(caplog.records) >= 2
    pre = json.loads(caplog.records[-2].msg)
    post = json.loads(caplog.records[-1].msg)

    assert pre["event"] in {"do", "Executing function"}
    assert post["event"] == "do"
    assert post["args"] == {"x": 1}


def test_sync_exception_path_emits_error_and_reraises(caplog):
    caplog.set_level(logging.INFO, logger="document_resolution")

    @logger("kaboom", arg_names=["x"])
    def kaboom(x: int) -> int:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        kaboom(3)

    payload = _last(caplog)
    assert payload["event"].startswith("Exception in ")
    assert payload["function"] == "kaboom"
    assert payload["args"] == {"x": 3}
    assert "ValueError" in payload["error"]


def test_result_transform_failure_is_captured(caplog):
    caplog.set_level(logging.INFO, logger="document_resolution")

    def bad_transform(_):
        raise RuntimeError("transform broke")

    @logger("calc", log_result=True, result_transform=bad_transform)
    def calc():
        return {"ok": True}

    calc()
    payload = _last(caplog)
    assert payload["event"] == "calc"
    assert payload["result"] == '"<result_transform_failed>"'


def test_preview_falls_back_to_repr_when_json_fails(caplog):
    caplog.set_level(logging.INFO, logger="document_resolution")

    class Unjsonable:
        def __repr__(self):
            return "UNJSONABLE_REPR"

    @logger("pj", log_result=True)
    def pj():
        # json.dumps(..., default=str) usually succeeds; force failure by returning
        # something that breaks json itself: set is not JSON serializable but default=str handles it.
        # So we instead cause json.dumps to choke by returning an object whose __str__ explodes.
        class BadStr:
            def __str__(self):
                raise RuntimeError("no str")

            def __repr__(self):
                return "BADSTR_REPR"

        return BadStr()

    pj()
    payload = _last(caplog)
    assert payload["event"] == "pj"
    # repr fallback should surface
    assert "BADSTR_REPR" in payload["result"]


def test_db_sink_resolved_from_string_attribute():
    sink = _CaptureSink()

    class Service:
        def __init__(self):
            self.sink = sink

        @logger("x", db_sink="sink")
        def do(self):
            return 1

    s = Service()
    assert s.do() == 1
    assert len(sink.items) == 1
    assert sink.items[0]["event"] == "x"


def test_db_sink_string_requires_bound_method_error():
    class Service:
        @logger("x", db_sink="sink")
        def do(self):
            return 1

    with pytest.raises(RuntimeError, match="expects a bound method"):
        Service.do()


def test_db_sink_string_missing_attribute_error():
    class Service:
        @logger("x", db_sink="missing")
        def do(self):
            return 1

    with pytest.raises(RuntimeError, match="Attribute 'missing' not found"):
        Service().do()


def test_db_sink_callable_resolution_bound():
    sink = _CaptureSink()

    class Service:
        def __init__(self):
            self.sink = sink

        @logger("x", db_sink=lambda self: self.sink)
        def do(self):
            return 1

    assert Service().do() == 1
    assert len(sink.items) == 1


@pytest.mark.asyncio
async def test_async_success_and_log_before_and_db_sink():
    sink = _CaptureSink()

    @logger("a", arg_names=["x"], log_before=True, db_sink=sink)
    async def a(x: int) -> int:
        return x * 2

    assert await a(2) == 4
    # pre-call + final success: db_sink only receives the final success entry (pre-call also uses emit_async,
    # but still hits db_sink)
    assert len(sink.items) >= 1
    assert any(p["event"] == "a" for p in sink.items)


@pytest.mark.asyncio
async def test_async_exception_path_emits_and_reraises(caplog):
    caplog.set_level(logging.INFO, logger="document_resolution")

    @logger("a", arg_names=["x"])
    async def a(x: int) -> int:
        raise KeyError("nope")

    with pytest.raises(KeyError):
        await a(9)

    payload = _last(caplog)
    assert payload["event"].startswith("Exception in ")
    assert payload["function"] == "a"
    assert payload["args"] == {"x": 9}
    assert "KeyError" in payload["error"]
