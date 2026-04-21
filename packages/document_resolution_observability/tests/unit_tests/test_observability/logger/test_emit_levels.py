import json
import logging

import pytest
from observability.logger.emit import emit, emit_async
from observability.logger.levels import STD_LEVEL, LogLevel


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

    from observability.logger.decorator import logger
    @logger("work", arg_names=["x"])
    def work(x: int) -> int:
        return x * 2

    assert work(3) == 6
    payload, rec = _last_json(caplog)
    assert payload["event"] == "work"
    assert payload["level"] == "info"             # default in decorator
    assert rec.levelno == logging.INFO


class _SyncSink:
    def __init__(self):
        self.items = []

    def enqueue(self, payload):
        self.items.append(payload)


class _AsyncSink:
    def __init__(self):
        self.items = []

    async def enqueue_async(self, payload):
        self.items.append(payload)


class _BoomSink:
    def enqueue(self, payload):
        raise RuntimeError("boom")


@pytest.mark.parametrize("which", ["enqueue", "enqueue_async"])
def test_emit_calls_db_sink_variants(caplog, which):
    caplog.set_level(logging.DEBUG, logger="document_resolution")

    sink = _SyncSink() if which == "enqueue" else _AsyncSink()
    emit("evt", db_sink=sink, logger_type="t")

    assert len(sink.items) == 1
    assert sink.items[0]["event"] == "evt"

    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "evt"


def test_emit_db_sink_exception_does_not_break_logging(caplog):
    caplog.set_level(logging.DEBUG, logger="document_resolution")

    emit("evt", db_sink=_BoomSink(), logger_type="t")

    # We should still have an emitted JSON payload log entry at the end.
    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "evt"

    # And we should have warned about the sink failure.
    assert any("db_sink failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_emit_async_prefers_enqueue_async(caplog):
    caplog.set_level(logging.DEBUG, logger="document_resolution")

    sink = _AsyncSink()
    await emit_async("evt", db_sink=sink, logger_type="t")

    assert len(sink.items) == 1
    assert sink.items[0]["event"] == "evt"

    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "evt"


@pytest.mark.asyncio
async def test_emit_async_falls_back_to_to_thread_for_sync_sink(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="document_resolution")

    sink = _SyncSink()

    calls = {"n": 0}

    async def _fake_to_thread(fn, *args, **kwargs):
        calls["n"] += 1
        return fn(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", _fake_to_thread, raising=True)

    await emit_async("evt", db_sink=sink, logger_type="t")

    assert calls["n"] == 1
    assert len(sink.items) == 1
    assert sink.items[0]["event"] == "evt"
