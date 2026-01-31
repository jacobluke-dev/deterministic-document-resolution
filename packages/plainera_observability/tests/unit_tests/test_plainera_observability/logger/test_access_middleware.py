import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from observability.config import REQ_ID_HEADER
from observability.logger.access_middleware import access_middleware


@pytest.fixture
def app():
    app = FastAPI()
    access_middleware(app)

    @app.get("/ping")
    def ping(): return {"ok": True}

    return app


def test_access_log_and_header(app, monkeypatch, capture_sink):
    sink = capture_sink()

    from observability.logger import access_middleware  # wherever emit is imported
    real_emit = access_middleware.emit

    def wrapped_emit(event, **kw):
        kw["db_sink"] = sink
        return real_emit(event, **kw)

    monkeypatch.setattr(access_middleware, "emit", wrapped_emit)

    with TestClient(app) as client:
        r = client.get("/ping", headers={REQ_ID_HEADER: "test-req-1"})
        _ = r.content

    assert r.status_code == 200
    assert r.headers[REQ_ID_HEADER] == "test-req-1"
    assert sink.items, "no access payload captured"

    p = sink.items[-1]
    assert p["event"] == "http_access"
    assert p["path"] == "/ping"
