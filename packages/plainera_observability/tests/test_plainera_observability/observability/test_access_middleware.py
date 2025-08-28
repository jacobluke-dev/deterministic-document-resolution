import json, logging, pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plainera_observability.config import REQ_ID_HEADER
from plainera_observability.observability.access_middleware import access_middleware

def _last_json(caplog) -> dict:
    return json.loads(caplog.records[-1].msg)

@pytest.fixture
def app():
    app = FastAPI()
    access_middleware(app)

    @app.get("/ping")
    def ping(): return {"ok": True}

    return app

def test_access_log_and_header(app, caplog):
    caplog.set_level(logging.INFO, logger="plainera")
    client = TestClient(app)
    r = client.get("/ping", headers={REQ_ID_HEADER: "test-req-1"})
    assert r.status_code == 200
    assert r.headers[REQ_ID_HEADER] == "test-req-1"

    payload = json.loads(caplog.records[-1].msg)
    assert payload["event"] == "http_access"
    assert payload["path"] == "/ping"
    assert payload["method"] == "GET"
    assert payload["status"] == 200
    assert payload["request_id"] == "test-req-1"
    assert isinstance(payload["duration_ms"], int)
