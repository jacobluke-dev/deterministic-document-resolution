from fastapi import FastAPI
from fastapi.testclient import TestClient
from plainera_observability.config import REQ_ID_HEADER
from plainera_observability.http.request_id import RequestIDMiddleware

def test_request_id_injected_and_preserved(caplog):
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    @app.get("/ping")  # simple route
    def ping(): return {"ok": True}
    c = TestClient(app)

    # injected when missing
    r = c.get("/ping")
    assert REQ_ID_HEADER in r.headers

    # preserved when provided
    r2 = c.get("/ping", headers={REQ_ID_HEADER: "rid-123"})
    assert r2.headers[REQ_ID_HEADER] == "rid-123"
