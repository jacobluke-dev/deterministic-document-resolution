from fastapi import FastAPI
from fastapi.testclient import TestClient
from observability.config import REQ_ID_HEADER
from observability.http.request_id import RequestIDMiddleware

def test_request_id_injected_and_preserved():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    c = TestClient(app)

    # Injected when missing
    r = c.get("/ping")
    assert REQ_ID_HEADER in r.headers
    assert r.headers[REQ_ID_HEADER]  # non-empty

    # Preserved when provided
    r2 = c.get("/ping", headers={REQ_ID_HEADER: "rid-123"})
    assert r2.headers[REQ_ID_HEADER] == "rid-123"
