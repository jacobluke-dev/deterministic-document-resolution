from fastapi import FastAPI
from fastapi.testclient import TestClient
from observability.config import REQ_ID_HEADER
from observability.http.request_id import RequestIDMiddleware
from observability.logger.access_middleware import access_middleware


def test_request_id_with_access_middleware_no_clash():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    access_middleware(app)  # function-based middleware

    @app.get("/ping")
    def ping():
        return {"ok": True}

    c = TestClient(app)

    # Missing -> injected
    r = c.get("/ping")
    assert REQ_ID_HEADER in r.headers
    v1 = r.headers[REQ_ID_HEADER]
    assert v1

    # Provided -> preserved (single authoritative value set by access_middleware)
    r2 = c.get("/ping", headers={REQ_ID_HEADER: "rid-abc"})
    assert r2.headers[REQ_ID_HEADER] == "rid-abc"
