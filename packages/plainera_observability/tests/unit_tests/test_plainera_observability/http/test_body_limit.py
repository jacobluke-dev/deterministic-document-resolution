import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from observability.http.body_limit import BodySizeLimitMiddleware  # adjust import if needed


def make_json_bytes(n: int) -> bytes:
    """Return a JSON body whose encoded size is guaranteed > n bytes."""
    doc = {"body": "x" * (n + 20)}  # overshoot by ~20 to avoid fencepost issues
    return json.dumps(doc).encode("utf-8")


class TestBodySizeLimitMiddleware:
    @pytest.fixture(autouse=True)
    def client(self) -> TestClient:
        app = FastAPI()
        app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=100)

        @app.post("/echo")
        def echo(payload: dict):
            return payload

        return TestClient(app)

    def test_body_size_limit_returns_200(self, client: TestClient) -> None:
        ok = json.dumps({"x": "ok"}).encode("utf-8")
        r = client.post("/echo", data=ok, headers={"content-type": "application/json"})
        assert r.status_code == 200

    def test_body_limit_returns_413(self, client: TestClient) -> None:
        too_big = make_json_bytes(100)
        r = client.post("/echo", data=too_big, headers={"content-type": "application/json"})
        assert r.status_code == 413
