import asyncio
from typing import Any

import pytest
from httpx import Response
from public_api.db.models import GlossaryEntry
from public_api.schemas.error import ErrorCode


def _get_fastapi_app_from_client(client):
    transport = getattr(client, "_transport", None)
    if transport is None:
        raise RuntimeError("client._transport missing")
    # httpx versions vary: try both
    return getattr(transport, "app", None) or getattr(transport, "_app", None)


class TestV1Resolve:
    @pytest.fixture(autouse=True)
    def seed_for_this_class(self, session_factory):
        # minimal deterministic seed used by multiple tests here
        with session_factory() as s:
            # Upsert-ish for idempotence across parametrized runs
            if not s.query(GlossaryEntry).filter_by(acronym="MPS").first():
                s.add(GlossaryEntry(
                    acronym="MPS",
                    definition="Metropolitan Police Service, the territorial police force for Greater London.",
                    source="test",
                ))
            if not s.query(GlossaryEntry).filter_by(acronym="ABC").first():
                s.add(GlossaryEntry(
                    acronym="ABC",
                    definition="Alpha Beta Charlie.",
                    source="test",
                ))
            s.commit()
        yield

    @pytest.mark.anyio
    async def test_happy_path(self, client):
        payload = {"text": "The Metropolitan Police Service (MPS) operates in London."}
        r = await client.post("/v1/resolve", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "acronyms" in body and "meta" in body
        assert body["meta"]["input_chars"] == len(payload["text"])
        assert r.headers.get("X-Request-Id")
        assert int(r.headers["X-Input-Bytes"]) > 0
        assert int(r.headers["X-Body-Limit-Bytes"]) > 0

    @pytest.mark.anyio
    async def test_empty_text_422(self, client):
        r = await client.post("/v1/resolve", json={"text": "   "})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == ErrorCode.UNPROCESSABLE_ENTITY

    @pytest.mark.anyio
    async def test_invalid_option_bounds(self, client):
        r = await client.post(
            "/v1/resolve",
            json={"text": "x", "options": {"max_definitions_per_acronym": 999}},
        )
        assert r.status_code in (400, 422)

    @pytest.mark.anyio
    async def test_text_too_large(self, client):
        big_text = "X" * 100_001
        r = await client.post("/v1/resolve", json={"text": big_text})
        assert r.status_code == 413
        err = r.json()["error"]
        assert err["code"] == ErrorCode.PAYLOAD_TOO_LARGE

    @pytest.mark.anyio
    async def test_timeout(self, monkeypatch, client):
        async def slow_resolve(self, *args, **kwargs):
            await asyncio.sleep(2.0)
            return []

        monkeypatch.setattr(
            "services.unacronym_api.src.public_api.api.routers.resolve.app_settings.REQUEST_TIMEOUT_MS",
            500,  # 0.5s
            raising=False,
        )

        # Patch the core call to be slow (async)
        monkeypatch.setattr(
            "core.services.resolver.AcronymResolver.resolve",
            slow_resolve,
        )

        r = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert body["error"]["details"]["timeout_ms"] == 500


        _value = 0

    def _stable_json(self, resp: Response) -> dict[str, Any]:
        body = resp.json()
        meta = dict(body.get("meta", {}))
        # Strip fields that can vary run-to-run
        for k in ("processing_ms", "created_at"):
            meta.pop(k, None)
        body["meta"] = meta
        return body

    @pytest.mark.anyio
    async def test_deterministic_output(self, client):
        payload = {"text": "Alpha (ABC). Another (ABC)."}
        r1 = await client.post("/v1/resolve", json=payload)
        r2 = await client.post("/v1/resolve", json=payload)
        assert self._stable_json(r1) == self._stable_json(r2)

    @pytest.mark.anyio
    async def test_overloaded(self, client):
        class DummySemaphore:
            def __init__(self):
                self._value = 0

            def locked(self): return True
        from public_api.core import deps as deps_mod

        app = _get_fastapi_app_from_client(client)
        app.dependency_overrides[deps_mod.get_semaphore] = lambda: DummySemaphore()
        try:
            r = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})
            assert r.status_code == 503
            assert r.json()["error"]["details"]["reason"] == "OVERLOADED"
        finally:
            app.dependency_overrides.pop(deps_mod.get_semaphore, None)
