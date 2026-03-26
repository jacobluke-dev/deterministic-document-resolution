from __future__ import annotations

import pytest
from public_api.api.routers import resolve as resolve_mod
from public_api.core.di import deps as deps_mod, deps_auth as deps_auth_mod
from public_api.core.auth.api_keys import Principal
from public_api.schemas.error import ErrorCode

pytestmark = [pytest.mark.integration]

_ORIGINAL_REQUEST_TIMEOUT_MS = resolve_mod.app_settings.REQUEST_TIMEOUT_MS


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setattr(
        resolve_mod.app_settings,
        "REQUEST_TIMEOUT_MS",
        _ORIGINAL_REQUEST_TIMEOUT_MS,
    )


def _get_fastapi_app_from_client(client):
    transport = getattr(client, "_transport", None)
    if transport is None:
        raise RuntimeError("client._transport missing")
    return getattr(transport, "app", None) or getattr(transport, "_app", None)


class TestResilienceE2E:
    @pytest.mark.anyio
    async def test_empty_text_returns_422(self, client):
        response = await client.post("/v1/resolve", json={"text": "   "})

        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == ErrorCode.UNPROCESSABLE_ENTITY

    @pytest.mark.anyio
    async def test_payload_too_large_returns_413(self, client):
        response = await client.post("/v1/resolve", json={"text": "X" * 100_001})

        assert response.status_code == 413
        body = response.json()
        assert body["error"]["code"] == ErrorCode.PAYLOAD_TOO_LARGE

    @pytest.mark.anyio
    async def test_invalid_option_bounds_returns_validation_error(self, client):
        response = await client.post(
            "/v1/resolve",
            json={
                "text": "x",
                "options": {"max_definitions_per_acronym": 999},
            },
        )

        assert response.status_code in (400, 422)

    @pytest.mark.anyio
    async def test_timeout_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(resolve_mod.app_settings, "REQUEST_TIMEOUT_MS", 500)

        import public_api.core.services.resolve_service as rs

        def slow_detect_and_extract(*args, **kwargs):
            import time
            time.sleep(2.0)
            raise RuntimeError("should have timed out")

        monkeypatch.setattr(rs, "detect_and_extract", slow_detect_and_extract, raising=True)

        response = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})

        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"]["code"] == ErrorCode.SERVICE_UNAVAILABLE
        assert body["error"]["details"]["timeout_ms"] == 500

    @pytest.mark.anyio
    async def test_overloaded_returns_503(self, client):
        class DummySemaphore:
            def locked(self):
                return True

        app = _get_fastapi_app_from_client(client)
        app.dependency_overrides[deps_mod.get_semaphore] = lambda: DummySemaphore()
        app.dependency_overrides[deps_auth_mod.require_api_key] = lambda: Principal(
            key_id=1,
            prefix="test",
            user_id=None,
            scopes=(),
        )
        try:
            response = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})

            assert response.status_code == 503
            body = response.json()
            assert body["error"]["code"] == ErrorCode.SERVICE_UNAVAILABLE
            assert body["error"]["details"]["reason"] == "OVERLOADED"
        finally:
            app.dependency_overrides.pop(deps_mod.get_semaphore, None)
            app.dependency_overrides.pop(deps_auth_mod.require_api_key, None)
