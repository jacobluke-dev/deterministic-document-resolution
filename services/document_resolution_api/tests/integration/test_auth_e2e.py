import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration]


class TestAuthE2E:
    @pytest.mark.anyio
    async def test_missing_key_returns_401(self, client_no_auth):
        response = await client_no_auth.post("/v1/resolve", json={"text": "Foo (BAR)"})
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == "UNAUTHENTICATED"

    @pytest.mark.anyio
    async def test_inactive_key_returns_401(self, app, session_factory):
        from httpx import ASGITransport, AsyncClient
        from public_api.core.auth.api_keys import generate_key, hash_secret

        key_id, secret, full = generate_key("test")
        key_hash = hash_secret(secret, scheme="argon2id")

        with session_factory() as s:
            s.execute(
                text(
                    """
                    INSERT INTO document_resolution.api_keys
                        (key_id, key_hash, prefix, scopes, is_active, created_at)
                    VALUES
                        (:key_id, :key_hash, :prefix, '{}'::text[], false, now())
                    """
                ),
                {"key_id": key_id, "key_hash": key_hash, "prefix": "test"},
            )
            s.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-API-Key": full},
        ) as client:
            response = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"
