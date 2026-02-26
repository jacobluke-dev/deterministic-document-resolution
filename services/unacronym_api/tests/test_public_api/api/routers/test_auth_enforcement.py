import pytest
from sqlalchemy import text

from public_api.core.auth.api_keys import generate_key, hash_secret


@pytest.mark.anyio
async def test_v1_requires_api_key(client_no_auth):
    r = await client_no_auth.post("/v1/resolve", json={"text": "Foo (BAR)"})
    assert r.status_code == 401


@pytest.mark.anyio
async def test_v1_happy_path_with_key(client, session_factory):
    key_id, secret, full = generate_key("test")
    key_hash = hash_secret(secret, scheme="argon2id")

    with session_factory() as s:
        s.execute(
            text(
                """
                INSERT INTO unacronym.api_keys (key_id, key_hash, name, prefix, scopes, is_active, created_at)
                VALUES (:key_id, :key_hash, 'test', :prefix, '{}'::text[], true, now())
                """
            ),
            {"key_id": key_id, "key_hash": key_hash, "prefix": "test"},
        )
        s.commit()

    r = await client.post(
        "/v1/resolve",
        json={"text": "The new iPhone Operating system (iOS) was developed by Apple."},
        headers={"X-API-Key": full},
    )
    assert r.status_code == 200


@pytest.mark.anyio
async def test_revoked_key_401(client, session_factory):
    key_id, secret, full = generate_key("test")
    key_hash = hash_secret(secret, scheme="argon2id")

    with session_factory() as s:
        s.execute(
            text(
                """
                INSERT INTO unacronym.api_keys (key_id, key_hash, prefix, scopes, is_active, created_at)
                VALUES (:key_id, :key_hash, :prefix, '{}'::text[], false, now())
                """
            ),
            {"key_id": key_id, "key_hash": key_hash, "prefix": "test"},
        )
        s.commit()

    r = await client.post("/v1/resolve", json={"text": "Foo (BAR)"}, headers={"X-API-Key": full})
    assert r.status_code == 401
