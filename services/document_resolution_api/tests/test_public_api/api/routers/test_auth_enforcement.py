from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from public_api.core.auth.api_keys import generate_key, hash_secret
from sqlalchemy import text


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _insert_api_key(
    session_factory,
    *,
    prefix: str = "test",
    name: str | None = "test-key",
    scopes: list[str] | None = None,
    is_active: bool = True,
    expires_at: datetime | None = None,
) -> str:
    """
    Insert a fresh API key row and return the full key string (uak_*).
    """
    key_id, secret, full = generate_key(prefix)
    key_hash = hash_secret(secret, scheme="argon2id")

    with session_factory() as s:
        s.execute(
            text(
                """
                INSERT INTO document_resolution.api_keys
                  (key_id, key_hash, name, prefix, scopes, is_active, created_at, expires_at)
                VALUES
                  (:key_id, :key_hash, :name, :prefix, :scopes, :is_active, :created_at, :expires_at)
                """
            ),
            {
                "key_id": key_id,
                "key_hash": key_hash,
                "name": name,
                "prefix": prefix,
                "scopes": scopes or [],
                "is_active": is_active,
                "created_at": _utcnow(),
                "expires_at": expires_at,
            },
        )
        s.commit()

    return full


def _get_last_used_at(session_factory, *, key_id: str) -> Any:
    with session_factory() as s:
        row = s.execute(
            text(
                """
                SELECT last_used_at
                FROM document_resolution.api_keys
                WHERE key_id = :key_id
                """
            ),
            {"key_id": key_id},
        ).fetchone()
        return row[0] if row else None


def _extract_key_id(full_key: str) -> str:
    # uak_{prefix}_{keyId}_{secret}
    parts = full_key.split("_", 3)
    if len(parts) < 4:
        raise ValueError("not a full key")
    return parts[2]


@pytest.mark.anyio
async def test_v1_requires_api_key(client):
    r = await client.post(
        "/v1/resolve",
        json={"text": "Foo (BAR)"},
        headers={"X-API-Key": ""},  # forces missing/empty
    )
    assert r.status_code == 401


@pytest.mark.anyio
async def test_v1_rejects_malformed_key(client):
    r = await client.post(
        "/v1/resolve",
        headers={"X-API-Key": "uak_test_bad_bad"},
        json={"text": "Foo (BAR)"},
    )
    assert r.status_code == 401


@pytest.mark.anyio
async def test_v1_rejects_wrong_secret(client, session_factory):
    full = _insert_api_key(session_factory, prefix="test", name="wrong-secret")
    key_id = _extract_key_id(full)

    # keep same prefix + key_id, but swap secret portion
    bad = f"uak_test_{key_id}_THIS_IS_NOT_THE_SECRET"

    r = await client.post(
        "/v1/resolve",
        headers={"X-API-Key": bad},
        json={"text": "Foo (BAR)"},
    )
    assert r.status_code == 401


@pytest.mark.anyio
async def test_v1_rejects_inactive_key(client, session_factory):
    full = _insert_api_key(session_factory, is_active=False, name="inactive")
    r = await client.post(
        "/v1/resolve",
        headers={"X-API-Key": full},
        json={"text": "Foo (BAR)"},
    )
    assert r.status_code == 401


@pytest.mark.anyio
async def test_v1_rejects_expired_key(client, session_factory):
    full = _insert_api_key(
        session_factory,
        name="expired",
        expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )
    r = await client.post(
        "/v1/resolve",
        headers={"X-API-Key": full},
        json={"text": "Foo (BAR)"},
    )
    assert r.status_code == 401


@pytest.mark.anyio
async def test_v1_accepts_valid_key_and_updates_last_used(client, session_factory):
    after = None
    full = _insert_api_key(session_factory, name="valid")
    key_id = _extract_key_id(full)

    before = _get_last_used_at(session_factory, key_id=key_id)
    assert before is None

    r = await client.post(
        "/v1/resolve",
        headers={"X-API-Key": full},
        json={"text": "The new iPhone Operating system (iOS) was developed by Apple."},
    )
    assert r.status_code == 200

    # last_used_at is async/eventual; give it a small window.
    for _ in range(20):
        await asyncio.sleep(0.1)
        after = _get_last_used_at(session_factory, key_id=key_id)
        if after is not None:
            break

    assert after is not None
