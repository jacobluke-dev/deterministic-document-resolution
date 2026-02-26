from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy import text

from public_api.core.settings import app_settings

API_KEY_RE = re.compile(
    r"^uak_(?P<prefix>[a-z0-9]{2,10})_(?P<key_id>[A-Za-z0-9_-]{8,24})_(?P<secret>[A-Za-z0-9_-]{32,80})$"
)

# Argon2id defaults here are reasonable; tune later if needed.
_PH = PasswordHasher()


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated API principal derived from an API key."""
    key_id: str
    prefix: str
    user_id: int | None
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedApiKey:
    prefix: str
    key_id: str
    secret: str


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    id: int
    user_id: int | None
    prefix: str
    key_hash: str
    scopes: tuple[str, ...]
    is_active: bool
    expires_at: datetime | None


class _TTLCache:
    def __init__(self) -> None:
        self._by_key_id: dict[str, tuple[float, ApiKeyRecord]] = {}

    def get(self, key_id: str) -> ApiKeyRecord | None:
        ttl = int(app_settings.API_KEY_CACHE_TTL_SECONDS)
        if ttl <= 0:
            return None

        item = self._by_key_id.get(key_id)
        if not item:
            return None

        expires_ts, rec = item
        if time.time() > expires_ts:
            self._by_key_id.pop(key_id, None)
            return None
        return rec

    def put(self, key_id: str, rec: ApiKeyRecord) -> None:
        ttl = int(app_settings.API_KEY_CACHE_TTL_SECONDS)
        if ttl <= 0:
            return
        self._by_key_id[key_id] = (time.time() + ttl, rec)

    def invalidate(self, key_id: str) -> None:
        self._by_key_id.pop(key_id, None)


_CACHE = _TTLCache()


def parse_api_key(raw: str, *, allow_prefixes: set[str]) -> ParsedApiKey | None:
    m = API_KEY_RE.match(raw.strip())
    if not m:
        return None

    prefix = m.group("prefix")
    if prefix not in allow_prefixes:
        return None

    return ParsedApiKey(
        prefix=prefix,
        key_id=m.group("key_id"),
        secret=m.group("secret"),
    )


def hash_secret(secret: str, *, scheme: Literal["argon2id", "bcrypt"] = "argon2id") -> str:
    if scheme == "argon2id":
        return _PH.hash(secret)
    raise RuntimeError("bcrypt scheme not implemented in this service yet (choose argon2id).")


def verify_secret(secret: str, key_hash: str, *, scheme: Literal["argon2id", "bcrypt"] = "argon2id") -> bool:
    if scheme == "argon2id":
        try:
            return _PH.verify(key_hash, secret)
        except VerifyMismatchError:
            return False
        except Exception:
            # Treat unknown hash formats as failure (do not leak)
            return False
    return False


def generate_key(prefix: str) -> tuple[str, str, str]:
    """
    Returns (key_id, secret, full_key_string).
    """
    key_id = secrets.token_urlsafe(9).replace("-", "").replace("_", "")[:12]
    secret = secrets.token_urlsafe(32).replace("-", "").replace("_", "")[:40]
    full = f"uak_{prefix}_{key_id}_{secret}"
    return key_id, secret, full


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fetch_key_record(dbm: Any, key_id: str) -> ApiKeyRecord | None:
    cached = _CACHE.get(key_id)
    if cached is not None:
        return cached

    stmt = text(
        """
        SELECT id, user_id, prefix, key_hash, scopes, is_active, expires_at
        FROM unacronym.api_keys
        WHERE key_id = :key_id
          AND is_active = true
          AND (expires_at IS NULL OR expires_at > now())
        LIMIT 1
        """
    )

    with dbm.session() as s:
        row = s.execute(stmt, {"key_id": key_id}).mappings().first()

    if not row:
        return None

    rec = ApiKeyRecord(
        id=int(row["id"]),
        user_id=(int(row["user_id"]) if row["user_id"] is not None else None),
        prefix=str(row["prefix"]),
        key_hash=str(row["key_hash"]),
        scopes=tuple(row["scopes"] or []),
        is_active=bool(row["is_active"]),
        expires_at=row["expires_at"],
    )
    _CACHE.put(key_id, rec)
    return rec


def update_last_used(dbm: Any, key_id: str) -> None:
    stmt = text(
        """
        UPDATE unacronym.api_keys
        SET last_used_at = now()
        WHERE key_id = :key_id
        """
    )
    with dbm.session() as s:
        s.execute(stmt, {"key_id": key_id})
        s.commit()
