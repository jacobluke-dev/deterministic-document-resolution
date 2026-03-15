from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from public_api.core.settings import app_settings
from sqlalchemy import text

API_KEY_RE = re.compile(
    r"^uak_(?P<prefix>[a-z0-9]{2,10})_(?P<key_id>[A-Za-z0-9_-]{8,24})_(?P<secret>[A-Za-z0-9_-]{32,80})$"
)

# Argon2id defaults here are reasonable; tune later if needed.
_PH = PasswordHasher()


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated API principal derived from an API key."""
    key_id: int
    prefix: str
    user_id: int | None
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedApiKey:
    """Parsed API key components extracted from a raw `uak_...` key string."""
    prefix: str
    key_id: str
    secret: str


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    """Database representation of an API key record suitable for auth decisions."""
    id: int
    user_id: int | None
    prefix: str
    key_hash: str
    scopes: tuple[str, ...]
    is_active: bool
    expires_at: datetime | None
    daily_quota: int | None


class _TTLCache:
    def __init__(self) -> None:
        self._by_key_id: dict[str, tuple[float, ApiKeyRecord]] = {}

    def get(self, key_id: str) -> ApiKeyRecord | None:
        """
        Return a cached key record if present and not expired.

        Uses `API_KEY_CACHE_TTL_SECONDS` to control caching. If TTL is <= 0, the
        cache is treated as disabled and this always returns `None`.

        Args:
          key_id: API key identifier used as the cache key.

        Returns:
          The cached `ApiKeyRecord` if available and fresh; otherwise `None`.

        Notes:
          - Expired entries are evicted on read.
          - This cache is in-process only (per worker).
        """
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
        """
        Store a key record in the cache with the configured TTL.

        Args:
          key_id: API key identifier used as the cache key.
          rec: Record to cache.

        Notes:
          - If TTL <= 0, this is a no-op.
          - Overwrites any existing entry for the same key id.
        """
        ttl = int(app_settings.API_KEY_CACHE_TTL_SECONDS)
        if ttl <= 0:
            return
        self._by_key_id[key_id] = (time.time() + ttl, rec)

    def invalidate(self, key_id: str) -> None:
        """
        Remove a cached key record, if present.

        Args:
          key_id: API key identifier to evict.
        """
        self._by_key_id.pop(key_id, None)


_CACHE = _TTLCache()


def parse_api_key(raw: str, *, allow_prefixes: set[str]) -> ParsedApiKey | None:
    """
    Parse and validate a raw `X-API-Key` value into structured components.

    Validates format against `API_KEY_RE`, enforces the provided prefix allowlist,
    and returns the parsed (prefix, key_id, secret) on success.

    Args:
      raw: Raw API key string, typically from the `X-API-Key` header.
      allow_prefixes: Set of allowed key prefixes (e.g. {"live", "test"}).

    Returns:
      A `ParsedApiKey` if parsing and prefix validation succeed; otherwise `None`.

    Notes:
      - Input is stripped before matching.
      - Returns `None` for malformed keys rather than raising, to avoid leaks.
    """
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
    """
    Hash API key secret material for storage.

    Uses the configured hashing scheme to produce a stored hash (never store the
    raw secret). Currently supports Argon2id only.

    Args:
      secret: Raw secret material from a generated API key.
      scheme: Hashing scheme identifier (default: "argon2id").

    Returns:
      A hash string suitable for persistence in the `api_keys.key_hash` column.

    Raises:
      RuntimeError: If an unsupported hashing scheme is requested.

    Notes:
      - This function is intentionally strict; unsupported schemes fail fast.
      - Argon2id parameters are provided by `argon2.PasswordHasher` defaults.
    """
    if scheme == "argon2id":
        return _PH.hash(secret)
    raise RuntimeError("bcrypt scheme not implemented in this service yet (choose argon2id).")


def verify_secret(
    secret: str,
    key_hash: str,
    *,
    scheme: Literal["argon2id", "bcrypt"] = "argon2id",
) -> bool:
    """
    Verify presented secret material against a stored hash.

    Performs a constant-time verification appropriate to the configured scheme.
    Any mismatch or unexpected error is treated as authentication failure.

    Args:
      secret: Presented secret material from the client.
      key_hash: Stored hash retrieved from the database.
      scheme: Hashing scheme identifier (default: "argon2id").

    Returns:
      `True` if verification succeeds; otherwise `False`.

    Notes:
      - Unknown hash formats and unexpected errors are treated as failure to
        avoid leaking details to callers.
      - bcrypt is not implemented; non-argon2id schemes return `False`.
    """
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
    Generate a new API key triplet: (key_id, secret, full_key_string).

    Produces a short `key_id` used for database lookup, a longer random `secret`
    used for authentication, and the full `uak_{prefix}_{key_id}_{secret}` string
    intended to be shown once to an operator/customer.

    Args:
      prefix: Environment/category prefix embedded into the full key string.

    Returns:
      Tuple of `(key_id, secret, full_key_string)`.

    Notes:
      - `key_id` and `secret` are generated from URL-safe randomness and then
        normalised to match the regex constraints.
      - This function does not validate the prefix allowlist; callers should.
    """
    key_id = secrets.token_urlsafe(9).replace("-", "").replace("_", "")[:12]
    secret = secrets.token_urlsafe(32).replace("-", "").replace("_", "")[:40]
    full = f"uak_{prefix}_{key_id}_{secret}"
    return key_id, secret, full


def _now_utc() -> datetime:
    """
    Return the current UTC time as a timezone-aware `datetime`.

    Returns:
      A timezone-aware UTC `datetime`.
    """
    return datetime.now(timezone.utc)


def fetch_key_record(dbm: Any, key_id: str) -> ApiKeyRecord | None:
    """
    Load an active, unexpired API key record from cache or database.

    Checks the in-process TTL cache first; on a miss, queries `unacronym.api_keys`
    for a matching key id that is active and not expired, then caches and returns
    a normalised `ApiKeyRecord`.

    Args:
      dbm: Database/session manager providing a `session()` context.
      key_id: API key identifier to look up.

    Returns:
      An `ApiKeyRecord` if found and eligible for auth; otherwise `None`.

    Notes:
      - Cache behaviour is controlled by `API_KEY_CACHE_TTL_SECONDS`; TTL <= 0
        disables caching.
      - The database query filters by `is_active=true` and `expires_at` in the
        future (or NULL).
    """
    cached = _CACHE.get(key_id)
    if cached is not None:
        return cached

    stmt = text(
        """
        SELECT id,
               user_id,
               prefix,
               key_hash,
               scopes,
               is_active,
               expires_at,
               daily_quota
        FROM unacronym.api_keys
        WHERE key_id = :key_id
          AND is_active = true
          AND (expires_at IS NULL OR expires_at > now()) LIMIT 1
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
        daily_quota=(int(row["daily_quota"]) if row["daily_quota"] is not None else None),
    )
    _CACHE.put(key_id, rec)
    return rec


def update_last_used(dbm: Any, key_id: str) -> None:
    """
    Update the `last_used_at` timestamp for an API key record.

    Performs a simple `UPDATE` to set `last_used_at = now()` for the given key id.

    Args:
      dbm: Database/session manager providing a `session()` context.
      key_id: API key identifier to mark as used.

    Notes:
      - Intended to be best-effort telemetry; callers may choose to execute this
        asynchronously depending on request latency goals.
      - This function does not validate that the key is active/unexpired.
    """
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
