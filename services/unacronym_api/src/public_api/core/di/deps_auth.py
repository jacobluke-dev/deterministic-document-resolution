from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Annotated, Any

import anyio
from fastapi import Depends, Header, HTTPException
from wiring.composition import sink

from public_api.cli.api_keys import parse_hash_scheme
from public_api.core.auth.api_keys import Principal, fetch_key_record, parse_api_key, update_last_used, verify_secret
from public_api.core.di.deps import get_dbm
from public_api.core.di.deps_settings import get_settings
from public_api.core.services.api_abuse_protection import ApiAbuseProtectionService
from public_api.core.settings import AppSettings

logger = logging.getLogger("plainera")


def _unauthenticated_exc() -> HTTPException:
    return HTTPException(status_code=401, detail="UNAUTHENTICATED")


async def require_api_key(
    settings: Annotated[AppSettings, Depends(get_settings)],
    dbm: Annotated[Any, Depends(get_dbm)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """
    Authenticate a request using the `X-API-Key` header and return a `Principal`.

    Parses and validates the presented API key (prefix/key_id/secret), enforces an
    allowed-prefix policy, loads the corresponding key record from storage, and
    verifies the secret against the stored hash using the configured scheme. On
    success, returns a `Principal` representing the authenticated key identity.

    Args:
      settings: Application settings injected via `Depends(get_settings)`.
      dbm: Database/session manager injected via `Depends(get_dbm)`.
      x_api_key: Raw API key from the `X-API-Key` request header.

    Returns:
      A `Principal` containing the authenticated key identity, prefix, optional user
      linkage, and scopes.

    Raises:
      HTTPException: If the header is missing/malformed, the key record is not found,
        the prefix mismatches the record, or secret verification fails.

    Notes:
      - If `AUTH_DISABLED=true` and `APP_ENV=="local"`, authentication is bypassed and
        a development `Principal` is returned.
      - If `API_KEY_LAST_USED_ASYNC` is enabled, `last_used_at` may be updated in a
        best-effort background task; ensure the DB/session lifecycle supports this.
    """
    # Local-only bypass
    if settings.AUTH_DISABLED and settings.APP_ENV == "local":
        logger.warning("AUTH_DISABLED=true in local; skipping API key auth.")
        return Principal(key_id=90909090, prefix="dev", user_id=None, scopes=())

    allow = {
        p.strip()
        for p in (settings.API_KEY_PREFIX_ALLOWLIST or "live,test").split(",")
        if p.strip()
    }

    if not x_api_key:
        raise _unauthenticated_exc()

    parsed = parse_api_key(x_api_key, allow_prefixes=allow)
    if parsed is None:
        raise _unauthenticated_exc()

    rec = await anyio.to_thread.run_sync(fetch_key_record, dbm, parsed.key_id)
    if rec is None or rec.prefix != parsed.prefix:
        raise _unauthenticated_exc()

    scheme = parse_hash_scheme(settings.API_KEY_HASH_SCHEME or "argon2id")
    ok = await anyio.to_thread.run_sync(
        partial(verify_secret, parsed.secret, rec.key_hash, scheme=scheme)
    )
    if not ok:
        raise _unauthenticated_exc()

    limiter = ApiAbuseProtectionService(dbm, sink)
    await anyio.to_thread.run_sync(
        partial(
            limiter.enforce,
            api_key_id=rec.id,
            daily_quota_override=rec.daily_quota,
        )
    )

    principal = Principal(
        key_id=rec.id,
        prefix=rec.prefix,
        user_id=rec.user_id,
        scopes=tuple(rec.scopes or ()),
    )

    # last_used_at best-effort
    if settings.API_KEY_LAST_USED_ASYNC:
        async def _touch() -> None:
            try:
                await anyio.to_thread.run_sync(update_last_used, dbm, parsed.key_id)
            except Exception as e:
                logger.warning("failed to update last_used_at for key_id=%s: %r", parsed.key_id, e)

        # Fire-and-forget (may fail if dbm is request-scoped and closes quickly)
        asyncio.create_task(_touch())
    else:
        await anyio.to_thread.run_sync(update_last_used, dbm, parsed.key_id)

    return principal
