from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

import anyio
from fastapi import Depends, Header, Request, HTTPException

from public_api.core.auth.api_keys import Principal, fetch_key_record, parse_api_key, update_last_used, verify_secret
from public_api.core.deps import get_dbm
from public_api.core.settings import app_settings

logger = logging.getLogger("plainera")


def _unauthenticated_exc() -> HTTPException:
    return HTTPException(status_code=401, detail="UNAUTHENTICATED")


async def require_api_key(
    request: Request,
    dbm: Annotated[Any, Depends(get_dbm)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    # Local-only bypass
    if app_settings.AUTH_DISABLED and app_settings.APP_ENV == "local":
        logger.warning("AUTH_DISABLED=true in local; skipping API key auth.")
        return Principal(key_id="dev", prefix="dev", user_id=None, scopes=())

    allow = {p.strip() for p in (app_settings.API_KEY_PREFIX_ALLOWLIST or "live,test").split(",") if p.strip()}

    if not x_api_key:
        raise _unauthenticated_exc()

    parsed = parse_api_key(x_api_key, allow_prefixes=allow)
    if parsed is None:
        raise _unauthenticated_exc()

    rec = await anyio.to_thread.run_sync(fetch_key_record, dbm, parsed.key_id)
    if rec is None:
        raise _unauthenticated_exc()

    # Prefix must match record (prevents keyId reuse across prefixes)
    if rec.prefix != parsed.prefix:
        raise _unauthenticated_exc()

    scheme = app_settings.API_KEY_HASH_SCHEME
    ok = await anyio.to_thread.run_sync(
        lambda: verify_secret(parsed.secret, rec.key_hash, scheme=scheme)
    )
    if not ok:
        raise _unauthenticated_exc()

    principal = Principal(key_id=parsed.key_id, prefix=rec.prefix, user_id=rec.user_id, scopes=rec.scopes)

    # last_used_at best-effort (do not block)
    if app_settings.API_KEY_LAST_USED_ASYNC:
        async def _touch() -> None:
            try:
                await anyio.to_thread.run_sync(update_last_used, dbm, parsed.key_id)
            except Exception as e:
                logger.warning("failed to update last_used_at for key_id=%s: %r", parsed.key_id, e)

        asyncio.create_task(_touch())
    else:
        await anyio.to_thread.run_sync(update_last_used, dbm, parsed.key_id)

    return principal
