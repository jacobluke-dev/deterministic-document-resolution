import logging

import pytest
from httpx import AsyncClient
from public_api.main import create_app


@pytest.mark.asyncio
async def test_access_log_emits():
    app = create_app()
    logger = logging.getLogger("access")
    seen = {}

    class _Handler(logging.Handler):
        def emit(self, record):
            seen["ok"] = True

    h = _Handler()
    logger.addHandler(h)
    try:
        async with AsyncClient(app=app, base_url="http://testserver") as ac:
            r = await ac.get("/healthz")
        assert r.status_code == 200
        assert seen.get("ok") is True
    finally:
        logger.removeHandler(h)
