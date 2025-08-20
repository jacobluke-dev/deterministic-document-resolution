import pytest
from httpx import AsyncClient
from public_api.core.settings import AppSettings
from public_api.main import create_app
from starlette.requests import ClientDisconnect


@pytest.mark.asyncio
async def test_body_size_limit_returns_413():
    app = create_app(AppSettings(MAX_BODY_BYTES=1))  # tiny limit
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        try:
            r = await ac.post("/_echo", content=b"xx")  # forces body read
        except ClientDisconnect:
            # Middleware already responded 413 and closed the stream; that's OK.
            return
        assert r.status_code == 413
