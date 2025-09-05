import pytest
from httpx import ASGITransport, AsyncClient
from public_api.core.settings import AppSettings
from public_api.main import create_app


@pytest.mark.anyio
async def test_body_size_limit_returns_413():
    app = create_app(settings=AppSettings(MAX_BODY_BYTES=1))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        r = await ac.post("/v1/resolve", json={"text": "xx"})
        assert r.status_code == 413
