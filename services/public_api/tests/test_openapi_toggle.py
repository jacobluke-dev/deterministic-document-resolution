import pytest
from httpx import AsyncClient
from public_api.core.settings import AppSettings
from public_api.main import create_app


@pytest.mark.asyncio
async def test_docs_disabled_returns_404():
    app = create_app(AppSettings(ENABLE_DOCS=False))
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        r = await ac.get("/docs")
        assert r.status_code == 404
        r2 = await ac.get("/openapi.json")
        assert r2.status_code == 404
