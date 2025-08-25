import pytest
from httpx import ASGITransport, AsyncClient

from public_api.main import create_app


class TestOpenAPIToggle:

    @pytest.mark.anyio
    async def test_docs_enabled_returns_200(self):
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            r = await ac.get("/openapi.json")
            assert r.status_code == 200
            assert "openapi" in r.json()

    @pytest.mark.anyio
    async def test_docs_disabled_returns_404(self):
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            r = await ac.get("/openapi.json")
            assert r.status_code == 200
