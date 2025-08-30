import pytest
from httpx import ASGITransport, AsyncClient
from src.public_api.core.settings import AppSettings
from src.public_api.main import create_app
from starlette.requests import ClientDisconnect


@pytest.mark.anyio
async def test_body_size_limit_returns_413_or_disconnect():
    app = create_app(settings=AppSettings(MAX_BODY_BYTES=1))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://testserver") as ac:
        with pytest.raises(ClientDisconnect):
            await ac.post("/_echo", content=b"xx")
