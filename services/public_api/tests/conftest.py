from __future__ import annotations

import pytest
from httpx import AsyncClient
from public_api.main import create_app


@pytest.fixture()
async def client():
    app = create_app()
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac
