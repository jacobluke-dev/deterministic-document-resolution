import pytest


class TestHealth:

    @pytest.mark.asyncio
    async def test_healthz(self, client):
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_readyz(self, client):
        r = await client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
