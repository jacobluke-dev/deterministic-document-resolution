import pytest

HEADER = "X-Request-ID"

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

    @pytest.mark.asyncio
    async def test_generates_request_id_when_absent(self, client):
        r = await client.get("/healthz")
        assert HEADER in r.headers
        assert len(r.headers[HEADER]) > 0

    @pytest.mark.asyncio
    async def test_echoes_request_id_when_provided(self, client):
        rid = "test-123"
        r = await client.get("/healthz", headers={HEADER: rid})
        assert r.headers[HEADER] == rid
