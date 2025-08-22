import pytest

HEADER = "X-Request-ID"

@pytest.mark.asyncio
async def test_generates_request_id_when_absent(client):
    r = await client.get("/healthz")
    assert HEADER in r.headers
    assert len(r.headers[HEADER]) > 0

@pytest.mark.asyncio
async def test_echoes_request_id_when_provided(client):
    rid = "test-123"
    r = await client.get("/healthz", headers={HEADER: rid})
    assert r.headers[HEADER] == rid
