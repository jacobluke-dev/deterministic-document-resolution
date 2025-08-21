import pytest


@pytest.mark.anyio
async def test_openapi_validates(client):
    # Ensure /openapi.json is present and has our bits
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"] == "Unacronym API"
    assert any(s.get("url", "").startswith("https://") for s in spec.get("servers", []))
    # Operation present
    paths = spec["paths"]
    assert "/v1/resolve" in paths
    post = paths["/v1/resolve"]["post"]
    assert "X-Body-Limit-Bytes" in post["responses"]["200"]["headers"]
    # Security stub
    assert spec["components"]["securitySchemes"]["apiKey"]["name"] == "X-API-Key"
