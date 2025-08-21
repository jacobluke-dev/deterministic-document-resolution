import pytest


class TestResolveContract:

    @pytest.mark.anyio
    async def test_resolve_positive_shape(self, client):
        payload = {
            "text": "The Metropolitan Police Service (MPS) operates in London.",
            "options": {"return_occurrences": True, "include_glossary_enrichment": True},
        }
        r = await client.post("/v1/resolve", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "acronyms" in body and "meta" in body
        assert body["meta"]["model_version"].startswith("plainera-core@")
        # Headers
        assert r.headers.get("X-Request-Id")
        assert int(r.headers.get("X-Input-Bytes", "0")) > 0
        assert int(r.headers.get("X-Body-Limit-Bytes", "0")) > 0

    @pytest.mark.anyio
    async def test_resolve_empty_text_semantic_422(self, client):
        r = await client.post("/v1/resolve", json={"text": "   "})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "UNPROCESSABLE_ENTITY"

    @pytest.mark.anyio
    async def test_resolve_invalid_option_bounds_400(self, client):
        r = await client.post("/v1/resolve", json={"text": "x", "options": {"max_definitions_per_acronym": 999}})
        assert r.status_code in (400, 422)
