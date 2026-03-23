from __future__ import annotations

import pytest
from sqlalchemy import text

from public_api.schemas.error import ErrorCode

pytestmark = [pytest.mark.integration]


class TestResolveE2E:
    @pytest.mark.anyio
    async def test_happy_path_includes_glossary(self, client, seed_glossary):
        payload = {
            "text": "The Metropolitan Police Service (MPS) operates in London.",
            "options": {"include_glossary_enrichment": True},
        }

        response = await client.post("/v1/resolve", json=payload)

        assert response.status_code == 200
        body = response.json()

        assert body["meta"]["input_chars"] == len(payload["text"])
        assert body["meta"]["processing_ms"] >= 0
        assert response.headers["X-Request-Id"]
        assert int(response.headers["X-Input-Bytes"]) > 0
        assert int(response.headers["X-Body-Limit-Bytes"]) > 0

        block = body["acronyms"][0]
        assert block["acronym"] == "MPS"
        assert block["glossary"] is not None
        assert any(
            "Metropolitan Police Service" in match["definition"]
            for match in block["glossary"]["matches"]
        )

    @pytest.mark.anyio
    async def test_offset_slice_matches_acronym(self, client, seed_glossary):
        text = "The Metropolitan Police Service (MPS) operates in London."

        response = await client.post("/v1/resolve", json={"text": text})

        assert response.status_code == 200
        block = response.json()["acronyms"][0]
        start = block["first_occurrence"]["start"]
        end = block["first_occurrence"]["end"]

        assert text[start:end] == "MPS"

    @pytest.mark.anyio
    async def test_empty_text_422(self, client):
        response = await client.post("/v1/resolve", json={"text": "   "})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == ErrorCode.UNPROCESSABLE_ENTITY

    @pytest.mark.anyio
    async def test_payload_too_large_413(self, client):
        response = await client.post("/v1/resolve", json={"text": "X" * 100_001})

        assert response.status_code == 413
        assert response.json()["error"]["code"] == ErrorCode.PAYLOAD_TOO_LARGE

    @pytest.mark.anyio
    async def test_inactive_glossary_removes_enrichment_but_not_extraction(
        self,
        client,
        session_factory,
        seed_glossary,
    ):
        with session_factory() as s:
            s.execute(
                text(
                    """
                    UPDATE unacronym.glossary_meanings
                    SET is_active = false
                    WHERE definition = 'Metropolitan Police Service.'
                    """
                )
            )
            s.commit()

        response = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "options": {"include_glossary_enrichment": True},
            },
        )

        assert response.status_code == 200
        block = response.json()["acronyms"][0]

        assert block["acronym"] == "MPS"
        assert block["definitions"]
        assert block["glossary"] is None
        assert block["selection"]["filtered_inactive_count"] == 1
