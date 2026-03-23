import pytest

pytestmark = [pytest.mark.integration]


class TestGlossaryE2E:
    @pytest.mark.anyio
    async def test_glossary_enrichment_present_when_enabled(self, client, seed_glossary):
        payload = {
            "text": "The Metropolitan Police Service (MPS) operates in London.",
            "options": {"include_glossary_enrichment": True},
        }

        response = await client.post("/v1/resolve", json=payload)

        assert response.status_code == 200, response.text
        block = response.json()["acronyms"][0]

        assert block["acronym"] == "MPS"
        assert block["glossary"] is not None
        assert block["glossary"]["matches"]
        assert any(
            "Metropolitan Police Service" in match["definition"]
            for match in block["glossary"]["matches"]
        )

    @pytest.mark.anyio
    async def test_glossary_block_omitted_when_enrichment_disabled(self, client, seed_glossary):
        response = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "options": {"include_glossary_enrichment": False},
            },
        )

        assert response.status_code == 200, response.text
        block = response.json()["acronyms"][0]

        assert block["acronym"] == "MPS"
        assert block["glossary"] is None

    @pytest.mark.anyio
    async def test_inactive_glossary_removes_glossary_but_preserves_extraction(
        self,
        client,
        session_factory,
        seed_glossary,
    ):
        from sqlalchemy import text

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

        assert response.status_code == 200, response.text
        block = response.json()["acronyms"][0]

        assert block["acronym"] == "MPS"
        assert block["definitions"]
        assert block["glossary"] is None
        assert block["selection"]["filtered_inactive_count"] == 1
