from __future__ import annotations

from typing import Any

import pytest
from public_api.core.pipelines import DefinedTermsPipelineExecutor
from public_api.schemas.error import ErrorCode
from sqlalchemy import text

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
                    UPDATE document_resolution.glossary_meanings
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


class TestResolveE2EPartialSuccess:
    @staticmethod
    def _text_with_all_patterns() -> str:
        return (
            'In this Agreement, "Services" means the consulting services provided by the '
            "Metropolitan Police Service (MPS). The MPS shall provide the Services in "
            "accordance with Section 2 and Schedule 1."
        )

    @pytest.mark.anyio
    async def test_partial_success_returns_completed_sections_and_pipeline_error(
        self,
        client,
        seed_glossary,
        monkeypatch,
    ):
        async def broken_execute(self, *, request, opts, lang, resolution_mode):
            raise RuntimeError("defined-term pipeline exploded")

        monkeypatch.setattr(
            DefinedTermsPipelineExecutor,
            "execute",
            broken_execute,
        )

        response = await client.post(
            "/v1/resolve",
            json={
                "text": self._text_with_all_patterns(),
                "targets": ["acronyms", "defined_terms", "structural_references"],
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()

        assert body["acronyms"] != []
        assert body["defined_terms"] == []
        assert body["structural_references"] != []

        assert body["orchestration"]["requested"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["completed"] == [
            "acronyms",
            "structural_references",
        ]
        assert body["orchestration"]["failed"] == ["defined_terms"]

        assert len(body["errors"]) == 1
        assert body["errors"][0]["pipeline"] == "defined_terms"
        assert body["errors"][0]["message"]
        assert body["errors"][0]["code"]


class TestResolveE2ETargetCombinationMatrix:
    @staticmethod
    def _text_with_all_patterns() -> str:
        return (
            'In this Agreement, "Services" means the consulting services provided by the '
            "Metropolitan Police Service (MPS). The MPS shall provide the Services in "
            "accordance with Section 2 and Schedule 1."
        )

    @staticmethod
    def _assert_orchestration(
        body: dict[str, Any],
        *,
        requested: list[str],
        completed: list[str],
        failed: list[str],
    ) -> None:
        assert body["orchestration"]["requested"] == requested
        assert body["orchestration"]["completed"] == completed
        assert body["orchestration"]["failed"] == failed

    @staticmethod
    def _assert_section_population(
        body: dict[str, Any],
        *,
        acronyms: bool,
        defined_terms: bool,
        structural_references: bool,
    ) -> None:
        assert isinstance(body["acronyms"], list)
        assert isinstance(body["defined_terms"], list)
        assert isinstance(body["structural_references"], list)

        assert (body["acronyms"] != []) is acronyms
        assert (body["defined_terms"] != []) is defined_terms
        assert (body["structural_references"] != []) is structural_references

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("targets", "expected"),
        [
            (
                ["acronyms", "defined_terms"],
                {
                    "acronyms": True,
                    "defined_terms": True,
                    "structural_references": False,
                },
            ),
            (
                ["acronyms", "structural_references"],
                {
                    "acronyms": True,
                    "defined_terms": False,
                    "structural_references": True,
                },
            ),
            (
                ["defined_terms", "structural_references"],
                {
                    "acronyms": False,
                    "defined_terms": True,
                    "structural_references": True,
                },
            ),
        ],
    )
    async def test_pairwise_target_combinations_return_expected_sections_and_metadata(
        self,
        client,
        seed_glossary,
        targets,
        expected,
    ):
        response = await client.post(
            "/v1/resolve",
            json={
                "text": self._text_with_all_patterns(),
                "targets": targets,
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()

        self._assert_section_population(
            body,
            acronyms=expected["acronyms"],
            defined_terms=expected["defined_terms"],
            structural_references=expected["structural_references"],
        )
        self._assert_orchestration(
            body,
            requested=targets,
            completed=targets,
            failed=[],
        )
        assert body["errors"] == []


class TestResolveE2EDeterministicCombinedResponse:
    @staticmethod
    def _stable_json(body: dict[str, Any]) -> dict[str, Any]:
        stable = dict(body)
        meta = dict(stable.get("meta", {}))
        meta.pop("processing_ms", None)
        stable["meta"] = meta
        return stable

    @staticmethod
    def _text_with_all_patterns() -> str:
        return (
            'In this Agreement, "Services" means the consulting services provided by the '
            "Metropolitan Police Service (MPS). The MPS shall provide the Services in "
            "accordance with Section 2 and Schedule 1."
        )

    @pytest.mark.anyio
    async def test_all_targets_response_is_deterministic(self, client, seed_glossary):
        payload = {
            "text": self._text_with_all_patterns(),
            "targets": ["acronyms", "defined_terms", "structural_references"],
        }

        response_1 = await client.post("/v1/resolve", json=payload)
        response_2 = await client.post("/v1/resolve", json=payload)

        assert response_1.status_code == 200, response_1.text
        assert response_2.status_code == 200, response_2.text

        assert self._stable_json(response_1.json()) == self._stable_json(response_2.json())
