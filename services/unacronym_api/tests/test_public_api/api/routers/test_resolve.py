from typing import Any

import pytest
from httpx import Response
from public_api.api.routers import resolve as resolve_mod
from public_api.core.di import deps_auth as deps_auth_mod
from public_api.core.auth.api_keys import Principal
from public_api.db.models import GlossaryAcronym, GlossaryMeaning
from public_api.schemas.error import ErrorCode


def _get_fastapi_app_from_client(client):
    transport = getattr(client, "_transport", None)
    if transport is None:
        raise RuntimeError("client._transport missing")
    return getattr(transport, "app", None) or getattr(transport, "_app", None)


_ORIGINAL_REQUEST_TIMEOUT_MS = resolve_mod.app_settings.REQUEST_TIMEOUT_MS


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setattr(
        resolve_mod.app_settings,
        "REQUEST_TIMEOUT_MS",
        _ORIGINAL_REQUEST_TIMEOUT_MS,
    )


class TestV1Resolve:
    @pytest.fixture(autouse=True)
    def seed_for_this_class(self, session_factory):
        # minimal deterministic seed used by multiple tests here
        with session_factory() as s:
            def _ensure(acr: str, definition: str) -> None:
                norm = acr.lower()

                ga = (
                    s.query(GlossaryAcronym)
                    .filter(GlossaryAcronym.tenant_id.is_(None))
                    .filter(GlossaryAcronym.normalized == norm)
                    .first()
                )

                if ga is None:
                    ga = GlossaryAcronym(
                        tenant_id=None,
                        acronym=acr,
                        normalized=norm,
                        is_active=True,
                    )
                    s.add(ga)
                    s.flush()
                else:
                    # keep canonical surface stable for tests
                    ga.acronym = acr
                    ga.is_active = True

                gm = (
                    s.query(GlossaryMeaning)
                    .filter(GlossaryMeaning.acronym_id == ga.id)
                    .filter(GlossaryMeaning.domain == "general")
                    .first()
                )

                if gm is None:
                    s.add(
                        GlossaryMeaning(
                            acronym_id=ga.id,
                            definition=definition,
                            domain="general",
                            provenance="test",
                            is_active=True,
                        )
                    )
                else:
                    gm.definition = definition
                    gm.provenance = "test"
                    gm.is_active = True

            _ensure("MPS", "Metropolitan Police Service.")
            _ensure("ABC", "Alpha Beta Charlie.")

            s.commit()
        yield

    @pytest.mark.anyio
    async def test_happy_path_includes_glossary(self, client):
        payload = {
            "text": "The Metropolitan Police Service (MPS) operates in London.",
            "options": {"include_glossary_enrichment": True},
        }
        r = await client.post("/v1/resolve", json=payload)
        assert r.status_code == 200

        body = r.json()
        assert "acronyms" in body and "meta" in body
        assert body["meta"]["input_chars"] == len(payload["text"])

        # headers
        assert r.headers.get("X-Request-Id")
        assert int(r.headers["X-Input-Bytes"]) > 0
        assert int(r.headers["X-Body-Limit-Bytes"]) > 0

        # glossary
        blocks = body["acronyms"]
        assert len(blocks) == 1
        assert blocks[0]["acronym"] == "MPS"
        assert blocks[0]["glossary"] is not None
        assert blocks[0]["glossary"] is not None
        assert "matches" in blocks[0]["glossary"]
        assert len(blocks[0]["glossary"]["matches"]) >= 1
        assert any(
            "Metropolitan Police Service" in m["definition"]
            for m in blocks[0]["glossary"]["matches"]
        )

    @pytest.mark.anyio
    async def test_empty_text_422(self, client):
        r = await client.post("/v1/resolve", json={"text": "   "})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == ErrorCode.UNPROCESSABLE_ENTITY

    @pytest.mark.anyio
    async def test_invalid_option_bounds(self, client):
        r = await client.post(
            "/v1/resolve",
            json={"text": "x", "options": {"max_definitions_per_acronym": 999}},
        )
        assert r.status_code in (400, 422)

    @pytest.mark.anyio
    async def test_text_too_large(self, client):
        big_text = "X" * 100_001
        r = await client.post("/v1/resolve", json={"text": big_text})
        assert r.status_code == 413
        err = r.json()["error"]
        assert err["code"] == ErrorCode.PAYLOAD_TOO_LARGE

    @pytest.mark.anyio
    async def test_timeout(self, client, monkeypatch):
        monkeypatch.setattr(resolve_mod.app_settings, "REQUEST_TIMEOUT_MS", 500)

        # Patch the pipeline entrypoint used by ResolveService
        import public_api.core.services.resolve_service as rs

        async def slow_or_timeout(*args, **kwargs):
            raise rs.ResolveError(
                http_status=503,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Request timed out.",
                details={"reason": "TIMEOUT", "timeout_ms": 500},
            )

        monkeypatch.setattr(
            rs.Orchestrator,
            "execute_orchestration_request",
            slow_or_timeout,
        )

        r = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert body["error"]["details"]["timeout_ms"] == 500

    @staticmethod
    def _stable_json(resp: Response) -> dict[str, Any]:
        body = resp.json()
        meta = dict(body.get("meta", {}))
        # Strip fields that can vary run-to-run
        meta.pop("processing_ms", None)
        body["meta"] = meta
        return body

    @pytest.mark.anyio
    async def test_deterministic_output(self, client):
        payload = {
            "text": "Alpha (ABC). Another (ABC).",
            "options": {"include_glossary_enrichment": False},
        }
        r1 = await client.post("/v1/resolve", json=payload)
        r2 = await client.post("/v1/resolve", json=payload)

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert self._stable_json(r1) == self._stable_json(r2)

    @pytest.mark.anyio
    async def test_overloaded(self, client):

        from public_api.core.di import deps as deps_mod
        class DummySemaphore:
            def locked(self):
                return True

        app = _get_fastapi_app_from_client(client)
        app.dependency_overrides[deps_mod.get_semaphore] = lambda: DummySemaphore()
        app.dependency_overrides[deps_auth_mod.require_api_key] = lambda: Principal(
            key_id=1, prefix="test", user_id=None, scopes=()
        )
        try:
            r = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})
            assert r.status_code == 503
            assert r.json()["error"]["details"]["reason"] == "OVERLOADED"
        finally:
            app.dependency_overrides.pop(deps_mod.get_semaphore, None)

    @pytest.mark.anyio
    async def test_ios_pick_surfaces_as_definition(self, client):
        r = await client.post("/v1/resolve",
                              json={"text": "The new iPhone Operating system (iOS) was developed by Apple."})
        assert r.status_code == 200
        blk = r.json()["acronyms"][0]
        assert blk["acronym"] == "iOS"
        assert any("iPhone Operating system" in d["text"] for d in blk["definitions"])

    @pytest.mark.anyio
    async def test_happy_path_includes_resolution_metadata(self, client):
        payload = {
            "text": "The Metropolitan Police Service (MPS) operates in London.",
            "options": {"include_glossary_enrichment": True},
        }

        r = await client.post("/v1/resolve", json=payload)
        assert r.status_code == 200, r.text

        block = r.json()["acronyms"][0]

        assert "candidates" in block
        assert "selected" in block
        assert "conflict" in block
        assert "conflict_count" in block
        assert "selection" in block

        assert isinstance(block["candidates"], list)
        assert block["selected"] is not None
        assert isinstance(block["conflict"], bool)
        assert isinstance(block["conflict_count"], int)

        assert block["candidates"]
        assert block["selected"]["definition"] == block["candidates"][0]["definition"]
        assert "reason" in block["selected"]

    @pytest.mark.anyio
    async def test_single_candidate_sets_conflict_false(self, client):
        payload = {
            "text": "The Metropolitan Police Service (MPS) operates in London.",
            "options": {"include_glossary_enrichment": True},
        }

        r = await client.post("/v1/resolve", json=payload)
        assert r.status_code == 200, r.text

        block = r.json()["acronyms"][0]

        assert block["conflict"] is False or block["conflict_count"] == 1
        assert block["conflict_count"] >= 1

    @pytest.mark.anyio
    async def test_multi_meaning_acronym_returns_conflict_metadata(self, client, session_factory):
        with session_factory() as s:
            ga = GlossaryAcronym(
                tenant_id=None,
                acronym="GP",
                normalized="gp",
                is_active=True,
            )
            s.add(ga)
            s.flush()

            s.add_all(
                [
                    GlossaryMeaning(
                        acronym_id=ga.id,
                        definition="General Practitioner",
                        domain="medical",
                        provenance="test",
                        is_active=True,
                    ),
                    GlossaryMeaning(
                        acronym_id=ga.id,
                        definition="General Partner",
                        domain="finance",
                        provenance="test",
                        is_active=True,
                    ),
                ]
            )
            s.commit()

        r = await client.post(
            "/v1/resolve",
            json={"text": "The General Practitioner (GP) signed the report."},
        )
        assert r.status_code == 200, r.text

        block = next(b for b in r.json()["acronyms"] if b["acronym"] == "GP")

        assert len(block["candidates"]) >= 2
        assert block["conflict"] is True
        assert block["conflict_count"] >= 2
        assert block["selected"]["reason"] in {
            "in_document_definition",
            "single_candidate",
            "highest_score",
            "fallback_general",
            "inactive_filtered",
        }

    @pytest.mark.anyio
    async def test_selected_candidate_is_first(self, client):
        r = await client.post(
            "/v1/resolve",
            json={"text": "The Metropolitan Police Service (MPS) operates in London."},
        )
        assert r.status_code == 200, r.text

        block = r.json()["acronyms"][0]
        assert block["selected"]["definition"] == block["candidates"][0]["definition"]
        assert block["selected"]["domain"] == block["candidates"][0]["domain"]

    @pytest.mark.anyio
    async def test_glossary_block_includes_multiple_matches_for_multi_meaning_acronym(self, client, session_factory):
        with session_factory() as s:
            ga = (
                s.query(GlossaryAcronym)
                .filter(GlossaryAcronym.tenant_id.is_(None))
                .filter(GlossaryAcronym.normalized == "gp")
                .first()
            )

            if ga is None:
                ga = GlossaryAcronym(
                    tenant_id=None,
                    acronym="GP",
                    normalized="gp",
                    is_active=True,
                )
                s.add(ga)
                s.flush()
            else:
                ga.acronym = "GP"
                ga.is_active = True

            def _upsert_meaning(definition: str, domain: str) -> None:
                gm = (
                    s.query(GlossaryMeaning)
                    .filter(GlossaryMeaning.acronym_id == ga.id)
                    .filter(GlossaryMeaning.domain == domain)
                    .first()
                )
                if gm is None:
                    s.add(
                        GlossaryMeaning(
                            acronym_id=ga.id,
                            definition=definition,
                            domain=domain,
                            provenance="test",
                            is_active=True,
                        )
                    )
                else:
                    gm.definition = definition
                    gm.provenance = "test"
                    gm.is_active = True

            _upsert_meaning("General Practitioner", "medical")
            _upsert_meaning("General Partner", "finance")
            s.commit()

        r = await client.post(
            "/v1/resolve",
            json={
                "text": "General Practitioner (GP). The GP signed the report.",
                "options": {"include_glossary_enrichment": True},
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()
        assert any(b["acronym"] == "GP" for b in body["acronyms"]), body
        block = next(b for b in body["acronyms"] if b["acronym"] == "GP")

        assert block["glossary"] is not None
        matches = block["glossary"]["matches"]
        assert len(matches) >= 2

        defs = {m["definition"] for m in matches}
        assert "General Practitioner" in defs
        assert "General Partner" in defs

        domains = {m["domain"] for m in matches}
        assert "medical" in domains
        assert "finance" in domains

    @pytest.mark.anyio
    async def test_glossary_block_omitted_when_enrichment_disabled(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "options": {"include_glossary_enrichment": False},
            },
        )
        assert r.status_code == 200, r.text

        block = r.json()["acronyms"][0]
        assert "glossary" not in block or block["glossary"] is None


class TestV1ResolveTargetSelection:
    @pytest.mark.anyio
    async def test_defined_terms_only_returns_empty_section_when_no_terms_found(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["defined_terms"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert body["defined_terms"] == []
        assert body["acronyms"] == []
        assert body["structural_references"] == []

        assert body["orchestration"]["requested"] == ["defined_terms"]
        assert body["orchestration"]["completed"] == ["defined_terms"]
        assert body["orchestration"]["failed"] == []
        assert body["errors"] == []

    @pytest.mark.anyio
    async def test_structural_only_returns_empty_section_when_no_references_found(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["structural_references"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert body["structural_references"] == []
        assert body["acronyms"] == []
        assert body["defined_terms"] == []

        assert body["orchestration"]["requested"] == ["structural_references"]
        assert body["orchestration"]["completed"] == ["structural_references"]
        assert body["orchestration"]["failed"] == []
        assert body["errors"] == []

    @pytest.mark.anyio
    async def test_acronyms_only_does_not_depend_on_other_pipeline_payload_shapes(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["acronyms"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert len(body["acronyms"]) == 1
        assert body["acronyms"][0]["acronym"] == "MPS"
        assert body["defined_terms"] == []
        assert body["structural_references"] == []

        assert body["orchestration"]["requested"] == ["acronyms"]
        assert body["orchestration"]["completed"] == ["acronyms"]
        assert body["orchestration"]["failed"] == []
        assert body["errors"] == []

    @pytest.mark.anyio
    async def test_all_targets_allow_mixed_non_empty_and_empty_sections(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["acronyms", "defined_terms", "structural_references"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert len(body["acronyms"]) == 1
        assert body["acronyms"][0]["acronym"] == "MPS"
        assert body["defined_terms"] == []
        assert body["structural_references"] == []

        assert body["orchestration"]["requested"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["completed"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["failed"] == []
        assert body["errors"] == []

    @pytest.mark.anyio
    async def test_duplicate_targets_are_deduplicated_preserving_request_order(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": [
                    "defined_terms",
                    "acronyms",
                    "defined_terms",
                    "structural_references",
                    "acronyms",
                ],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert body["orchestration"]["requested"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["completed"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["failed"] == []

    @pytest.mark.anyio
    async def test_defined_terms_target_returns_non_empty_section_when_terms_present(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": (
                'This Agreement is made between the Supplier and the Customer. '
                '"Services" means the consulting services described in Schedule 1. '
                'The Supplier shall provide the Services to the Customer.'
                ),
                "targets": ["defined_terms"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()
        assert body["defined_terms"] != []
        assert body["acronyms"] == []
        assert body["structural_references"] == []
        assert body["orchestration"]["requested"] == ["defined_terms"]
        assert body["orchestration"]["completed"] == ["defined_terms"]
        assert body["orchestration"]["failed"] == []

    @pytest.mark.anyio
    async def test_structural_target_returns_non_empty_section_when_references_present(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": (
                    'In this Agreement, "Services" means the consulting services described in '
                    "Section 2. The Supplier shall provide the Services in accordance with "
                    "Section 3 and Schedule 1."
                ),
                "targets": ["structural_references"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()
        assert body["structural_references"] != []
        assert body["acronyms"] == []
        assert body["defined_terms"] == []
        assert body["orchestration"]["requested"] == ["structural_references"]
        assert body["orchestration"]["completed"] == ["structural_references"]
        assert body["orchestration"]["failed"] == []

    @pytest.mark.anyio
    async def test_all_targets_return_mixed_non_empty_sections_when_text_contains_all_patterns(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": (
                    'In this Agreement, "Services" means the consulting services provided by the '
                    "Metropolitan Police Service (MPS). The MPS shall provide the Services in "
                    "accordance with Section 2 and Schedule 1."
                ),
                "targets": ["acronyms", "defined_terms", "structural_references"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()
        assert body["acronyms"] != []
        assert body["defined_terms"] != []
        assert body["structural_references"] != []

        assert body["orchestration"]["requested"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["completed"] == [
            "acronyms",
            "defined_terms",
            "structural_references",
        ]
        assert body["orchestration"]["failed"] == []


class TestV1ResolveResponseShapes:
    @pytest.mark.anyio
    async def test_all_targets_response_includes_expected_section_shapes(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": (
                    'In this Agreement, "Services" means the consulting services provided by the '
                    "Metropolitan Police Service (MPS). The MPS shall provide the Services in "
                    "accordance with Section 2 and Schedule 1."
                ),
                "targets": ["acronyms", "defined_terms", "structural_references"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert isinstance(body["acronyms"], list)
        assert isinstance(body["defined_terms"], list)
        assert isinstance(body["structural_references"], list)
        assert isinstance(body["errors"], list)

        acronym = body["acronyms"][0]
        assert acronym["acronym"] == "MPS"
        assert "first_occurrence" in acronym
        assert "occurrences" in acronym
        assert "definitions" in acronym
        assert "candidates" in acronym
        assert "selected" in acronym
        assert "conflict" in acronym
        assert "selection" in acronym

        defined_term = body["defined_terms"][0]
        assert "occurrence_span" in defined_term
        assert "term" in defined_term
        assert "normalized_key" in defined_term
        assert "chosen_meaning_id" in defined_term
        assert "chosen_definition_span" in defined_term
        assert "resolution_method" in defined_term
        assert "resolved" in defined_term
        assert "candidate_scores" in defined_term
        assert "chosen_meaning" in defined_term

        structural = body["structural_references"][0]
        assert "kind" in structural
        assert "label" in structural
        assert "normalized_key" in structural
        assert "reference_span" in structural
        assert "resolved" in structural

    @pytest.mark.anyio
    async def test_acronym_block_has_expected_nested_shape(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["acronyms"],
            },
        )
        assert r.status_code == 200, r.text

        block = r.json()["acronyms"][0]

        assert block["acronym"] == "MPS"

        first_occurrence = block["first_occurrence"]
        assert set(first_occurrence) >= {"start", "end"}

        assert isinstance(block["occurrences"], list)
        assert block["occurrences"]
        assert set(block["occurrences"][0]) >= {"start", "end"}

        assert isinstance(block["definitions"], list)
        assert isinstance(block["candidates"], list)

        if block["selected"] is not None:
            assert set(block["selected"]) >= {"definition", "reason"}

        assert set(block["selection"]) >= {"filtered_inactive_count"}

    @pytest.mark.anyio
    async def test_defined_term_block_has_expected_nested_shape(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": (
                    'In this Agreement, "Services" means the consulting services provided by the '
                    "Metropolitan Police Service (MPS). The MPS shall provide the Services."
                ),
                "targets": ["defined_terms"],
            },
        )
        assert r.status_code == 200, r.text

        block = r.json()["defined_terms"][0]

        assert block["term"] == "Services"
        assert block["normalized_key"] == "services"
        assert set(block["occurrence_span"]) >= {"start", "end"}

        if block["chosen_definition_span"] is not None:
            assert set(block["chosen_definition_span"]) >= {"start", "end"}

        assert block["resolution_method"] in {"tier1", "tier2_blend", "unresolved"}
        assert isinstance(block["resolved"], bool)
        assert isinstance(block["candidate_scores"], list)

        if block["candidate_scores"]:
            candidate = block["candidate_scores"][0]
            assert set(candidate) >= {
                "meaning_id",
                "total_score",
                "tier1_score",
                "tier2_score",
                "definition_span",
                "components",
            }

        if block["chosen_meaning"] is not None:
            chosen_meaning = block["chosen_meaning"]
            assert set(chosen_meaning) >= {
                "meaning_id",
                "surface",
                "normalized_key",
                "ordinal",
                "intro_span",
                "definition_span",
                "definition_text",
                "intro_kind",
                "section_path",
                "alias_target_span",
                "alias_target_text",
            }

    @pytest.mark.anyio
    async def test_structural_reference_block_has_expected_nested_shape(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Supplier shall provide the Services in accordance with Section 2 and Schedule 1.",
                "targets": ["structural_references"],
            },
        )
        assert r.status_code == 200, r.text

        block = r.json()["structural_references"][0]

        assert isinstance(block["kind"], str)
        assert isinstance(block["label"], str)
        assert isinstance(block["canonical_label"], str)
        assert isinstance(block["normalized_key"], str)
        assert isinstance(block["canonical_key"], str)
        assert isinstance(block["resolved"], bool)
        assert isinstance(block["strength"], float | int)

        assert set(block["reference_span"]) >= {"start", "end"}

        if block["target_span"] is not None:
            assert set(block["target_span"]) >= {"start", "end"}

    @pytest.mark.anyio
    async def test_response_meta_and_orchestration_have_expected_shape(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["acronyms"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert set(body["meta"]) >= {
            "processing_ms",
            "model_version",
            "input_chars",
            "resolution_mode",
        }
        assert isinstance(body["meta"]["processing_ms"], int)
        assert isinstance(body["meta"]["model_version"], str)
        assert isinstance(body["meta"]["input_chars"], int)
        assert body["meta"]["resolution_mode"] == "domain_priority"

        assert set(body["orchestration"]) == {"requested", "completed", "failed"}
        assert body["orchestration"]["requested"] == ["acronyms"]
        assert body["orchestration"]["completed"] == ["acronyms"]
        assert body["orchestration"]["failed"] == []

    @pytest.mark.anyio
    async def test_response_meta_and_orchestration_have_expected_shape(self, client):
        r = await client.post(
            "/v1/resolve",
            json={
                "text": "The Metropolitan Police Service (MPS) operates in London.",
                "targets": ["acronyms"],
            },
        )
        assert r.status_code == 200, r.text

        body = r.json()

        assert set(body["meta"]) >= {
            "processing_ms",
            "model_version",
            "input_chars",
            "resolution_mode",
        }
        assert isinstance(body["meta"]["processing_ms"], int)
        assert isinstance(body["meta"]["model_version"], str)
        assert isinstance(body["meta"]["input_chars"], int)
        assert body["meta"]["resolution_mode"] == "domain_priority"

        assert set(body["orchestration"]) == {"requested", "completed", "failed"}
        assert body["orchestration"]["requested"] == ["acronyms"]
        assert body["orchestration"]["completed"] == ["acronyms"]
        assert body["orchestration"]["failed"] == []
