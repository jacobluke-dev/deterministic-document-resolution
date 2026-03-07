from typing import Any

import pytest
from httpx import Response
from public_api.api.routers import resolve as resolve_mod
from public_api.core import deps_auth as deps_auth_mod
from public_api.core.auth.api_keys import Principal
from public_api.db.models import GlossaryAcronym, GlossaryMeaning
from public_api.schemas.error import ErrorCode


def _get_fastapi_app_from_client(client):
    transport = getattr(client, "_transport", None)
    if transport is None:
        raise RuntimeError("client._transport missing")
    return getattr(transport, "app", None) or getattr(transport, "_app", None)


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
        assert blocks[0]["glossary"]["matches"]
        assert "Metropolitan Police Service" in blocks[0]["glossary"]["matches"][0]["definition"]

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
        # Force a short timeout on the exact module the handler reads
        resolve_mod.app_settings.REQUEST_TIMEOUT_MS = 500  # 0.5s

        # Patch the pipeline entrypoint used by ResolveService
        import public_api.core.services.resolve_service as rs

        def slow_detect_and_extract(*args, **kwargs):
            import time
            time.sleep(2.0)  # longer than 0.5s
            # never reached, but keeps signature expectations sane
            raise RuntimeError("should have timed out")

        monkeypatch.setattr(rs, "detect_and_extract", slow_detect_and_extract, raising=True)

        r = await client.post("/v1/resolve", json={"text": "Foo (BAR)"})
        assert r.status_code == 503
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

        from public_api.core import deps as deps_mod
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
