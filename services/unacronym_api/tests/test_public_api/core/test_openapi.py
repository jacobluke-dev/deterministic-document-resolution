from __future__ import annotations

from fastapi import FastAPI
from public_api.core.openapi import apply_openapi_overrides


def test_apply_openapi_overrides_injects_servers_tags_and_security() -> None:
    app = FastAPI()
    apply_openapi_overrides(app)

    schema = app.openapi()

    assert schema["servers"] == [
        {"url": "https://api.unacronym.com", "description": "Production"},
        {"url": "https://staging.api.unacronym.com", "description": "Staging"},
    ]

    assert schema["tags"] == [
        {"name": "Resolve", "description": "Resolve acronyms in text."},
        {"name": "Health", "description": "Liveness/readiness probes."},
    ]

    components = schema["components"]
    assert "securitySchemes" in components
    assert components["securitySchemes"]["apiKey"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key authentication (not enforced until Epic 5/Story 2.6).",
    }

    assert schema["security"] == [{"apiKey": []}]


def test_apply_openapi_overrides_is_idempotent_uses_cached_schema() -> None:
    app = FastAPI()
    apply_openapi_overrides(app)

    schema_1 = app.openapi()
    # second call should return cached schema object
    schema_2 = app.openapi()

    assert schema_1 is schema_2
    assert app.openapi_schema is schema_1
