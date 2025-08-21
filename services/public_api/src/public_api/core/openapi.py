
from fastapi import FastAPI


def apply_openapi_overrides(app: FastAPI) -> None:
    original = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = original()
        # Servers
        schema["servers"] = [
            {"url": "https://api.unacronym.com", "description": "Production"},
            {"url": "https://staging.api.unacronym.com", "description": "Staging"},
        ]
        # Tags (ensure presence/order)
        schema["tags"] = [
            {"name": "Resolve", "description": "Resolve acronyms in text."},
            {"name": "Health", "description": "Liveness/readiness probes."},
        ]
        # SecuritySchemes (stub; enforcement added later in Story 2.6)
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["apiKey"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication (not enforced until Epic 5/Story 2.6).",
        }
        # Apply at operation level (global)
        schema["security"] = [{"apiKey": []}]
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore
