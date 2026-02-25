import json
from pathlib import Path
from typing import Any

import pytest


SNAP_DIR = Path(__file__).parent / "_snapshots"

SNAP_FILE = SNAP_DIR / "openapi.resolve.v1.json"


def _get_fastapi_app_from_client(client):
    transport = getattr(client, "_transport", None)
    if transport is None:
        raise RuntimeError("client._transport missing")
    return getattr(transport, "app", None) or getattr(transport, "_app", None)


def _extract_resolve_spec(openapi: dict[str, Any]) -> dict[str, Any]:
    paths = openapi.get("paths", {})
    resolve = paths.get("/v1/resolve")
    if resolve is None:
        raise AssertionError("Missing /v1/resolve in OpenAPI paths")

    # Snapshot a focused contract slice: endpoint + components.
    # This keeps noise down while still protecting SDK-relevant schema.
    return {
        "openapi": openapi.get("openapi"),
        "info": openapi.get("info"),
        "paths": {"/v1/resolve": resolve},
        "components": openapi.get("components", {}),
    }


@pytest.mark.anyio
async def test_openapi_snapshot(client):
    app = _get_fastapi_app_from_client(client)

    # Do NOT rely on /openapi.json existing; it is disabled when ENABLE_DOCS=false.
    spec = app.openapi()
    slim = _extract_resolve_spec(spec)

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    print("SNAP_DIR:", SNAP_DIR)
    print("SNAP_FILE:", SNAP_FILE)

    if not SNAP_FILE.exists():
        SNAP_FILE.write_text(json.dumps(slim, indent=2, sort_keys=True), encoding="utf-8")
        pytest.fail(f"Created snapshot at {SNAP_FILE}. Re-run tests.")

    expected = json.loads(SNAP_FILE.read_text(encoding="utf-8"))

    # Normalise ordering for stable diffs.
    assert json.loads(json.dumps(slim, sort_keys=True)) == json.loads(json.dumps(expected, sort_keys=True))
