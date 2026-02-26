from __future__ import annotations


from fastapi import Request

from public_api.core.settings import AppSettings


def get_settings(request: Request) -> AppSettings:
    """Return the per-app settings stored on `app.state`."""
    return request.app.state.settings  # type: ignore[attr-defined]
