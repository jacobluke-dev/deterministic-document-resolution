from __future__ import annotations

from typing import Any

from fastapi import Request


def get_settings(request: Request) -> Any:
    """Return the per-app settings stored on `app.state`."""
    return request.app.state.settings
