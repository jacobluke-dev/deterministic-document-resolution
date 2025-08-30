# services/public_api/src/public_api/api/openapi.py
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, TypeAlias

from src.public_api.schemas.error import ErrorResponse

COMMON_ERROR_RESPONSES: Mapping[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse},  # Bad Request (invalid options, malformed JSON, etc.)
    413: {"model": ErrorResponse},  # Payload Too Large (text length > allowed)
    415: {"model": ErrorResponse},  # Unsupported Media Type (wrong Content-Type / encoding)
    422: {"model": ErrorResponse},  # Unprocessable Entity (semantic validation, e.g. empty text)
    429: {"model": ErrorResponse},  # TODO Too Many Requests (rate limiting — future Epic 5)
    500: {"model": ErrorResponse},  # Internal Server Error (unexpected exception)
    503: {"model": ErrorResponse},  # Service Unavailable (timeout or concurrency overload)
}

# Success headers spec you can attach to 200/201/etc.
SUCCESS_HEADERS_SPEC: dict[str, Any] = {
    "X-Request-Id": {"description": "Echoed or generated correlation id.", "schema": {"type": "string"}},
    "X-Input-Bytes": {"description": "Parsed request body size in bytes.", "schema": {"type": "integer"}},
    "X-Body-Limit-Bytes": {"description": "Current configured body size limit.", "schema": {"type": "integer"}},
    "X-RateLimit-Limit": {"description": "Requests allowed in window.", "schema": {"type": "integer"}},
    "X-RateLimit-Remaining": {"description": "Remaining in current window.", "schema": {"type": "integer"}},
    "X-RateLimit-Reset": {"description": "Epoch seconds until window resets.", "schema": {"type": "integer"}},
}

ResponsesSpec: TypeAlias = dict[int | str, dict[str, Any]]

def build_responses(
    *,
    success_status: int = 200,
    success_headers: Mapping[str, Any] = SUCCESS_HEADERS_SPEC,
    extra_errors: Mapping[int | str, dict[str, Any]] | None = None,
) -> ResponsesSpec:
    """Compose a responses dict for FastAPI route decorators.

    Args:
        success_status: HTTP status code for the success response (e.g. 200/201).
        success_headers: Header schema to document on success responses.
        extra_errors: Additional error codes to merge with COMMON_ERROR_RESPONSES.

    Returns:
        A fresh dict suitable for `responses=...` in a route decorator.
    """
    resp: MutableMapping[int | str, dict[str, Any]] = dict(COMMON_ERROR_RESPONSES)
    # Add/override extra errors if provided
    if extra_errors:
        resp.update(extra_errors)
    # Add success entry with headers
    resp[success_status] = {"headers": dict(success_headers or {})}
    return dict(resp)
