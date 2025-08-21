from typing import Any, Optional

from fastapi import APIRouter, Header, Response, status
from starlette.responses import JSONResponse

from public_api.schemas.error import ErrorBody, ErrorCode, ErrorResponse
from public_api.schemas.resolve import ResolveOptions, ResolveRequest, ResolveResponse

router = APIRouter(prefix="/v1", tags=["Resolve"])

# Document response headers in OpenAPI
response_headers = {
    "X-Request-Id": {
        "description": "Echoed or generated correlation id.",
        "schema": {"type": "string"},
    },
    "X-Input-Bytes": {
        "description": "Parsed request body size in bytes.",
        "schema": {"type": "integer"},
    },
    "X-Body-Limit-Bytes": {
        "description": "Current configured body size limit.",
        "schema": {"type": "integer"},
    },
    # Placeholders for future rate limiting (Epic 5)
    "X-RateLimit-Limit": {"description": "Requests allowed in window.", "schema": {"type": "integer"}},
    "X-RateLimit-Remaining": {"description": "Remaining in current window.", "schema": {"type": "integer"}},
    "X-RateLimit-Reset": {"description": "Epoch seconds until window resets.", "schema": {"type": "integer"}},
}

@router.post(
    "/resolve",
    response_model=ResolveResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        200: {"headers": response_headers},
    },
    summary="Resolve acronyms in raw text",
    description=(
        "Detect acronyms, propose definitions, and (optionally) enrich from a curated glossary. "
        "Offsets use Python-slice semantics (end exclusive). Supported locales: en-GB, en-US. "
        "Idempotent: does not mutate server state. Content-Encoding: gzip supported."
    ),
)
def resolve_acronyms(
    payload: ResolveRequest,
    response: Response,
    x_request_id: Optional[str] = Header(default=None, convert_underscores=False),
    x_api_key: Optional[str] = Header(default=None, convert_underscores=False),
    content_encoding: Optional[str] = Header(default=None, convert_underscores=False),
) -> JSONResponse | dict[str, Any]:
    """
    Contract endpoint: returns deterministic example-shaped payload for tests & SDKs.
    """
    # Use Pydantic v2 API to avoid .json() mismatch
    text_bytes = len(payload.model_dump_json().encode("utf-8"))

    body_limit = 1_048_576  # 1 MiB default (documented; configurable)

    response.headers["X-Request-Id"] = x_request_id or "generated-req-id"
    response.headers["X-Input-Bytes"] = str(text_bytes)
    response.headers["X-Body-Limit-Bytes"] = str(body_limit)

    # Minimal semantic validation example (empty text → 422)
    if not payload.text.strip():
        err = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.UNPROCESSABLE_ENTITY,
                message="Text must not be empty.",
                details={"hint": "Provide non-empty 'text'"},
            )
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=err.model_dump())
    opts = payload.options or ResolveOptions.model_validate({})

    occurrences_part: dict[str, Any] = (
        {"occurrences": [{"start": 34, "end": 37}]}
        if opts.return_occurrences
        else {}
    )
    glossary_part: dict[str, Any] = (
        {
            "glossary": {
                "matches": [
                    {
                        "definition": "Metropolitan Police Service",
                        "domain": None,
                        "lang": "en",
                        "confidence": 0.99,
                        "source": "system",
                    }
                ]
            }
        }
        if opts.include_glossary_enrichment
        else {}
    )

    acronym = {
        "acronym": "MPS",
        "first_occurrence": {"start": 34, "end": 37},
        "definitions": [
            {
                "text": "Metropolitan Police Service",
                "start": 4,
                "end": 31,
                "confidence": 0.96,
                "source": "extracted",
            }
        ],
        **occurrences_part,
        **glossary_part,
    }

    example: dict[str, Any] = {
        "acronyms": [acronym],
        "meta": {
            "processing_ms": 12,
            "model_version": "plainera-core@1.0.0",
            "input_chars": len(payload.text),
        },
    }

    return example
