from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from observability.logger.message_logger import warning
from starlette.responses import JSONResponse

from public_api.api.response_types import build_responses
from public_api.core.deps import get_resolve_service
from public_api.core.deps_auth import require_api_key
from public_api.core.services.resolve_service import ResolveError, ResolveService
from public_api.core.settings import app_settings
from public_api.schemas.error import ErrorBody, ErrorResponse
from public_api.schemas.resolve import ResolveRequest, ResolveResponse

router = APIRouter(prefix="/v1", tags=["Resolve"], dependencies=[Depends(require_api_key)])


def _error_json(err: ResolveError) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=err.code,
            message=err.message,
            details=err.details or {},
        )
    ).model_dump()
    return JSONResponse(status_code=err.http_status, content=body)


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    responses=build_responses(success_status=200),
    summary="Resolve acronyms in raw text",
    description=(
        "Detect acronyms, propose definitions, and (optionally) enrich from a curated glossary. "
        "Offsets use Python-slice semantics (end exclusive). Supported locales: en-GB, en-US. "
        "Idempotent: does not mutate server state. Content-Encoding: gzip supported."
    ),
)
async def resolve_acronyms(
    payload: ResolveRequest,
    response: Response,
    svc: Annotated[ResolveService, Depends(get_resolve_service)],
) -> ResolveResponse | JSONResponse:
    started = time.perf_counter()

    # Headers (validated-model size; actual raw bytes belong in ASGI middleware if you want true wire size)
    text_bytes = len(payload.model_dump_json().encode("utf-8"))
    response.headers["X-Input-Bytes"] = str(text_bytes)
    response.headers["X-Body-Limit-Bytes"] = str(app_settings.MAX_BODY_BYTES)

    try:
        out = await svc.resolve(payload)
    except ResolveError as err:
        warning(
            "resolve request failed",
            logger_type="public_api",
            args={
                "code": err.code,
                "message": err.message,
                "http_status": err.http_status,
            },
        )
        return _error_json(err)

    # Ensure processing_ms is sane even if the service was refactored later
    if out.meta.processing_ms <= 0:
        out.meta.processing_ms = int((time.perf_counter() - started) * 1000)

    return out
