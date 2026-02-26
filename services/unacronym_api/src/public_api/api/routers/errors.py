from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from public_api.schemas.error import ErrorBody, ErrorCode, ErrorResponse


async def map_length_validation_to_413(
    request: Request, exc: Exception | RequestValidationError
) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        try:
            for e in exc.errors():
                loc = e.get("loc", [])
                field = loc[-1] if loc else None
                typ = e.get("type", "")
                ctx = e.get("ctx", {}) or {}
                if field == "text" and typ in ("string_too_long", "value_error.any_str.max_length"):
                    limit = ctx.get("max_length")
                    actual = ctx.get("actual_length")
                    body = ErrorResponse(
                        error=ErrorBody(
                            code=ErrorCode.PAYLOAD_TOO_LARGE,
                            message="Body/text too large.",
                            details={"limit": limit, "actual": actual},
                        )
                    )
                    return JSONResponse(status_code=413, content=body.model_dump())
        except Exception:
            pass
        # Not a length validation → use FastAPI’s standard 422
        return await request_validation_exception_handler(request, exc)
    # Not a RequestValidationError: re-raise to let FastAPI’s global machinery handle it
    raise exc

async def map_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Map FastAPI HTTPException into our canonical ErrorResponse envelope.

    - 401 -> UNAUTHENTICATED
    - 403 -> FORBIDDEN
    - Everything else -> pass-through (best-effort)
    """
    if exc.status_code == HTTP_401_UNAUTHORIZED:
        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.UNAUTHENTICATED,
                message="API key required or invalid.",
                details=None,
            )
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    if exc.status_code == HTTP_403_FORBIDDEN:
        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.FORBIDDEN,
                message="Forbidden.",
                details=None,
            )
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    # Fallback: keep FastAPI behaviour for other HTTPExceptions.
    # If you want ALL HTTPExceptions wrapped, extend this mapping.
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
