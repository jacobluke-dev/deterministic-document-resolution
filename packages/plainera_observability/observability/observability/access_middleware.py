import time
import uuid
from typing import Awaitable, Callable, Optional, Any, Coroutine
from starlette.requests import Request
from starlette.responses import Response
from .context import set_request_context
from .emit import emit
from .levels import LogLevel
from ..config import REQ_ID_HEADER

# //TODO write tests to make sure dplication of REQ_ID_header is accounted for
def access_middleware(app, *, header_name: str = REQ_ID_HEADER) -> Callable[
    [Request, Callable[[Request], Awaitable[Response]]], Coroutine[Any, Any, Response | None]]:
    @app.middleware("http")
    async def _mw(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response | None:
        # prefer ID from RequestIDMiddleware, fall back to incoming header, then generate
        rid = request.scope.get("request_id") or request.headers.get(header_name) or str(uuid.uuid4())
        start = time.perf_counter()

        key_id: Optional[str] = request.headers.get("X-API-Key-Id")
        client_ip: Optional[str] = request.client.host if request.client else None

        set_request_context(
            request_id=rid, key_id=key_id,
            path=request.url.path, method=request.method, client_ip=client_ip
        )

        response: Optional[Response] = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            if response is not None:
                # overwrite (single source of truth)
                response.headers[header_name] = rid

            content_len = None
            if response is not None:
                try:
                    v = response.headers.get("content-length")
                    content_len = int(v) if v is not None else None
                except Exception:
                    content_len = None

            dur_ms = int((time.perf_counter() - start) * 1000)
            emit(
                "http_access",
                level=LogLevel.INFO,
                logger_type="api",
                path=request.url.path,
                method=request.method,
                status=status_code,
                duration_ms=dur_ms,
                bytes=content_len,
            )
    return _mw
