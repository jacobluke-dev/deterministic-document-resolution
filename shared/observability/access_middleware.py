import time, uuid
from typing import Callable
from fastapi import Request, Response
from .context import set_request_context
from .emit import emit
from .levels import LogLevel
from ..config import REQ_ID_HEADER


def access_middleware(app, *, header_name: str = REQ_ID_HEADER):
    @app.middleware("http")
    async def _mw(request: Request, call_next: Callable):
        global response
        rid = request.headers.get(header_name) or str(uuid.uuid4())
        start = time.perf_counter()
        # Optional API key extraction
        key_id = request.headers.get("X-API-Key-Id") or None
        client_ip = request.client.host if request.client else None
        set_request_context(
            request_id=rid, key_id=key_id,
            path=request.url.path, method=request.method, client_ip=client_ip
        )
        response: Response
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            emit(
                "http_access",
                level=LogLevel.INFO,
                logger_type="api",
                path=request.url.path,
                method=request.method,
                status=getattr(response, "status_code", 500),
                duration_ms=dur_ms,
                bytes=getattr(response, "headers", {}).get("content-length"),
            )
            if REQ_ID_HEADER not in (response.headers or {}):
                try: response.headers[REQ_ID_HEADER] = rid
                except Exception: pass
    return _mw
