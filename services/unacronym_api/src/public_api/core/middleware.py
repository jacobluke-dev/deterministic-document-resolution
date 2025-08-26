from __future__ import annotations

import logging
import time
import uuid
from typing import Any, MutableMapping

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

REQ_ID_HEADER = "X-Request-ID"

class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        req_id = None
        for k, v in headers.items():
            if k.decode().lower() == REQ_ID_HEADER.lower():
                req_id = v.decode()
                break
        if not req_id:
            req_id = str(uuid.uuid4())

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers_list = message.setdefault("headers", [])
                headers_list.append((REQ_ID_HEADER.encode(), req_id.encode()))
            await send(message)

        scope["request_id"] = req_id
        await self.app(scope, receive, send_wrapper)

class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0
        async def limited_receive() -> MutableMapping[str, Any]:
            nonlocal received
            message: MutableMapping[str, Any] = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if received > self.max_body_bytes:
                    response = PlainTextResponse("Request entity too large", status_code=413)
                    await response(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message
        await self.app(scope, limited_receive, send)

class AccessLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.logger = logging.getLogger("access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        request = Request(scope, receive=receive)
        request_id = scope.get("request_id")
        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status = message.get("status")
                duration_ms = int((time.perf_counter() - start) * 1000)
                self.logger.info(
                    "request",
                    extra={
                        "request_id": request_id,
                        "path": request.url.path,
                        "method": request.method,
                        "status": status,
                        "duration_ms": duration_ms,
                    },
                )
            await send(message)
        await self.app(scope, receive, send_wrapper)


def apply_cors(app: ASGIApp, origins: list[str]) -> ASGIApp:
    return CORSMiddleware(
        app,
        allow_origins=origins or [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
