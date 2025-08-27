
import uuid
from typing import Any, MutableMapping

from starlette.types import ASGIApp, Receive, Scope, Send

from shared.config import REQ_ID_HEADER


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
