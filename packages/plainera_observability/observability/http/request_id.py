# observability/http/request_id.py
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from observability.config import REQ_ID_HEADER


class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Prefer incoming header, else generate
        req_id = None
        for k, v in scope.get("headers", []):
            if k.decode().lower() == REQ_ID_HEADER.lower():
                req_id = v.decode()
                break
        if not req_id:
            req_id = str(uuid.uuid4())

        scope["request_id"] = req_id

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # Only add if missing — lets access_middleware be the "single source of truth"
                if REQ_ID_HEADER not in headers:
                    headers[REQ_ID_HEADER] = req_id
            await send(message)

        await self.app(scope, receive, send_wrapper)
