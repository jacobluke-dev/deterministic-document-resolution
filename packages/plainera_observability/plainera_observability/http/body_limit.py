from typing import MutableMapping, Any

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Scope, Receive, Send


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
