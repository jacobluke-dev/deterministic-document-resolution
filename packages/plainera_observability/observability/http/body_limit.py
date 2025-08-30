from typing import MutableMapping, Any, Optional
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

        # 1) Short-circuit if Content-Length is present and over limit
        content_length: Optional[int] = None
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    content_length = int(v.decode("ascii", "ignore"))
                except Exception:
                    content_length = None
                break
        if content_length is not None and content_length > self.max_body_bytes:
            resp = PlainTextResponse("Request entity too large", status_code=413)
            await resp(scope, receive, send)
            return

        # 2) Wrap receive to count bytes and guard double-send
        received = 0
        rejected = False

        async def limited_receive() -> MutableMapping[str, Any]:
            nonlocal received, rejected
            message: MutableMapping[str, Any] = await receive()

            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                received += len(body)
                if received > self.max_body_bytes and not rejected:
                    rejected = True
                    resp = PlainTextResponse("Request entity too large", status_code=413)
                    # send 413 immediately
                    await resp(scope, receive, send)
                    # tell downstream we’re done; they’ll see disconnect
                    return {"type": "http.disconnect"}

            return message

        async def guarded_send(message: MutableMapping[str, Any]) -> None:
            # If we already sent 413, drop anything the app tries to send.
            if rejected:
                return
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
