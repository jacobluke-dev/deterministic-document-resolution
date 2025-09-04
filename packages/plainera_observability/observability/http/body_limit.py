from typing import Any, MutableMapping, Optional

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await self._send_413(scope, receive, send)
            return

        guard = _BodyGuard(self.max_body_bytes, scope, receive, send)
        await self.app(scope, guard.limited_receive, guard.guarded_send)

    @staticmethod
    def _content_length(scope: Scope) -> Optional[int]:
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    return int(v.decode("ascii", "ignore"))
                except Exception:
                    return None
        return None

    @staticmethod
    async def _send_413(scope: Scope, receive: Receive, send: Send) -> None:
        resp = PlainTextResponse("Request entity too large", status_code=413)
        await resp(scope, receive, send)


class _BodyGuard:
    def __init__(self, max_body_bytes: int, scope: Scope, receive: Receive, send: Send) -> None:
        self._max = max_body_bytes
        self._scope = scope
        self._receive = receive
        self._send = send
        self._received = 0
        self._rejected = False

    async def limited_receive(self) -> MutableMapping[str, Any]:
        message: MutableMapping[str, Any] = await self._receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"") or b""
            self._received += len(body)
            if self._received > self._max and not self._rejected:
                self._rejected = True
                await BodySizeLimitMiddleware._send_413(self._scope, self._receive, self._send)
                return {"type": "http.disconnect"}
        return message

    async def guarded_send(self, message: MutableMapping[str, Any]) -> None:
        if not self._rejected:
            await self._send(message)
