import uuid
from starlette.types import ASGIApp, Receive, Scope, Send
from observability.config import REQ_ID_HEADER

class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = None
        for k, v in scope.get("headers", []):
            if k.decode().lower() == REQ_ID_HEADER.lower():
                req_id = v.decode()
                break
        if not req_id:
            req_id = str(uuid.uuid4())

        scope["request_id"] = req_id

        await self.app(scope, receive, send)
