from typing import Protocol, Any


class AsyncSink(Protocol):
    async def enqueue_async(self, payload: dict[str, Any]) -> None: ...

class SyncSink(Protocol):
    def enqueue(self, payload: dict[str, Any]) -> None: ...
