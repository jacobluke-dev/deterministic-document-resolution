import asyncio
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from typing import Generic, Optional, TypeVar

from observability.logger.decorator import logger
from observability.logger.message_logger import message_logger

from plainera_unacronym.wiring.composition import sink

TResult = TypeVar("TResult")


class BaseDetector(Generic[TResult]):
    """Shared detector runtime behaviour.

    Intentionally small:
      - stores config
      - owns process-pool lifecycle
      - provides async wrapper
      - provides context-manager support

    Subclasses own:
      - detection semantics
      - result types
      - candidate iteration
      - scoring / filtering
      - result construction
    """

    def __init__(self, config, max_workers: Optional[int] = None):
        self.cfg = config
        self._max_workers = max_workers
        self._pool: Optional[ProcessPoolExecutor] = None
        self.sink = sink

    def detect(self, text: str) -> TResult:
        """Run synchronous detection."""
        raise NotImplementedError

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> TResult:
        """Run detection with optional process fan-out."""
        raise NotImplementedError

    async def detect_async(self, text: str) -> TResult:
        """
        Asynchronously run acronym detection without blocking the event loop.

        Async wrapper that offloads the sync detection pipeline to the event loop's default
        executor (a thread pool) by calling `detect_parallel` in a background thread. Use
        in FastAPI so the loop remains responsive while CPU-bound work executes off-loop.

        Args:
            text: The raw input text to scan for acronyms or defined texts or both.
        Returns:
            TResult: The detection result.
        """
        return await asyncio.to_thread(self.detect_parallel, text)

    def _get_or_create_pool(self) -> ProcessPoolExecutor:
        """Lazily create the process pool."""
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)
            message_logger(
                "detector.pool.created",
                logger_type="nlp",
                args={"max_workers": self._max_workers or cpu_count() or 1},
                db_sink=self.sink,
            )
        return self._pool

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """Shut down the internal process pool if it exists."""
        if self._pool is None:
            return

        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=False, cancel_futures=True)
        return False
