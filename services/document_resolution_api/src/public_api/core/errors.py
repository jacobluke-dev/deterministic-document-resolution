from dataclasses import dataclass
from typing import Any

from public_api.schemas.error import ErrorCode


@dataclass(frozen=True)
class ResolveError(Exception):
    """Structured domain exception for resolve endpoint failures.

    Attributes:
      http_status: HTTP status code to return to the caller.
      code: Stable public error code.
      message: Human-readable error message.
      details: Optional structured details for diagnostics and client handling.
    """
    http_status: int
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
