from enum import Enum
from typing import Any, Optional

from pydantic import Field

from public_api.schemas.base import BaseSchema


class ErrorCode(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    UNPROCESSABLE_ENTITY = "UNPROCESSABLE_ENTITY"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

class ErrorBody(BaseSchema):
    code: ErrorCode = Field(..., description="Stable machine-readable enum.")
    message: str = Field(..., description="Human-readable error.")
    details: Optional[dict[str, Any]] = Field(
        None, description="Structured diagnostics (limit, actual, etc.)."
    )

class ErrorResponse(BaseSchema):
    error: ErrorBody
