from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from src.public_api.schemas.base import BaseSchema


class LogEventCreate(BaseSchema):
    # minimal required
    date_time: datetime = Field(..., description="UTC ISO-8601")
    level_code: int
    level_name: str
    event: str
    logger_type: str

    # optional
    function_name: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[int] = None
    path: Optional[str] = None
    method: Optional[str] = None
    status: Optional[int] = None
    bytes: Optional[int] = None
    client_ip: Optional[str] = None
    key_id: Optional[str] = None
    info: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None
    keyword_arguments: Optional[dict[str, Any]] = None


class LogEventRead(LogEventCreate):
    id: int
