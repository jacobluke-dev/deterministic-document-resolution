from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Base for public API schemas.

    anything added here becomes part of the public contract
    everywhere, including nested objects.
    """
    model_config = ConfigDict(from_attributes=True)


class AuditSchema(BaseSchema):
    """Optional base for schemas that legitimately need audit timestamps.

    Only use this for internal/admin payloads (not public API responses)
    """
    created_at: str | None = None
    updated_at: str | None = None
