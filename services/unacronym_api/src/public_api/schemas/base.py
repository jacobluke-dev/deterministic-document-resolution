from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    class Config:
        model_config = ConfigDict(from_attributes=True)
    created_at: datetime | None = None
    updated_at: datetime | None = None
