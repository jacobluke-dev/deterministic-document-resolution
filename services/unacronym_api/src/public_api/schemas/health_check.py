from src.public_api.schemas.base import BaseSchema


class HealthCheckResponse(BaseSchema):
    status: str
