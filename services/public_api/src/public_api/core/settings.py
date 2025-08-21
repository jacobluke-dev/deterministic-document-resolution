from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    APP_ENV: str = "development"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"


    CORS_ORIGINS: str = ""
    MAX_BODY_BYTES: int = 2 * 1024 * 1024
    ENABLE_DOCS: bool = True
    REQUEST_TIMEOUT_MS: int = 5000


    SENTRY_DSN: Optional[str] = None


    model_config = SettingsConfigDict(env_file=(".env",), env_prefix="", case_sensitive=False)


    @property
    def cors_origins(self) -> List[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        return [o.strip() for o in raw.split(",") if o.strip()]
