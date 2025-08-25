from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import AnyUrl
from pydantic_settings import BaseSettings

# Resolve the .env location robustly (independent of CWD):
ENV_PATH = (Path(__file__).resolve().parents[2] / "../.env")  # -> services/public_api/.env
# Load .env into os.environ (silent=True avoids noise if file is missing)
load_dotenv(dotenv_path=ENV_PATH, override=False)

class AppSettings(BaseSettings):
    APP_ENV: str = "development"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = ""
    MAX_BODY_BYTES: int = 2 * 1024 * 1024
    ENABLE_DOCS: bool = True
    REQUEST_TIMEOUT_MS: int = 5000
    DEFAULT_LOCALE: str = "en-GB"
    DEFAULT_WINDOW_CHARS: int = 120
    MAX_INFLIGHT: int = 0
    SENTRY_DSN: str | None = None

    RUN_DB_MIGRATIONS: bool = True

    @property
    def cors_origins(self) -> list[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        return [o.strip() for o in raw.split(",") if o.strip()]

app_settings = AppSettings()


class DatabaseSettings(BaseSettings):
    APP_ENV: str = "local"
    AUTH_DISABLED: bool = False
    # Keep strict validation but avoid editor nags by allowing None and failing explicitly
    DATABASE_URL: AnyUrl | None = None

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return str(self.DATABASE_URL)  # cast for SQLAlchemy
        raise RuntimeError(
            f"DATABASE_URL missing. Expected in {ENV_PATH}. "
            "Set AUTH_DISABLED=true only if no DB is used."
        )


db_settings = DatabaseSettings()
