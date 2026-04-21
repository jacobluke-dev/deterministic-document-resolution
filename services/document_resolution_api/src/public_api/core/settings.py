from pathlib import Path

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings

from document_resolution_core.utils.utils import find_project_root

SERVICE_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = Path(find_project_root(), '.env')


try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except Exception:
    pass


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

    AUTH_DISABLED: bool = False
    API_KEY_PREFIX_ALLOWLIST: str = "live,test"
    API_KEY_HASH_SCHEME: str = "argon2id"  # argon2id|bcrypt
    API_KEY_CACHE_TTL_SECONDS: int = 60
    API_KEY_LAST_USED_ASYNC: bool = True
    DAILY_QUOTA_DEFAULT: int = 1000
    RATE_LIMIT_PER_MIN: int = 60

    TIER2_ENABLED: bool = True
    TIER2_STRICT: bool = False
    TIER2_MODEL_NAME: str = ''
    HF_CACHE_DIR: str = ''

    CHUNKING_ENABLED: bool = True
    CHUNK_THRESHOLD_CHARS: int = 30_000
    CHUNK_SIZE_CHARS: int = 20_000
    CHUNK_OVERLAP_CHARS: int = 1_200

    @property
    def cors_origins(self) -> list[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        return [o.strip() for o in raw.split(",") if o.strip()]

app_settings = AppSettings()


class DatabaseSettings(BaseSettings):
    APP_ENV: str = "local"
    DATABASE_DISABLED: bool = False
    # Keep strict validation but avoid editor nags by allowing None and failing explicitly
    DATABASE_URL: AnyUrl | None = None
    DB_SCHEMA: str = "document_resolution"
    NAMING_CONVENTION: dict[str, str] = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
    ALEMBIC_INI_PATH: Path = Field(default=SERVICE_ROOT / "alembic.ini")

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return str(self.DATABASE_URL)  # cast for SQLAlchemy
        raise RuntimeError(
            f"DATABASE_URL missing. Expected in {ENV_PATH}. "
            "set DATABASE_DISABLED=true only if no DB is used."
        )

db_settings = DatabaseSettings()
