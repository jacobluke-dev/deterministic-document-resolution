from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from document_resolution_core.db_manager.connection import DBManager
from document_resolution_core.db_manager.sinks import UniversalSink
from fastapi import Request
from observability.logger.message_logger import warning
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from public_api.core.settings import app_settings

SCHEMA = "document_resolution"


@dataclass(slots=True)
class QuotaExceededError(Exception):
    """Raised when an API key exceeds its configured daily quota.

    Attributes:
      limit: Maximum number of requests permitted for the current daily window.
      current_count: Request count after applying the request that exceeded the
        quota.
      reset_at: UTC timestamp at which the daily quota window resets.
    """
    limit: int
    current_count: int
    reset_at: datetime


@dataclass(slots=True)
class RateLimitExceededError(Exception):
    """Raised when an API key exceeds the per-minute rate limit.

    Attributes:
      limit: Maximum number of requests permitted for the current minute window.
      current_count: Request count after applying the request that exceeded the
        rate limit.
      reset_at: UTC timestamp at which the current minute window resets.
      retry_after: Number of seconds the client should wait before retrying.
    """
    limit: int
    current_count: int
    reset_at: datetime
    retry_after: int

async def quota_exceeded_handler(_: Request, exc: Exception) -> JSONResponse:
    """Build a JSON response for a quota exceeded error.

    This handler validates that the supplied exception is a
    ``QuotaExceededError`` and converts it into a structured HTTP 403 response.

    Args:
      _: Incoming FastAPI request. The request is unused by this handler.
      exc: Exception raised during request processing.

    Returns:
      A ``JSONResponse`` containing the quota error payload.

    Raises:
      TypeError: If ``exc`` is not an instance of ``QuotaExceededError``.
    """
    err = exc if isinstance(exc, QuotaExceededError) else None
    if err is None:
        raise TypeError(f"Expected QuotaExceededError, got {type(exc).__name__}")

    return JSONResponse(
        status_code=403,
        content={
            "error": "quota_exceeded",
            "limit": err.limit,
            "reset_at": err.reset_at.isoformat(),
        },
    )

async def rate_limited_handler(_: Request, exc: Exception) -> JSONResponse:
    """Build a JSON response for a rate limit exceeded error.

    This handler validates that the supplied exception is a
    ``RateLimitExceededError`` and converts it into a structured HTTP 429
    response, including a ``Retry-After`` header.

    Args:
      _: Incoming FastAPI request. The request is unused by this handler.
      exc: Exception raised during request processing.

    Returns:
      A ``JSONResponse`` containing the rate limit error payload.

    Raises:
      TypeError: If ``exc`` is not an instance of ``RateLimitExceededError``.
    """
    err = exc if isinstance(exc, RateLimitExceededError) else None
    if err is None:
        raise TypeError(f"Expected RateLimitExceededError, got {type(exc).__name__}")

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limited",
            "limit": err.limit,
            "reset_at": err.reset_at.isoformat(),
        },
        headers={"Retry-After": str(err.retry_after)},
    )


class ApiAbuseProtectionService:

    def __init__(self, dbm: DBManager, sink: UniversalSink) -> None:
        """Initialise the API abuse protection service.

        Args:
          dbm: Database manager used to open synchronous database sessions.
          sink: Observability sink used for warning logs when limits are exceeded.
        """
        self.dbm = dbm
        self.sink = sink

    @staticmethod
    def _utc_now() -> datetime:
        """Return the current UTC timestamp.

        Returns:
          A timezone-aware ``datetime`` in UTC.
        """
        return datetime.now(UTC)

    def enforce(self, *, api_key_id: int, daily_quota_override: int | None) -> None:
        """Record usage for an API key and enforce minute and daily limits.

        This method increments the current minute bucket first, then the current
        daily bucket, and compares the updated counts against the configured
        thresholds. If the per-minute limit is exceeded, a
        ``RateLimitExceededError`` is raised. If the daily quota is exceeded, a
        ``QuotaExceededError`` is raised.

        The effective daily limit is resolved from ``daily_quota_override`` when
        provided, otherwise ``app_settings.DAILY_QUOTA_DEFAULT`` is used.

        Args:
          api_key_id: Internal database identifier of the API key being charged for
            the request.
          daily_quota_override: Optional per-key daily quota override. When
            ``None``, the default application quota is applied.

        Raises:
          RateLimitExceededError: If the API key exceeds the configured per-minute
            request limit.
          QuotaExceededError: If the API key exceeds the configured daily request
            quota.
        """
        now = self._utc_now()
        usage_date = now.date()
        minute_bucket = now.replace(second=0, microsecond=0)

        minute_limit = app_settings.RATE_LIMIT_PER_MIN
        daily_limit = (
            daily_quota_override
            if daily_quota_override is not None
            else app_settings.DAILY_QUOTA_DEFAULT
        )

        next_minute = minute_bucket + timedelta(minutes=1)
        next_midnight = datetime.combine(usage_date + timedelta(days=1), time.min, tzinfo=UTC)

        with self.dbm.session() as s:
            minute_count = self._increment_minute(
                session=s,
                api_key_id=api_key_id,
                minute_bucket=minute_bucket,
            )
            if minute_count > minute_limit:
                self._log_rate_limited(
                    api_key_id=api_key_id,
                    limit=minute_limit,
                    current_count=minute_count,
                    reset_at=next_minute,
                )
                retry_after = max(1, int((next_minute - now).total_seconds()))
                raise RateLimitExceededError(
                    limit=minute_limit,
                    current_count=minute_count,
                    reset_at=next_minute,
                    retry_after=retry_after,
                )

            daily_count = self._increment_daily(
                session=s,
                api_key_id=api_key_id,
                usage_date=usage_date,
            )
            if daily_count > daily_limit:
                self._log_quota_exceeded(
                    api_key_id=api_key_id,
                    limit=daily_limit,
                    current_count=daily_count,
                    reset_at=next_midnight,
                )
                raise QuotaExceededError(
                    limit=daily_limit,
                    current_count=daily_count,
                    reset_at=next_midnight,
                )

    @staticmethod
    def _increment_daily(*, session: Session, api_key_id: int, usage_date: date) -> int:
        """Increment and return the daily request count for an API key.

        This performs an atomic upsert into ``api_usage_daily`` using PostgreSQL
        ``ON CONFLICT`` semantics.

        Args:
          session: Active SQLAlchemy session.
          api_key_id: Internal database identifier of the API key.
          usage_date: UTC date bucket for usage aggregation.

        Returns:
          The updated request count for the specified API key and date bucket.
        """
        stmt = text(f"""
            INSERT INTO {SCHEMA}.api_usage_daily (api_key_id, usage_date, request_count, created_at)
            VALUES (:api_key_id, :usage_date, 1, NOW())
            ON CONFLICT (api_key_id, usage_date)
            DO UPDATE SET request_count = {SCHEMA}.api_usage_daily.request_count + 1
            RETURNING request_count
        """)
        result = session.execute(
            stmt,
            {"api_key_id": api_key_id, "usage_date": usage_date},
        )
        return int(result.scalar_one())

    @staticmethod
    def _increment_minute(*, session: Session, api_key_id: int, minute_bucket: datetime) -> int:
        """Increment and return the per-minute request count for an API key.

        This performs an atomic upsert into ``api_usage_minute`` using PostgreSQL
        ``ON CONFLICT`` semantics.

        Args:
          session: Active SQLAlchemy session.
          api_key_id: Internal database identifier of the API key.
          minute_bucket: UTC minute bucket for usage aggregation, with seconds and
            microseconds normalised to zero.

        Returns:
          The updated request count for the specified API key and minute bucket.
        """
        stmt = text(f"""
            INSERT INTO {SCHEMA}.api_usage_minute (api_key_id, minute_bucket, request_count)
            VALUES (:api_key_id, :minute_bucket, 1)
            ON CONFLICT (api_key_id, minute_bucket)
            DO UPDATE SET request_count = {SCHEMA}.api_usage_minute.request_count + 1
            RETURNING request_count
        """)
        result = session.execute(
            stmt,
            {"api_key_id": api_key_id, "minute_bucket": minute_bucket},
        )
        return int(result.scalar_one())

    def _log_quota_exceeded(
        self,
        *,
        api_key_id: int,
        limit: int,
        current_count: int,
        reset_at: datetime,
    ) -> None:
        """Write a quota exceeded warning to the observability sink.

        Args:
          api_key_id: Internal database identifier of the API key.
          limit: Configured daily request limit.
          current_count: Request count at the point the limit was exceeded.
          reset_at: UTC timestamp at which the daily quota window resets.
        """
        warning(
            "quota_exceeded",
            logger_type="api_abuse_protection",
            args={
                "api_key_id": api_key_id,
                "limit": limit,
                "current_count": current_count,
                "reset_at": reset_at.isoformat(),
            },
            db_sink=self.sink,
        )

    def _log_rate_limited(
        self,
        *,
        api_key_id: int,
        limit: int,
        current_count: int,
        reset_at: datetime,
    ) -> None:
        """Write a rate-limited warning to the observability sink.

        Args:
          api_key_id: Internal database identifier of the API key.
          limit: Configured per-minute request limit.
          current_count: Request count at the point the limit was exceeded.
          reset_at: UTC timestamp at which the current minute window resets.
        """
        warning(
            "rate_limited",
            logger_type="api_abuse_protection",
            args={
                "api_key_id": api_key_id,
                "limit": limit,
                "current_count": current_count,
                "reset_at": reset_at.isoformat(),
            },
            db_sink=self.sink,
        )
