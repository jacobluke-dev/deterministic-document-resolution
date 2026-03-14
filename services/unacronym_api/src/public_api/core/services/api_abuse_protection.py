from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import text

from observability.logger.message_logger import warning
from plainera_core.db_manager.connection import DBManager
from plainera_core.db_manager.sink_factory import SinkSpec
from public_api.core.settings import app_settings


SCHEMA = "unacronym"


@dataclass(slots=True)
class QuotaExceededError(Exception):
    limit: int
    current_count: int
    reset_at: datetime


@dataclass(slots=True)
class RateLimitExceededError(Exception):
    limit: int
    current_count: int
    reset_at: datetime
    retry_after: int


class ApiAbuseProtectionService:

    def __init__(self, dbm: DBManager, sink: SinkSpec) -> None:
        self.dbm = dbm
        self.sink = sink

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    def enforce(self, *, api_key_id: int, daily_quota_override: int | None) -> None:
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
    def _increment_daily(*, session, api_key_id: int, usage_date: date) -> int:
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
    def _increment_minute(*, session, api_key_id: int, minute_bucket: datetime) -> int:
        stmt = text(f"""
            INSERT INTO {SCHEMA}.api_usage_minute (api_key_id, minute_bucket, request_count, created_at)
            VALUES (:api_key_id, :minute_bucket, 1, NOW())
            ON CONFLICT (api_key_id, minute_bucket)
            DO UPDATE SET request_count = {SCHEMA}.api_usage_minute.request_count + 1
            RETURNING request_count
        """)
        result = session.execute(
            stmt,
            {"api_key_id": api_key_id, "minute_bucket": minute_bucket},
        )
        return int(result.scalar_one())

    def _log_quota_exceeded(self, *, api_key_id: int, limit: int, current_count: int, reset_at: datetime) -> None:
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

    def _log_rate_limited(self, *, api_key_id: int, limit: int, current_count: int, reset_at: datetime) -> None:
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
