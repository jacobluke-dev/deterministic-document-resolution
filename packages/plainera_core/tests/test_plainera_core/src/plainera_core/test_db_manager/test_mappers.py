from datetime import datetime, timedelta, timezone

import pytest
from plainera_core.db_manager.mappers import logger_model_map


@pytest.mark.unit
class TestLoggerModelMapLevels:
    @pytest.mark.parametrize(
        "level_in, expected_code, expected_name",
        [
            ("DEBUG", 10, "debug"),
            ("info", 20, "info"),
            ("Warning", 30, "warning"),
            ("error", 40, "error"),
            ("critical", 50, "critical"),
            ("weird", 20, "weird"),  # unknown -> code fallback 20, name preserved lowercased
            (None, 20, "info"),      # default
        ],
    )
    def test_level_mapping(self, level_in, expected_code, expected_name):
        payload = {"level": level_in}
        out = logger_model_map(payload)
        assert out["level_code"] == expected_code
        # level_name should be lowercased string (default "info" when None)
        assert out["level_name"] == expected_name


@pytest.mark.unit
class TestLoggerModelMapTimestamp:
    def test_uses_iso_timestamp_when_provided(self):
        ts = "2025-09-04T12:34:56+00:00"
        out = logger_model_map({"timestamp": ts})
        assert isinstance(out["date_time"], datetime)
        # fromisoformat should parse exactly the same instant
        assert out["date_time"] == datetime.fromisoformat(ts)

    def test_defaults_to_utc_now_when_missing(self):
        before = datetime.now(timezone.utc)
        out = logger_model_map({})
        after = datetime.now(timezone.utc)

        dt = out["date_time"]
        assert isinstance(dt, datetime)
        # tz-aware and UTC
        assert dt.tzinfo == timezone.utc
        # sanity: within a small window of "now" to avoid flakiness
        assert before - timedelta(seconds=5) <= dt <= after + timedelta(seconds=5)


@pytest.mark.unit
class TestLoggerModelMapFields:
    def test_basic_field_mapping_and_defaults(self):
        payload = {
            "level": "info",
            "event": "fn_executed",
            "logger_type": "decorator",
            "function": "do_work",
            "request_id": "req-123",
            "duration_ms": 42.5,
            "args": ["a", 1],
            "kwargs": {"x": True},
            "path": "/api/thing",
            "method": "GET",
            "status": 200,
            "bytes": 512,
            "client_ip": "1.2.3.4",
            "key_id": "key_abc",
            "timestamp": "2025-09-04T00:00:00+00:00",
        }
        out = logger_model_map(payload)

        # Selected assertions (spot-checking all keys exist and pass through)
        assert out["event"] == "fn_executed"
        assert out["logger_type"] == "decorator"
        assert out["function_name"] == "do_work"
        assert out["request_id"] == "req-123"
        assert out["duration_ms"] == 42.5
        assert out["arguments"] == ["a", 1]
        assert out["keyword_arguments"] == {"x": True}
        assert out["path"] == "/api/thing"
        assert out["method"] == "GET"
        assert out["status"] == 200
        assert out["bytes"] == 512
        assert out["client_ip"] == "1.2.3.4"
        assert out["key_id"] == "key_abc"
        assert out["date_time"] == datetime.fromisoformat(str(payload["timestamp"]))

    @pytest.mark.parametrize(
        "payload, expected_info",
        [
            ({"result": "OK", "info": "ignored", "event": "evt"}, "OK"),         # result wins
            ({"info": "Only info", "event": "evt"}, "Only info"),                 # else info
            ({"event": "evt"}, "evt"),                                            # else event
            ({}, None),                                                           # nothing present
        ],
    )
    def test_info_fallback_order(self, payload, expected_info):
        out = logger_model_map(payload)
        assert out["info"] == expected_info

    def test_defaults_for_missing_values(self):
        out = logger_model_map({})
        assert out["event"] == ""                 # default empty
        assert out["logger_type"] == "decorator"  # default
        assert out["function_name"] is None
        assert out["arguments"] is None
        assert out["keyword_arguments"] is None
