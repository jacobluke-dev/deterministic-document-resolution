from datetime import datetime, timedelta, timezone

import plainera_core.db_manager.mappers as mod
import pytest
from plainera_core.db_manager.mappers import (
    _CODE_TO_LEVEL_NAME,
    _LEVEL_NAME_TO_CODE,
    _level_name,
    _parse_ts,
    _sanitize_status,
)


@pytest.mark.unit
class TestLevelName:
    @pytest.mark.parametrize(
        "code,name",
        [
            (10, "debug"),
            (20, "info"),
            (30, "warning"),
            (40, "error"),
            (50, "critical"),
        ],
    )
    def test_int_known_codes(self, code, name):
        assert _level_name(code) == name

    @pytest.mark.parametrize("code", [-1, 0, 5, 15, 21, 99, 1000])
    def test_int_unknown_codes_defaults_to_info(self, code):
        assert _level_name(code) == "info"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("debug", "debug"),
            ("INFO", "info"),
            ("Warning", "warning"),
            ("eRrOr", "error"),
            ("CRITICAL", "critical"),
        ],
    )
    def test_string_names_case_insensitive(self, raw, expected):
        assert _level_name(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            " ",
            "verbose",
            "warn",      # not in map (we use "warning")
            "criticality",
            "20",        # numeric string → not a key, so default
        ],
    )
    def test_string_unknown_defaults_to_info(self, raw):
        assert _level_name(raw) == "info"

    def test_bool_defaults_to_info(self):
        # bool is a subclass of int (True == 1), which is not in the code map
        assert _level_name(True) == "info"
        assert _level_name(False) == "info"

    def test_object_uses_str_and_defaults(self):
        class Weird:
            def __str__(self):  # not a valid level name
                return "WeIrD"
        assert _level_name(Weird()) == "info"

    def test_maps_are_consistent_for_known_levels(self):
        # Round-trip: name -> code -> name
        for name, code in _LEVEL_NAME_TO_CODE.items():
            assert _CODE_TO_LEVEL_NAME[code] == name
            assert _level_name(code) == name


class TestParseTs:
    def test_datetime_naive_becomes_utc(self):
        dt = datetime(2025, 1, 2, 3, 4, 5)  # naive
        out = _parse_ts(dt)
        assert out.tzinfo is timezone.utc
        # Naive path sets tzinfo without converting the clock time
        assert out.replace(tzinfo=None) == dt

    def test_datetime_aware_returns_unchanged(self):
        # Aware (non-UTC) must pass through unchanged per implementation
        tz = timezone(timedelta(hours=2))
        dt = datetime(2025, 1, 2, 3, 4, 5, tzinfo=tz)
        out = _parse_ts(dt)
        assert out is dt  # same object returned
        assert out.tzinfo is tz

    def test_iso_string_with_Z_normalizes_to_utc(self):
        s = "2025-09-17T12:34:56Z"
        out = _parse_ts(s)
        assert out.tzinfo is timezone.utc
        assert out == datetime(2025, 9, 17, 12, 34, 56, tzinfo=timezone.utc)

    def test_iso_string_with_offset_is_converted_to_utc(self):
        # 12:34:56+02:00 == 10:34:56Z
        s = "2025-09-17T12:34:56+02:00"
        out = _parse_ts(s)
        assert out.tzinfo is timezone.utc
        assert out == datetime(2025, 9, 17, 10, 34, 56, tzinfo=timezone.utc)

    @pytest.mark.parametrize("bad", ["", "not-a-date", "2025-09-17", 1234567890, object()])
    def test_invalid_inputs_default_to_frozen_now(self, bad, monkeypatch):
        FROZEN = datetime(2025, 9, 16, 23, 0, tzinfo=timezone.utc)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                # mirror stdlib signature/behavior
                return FROZEN if tz else FROZEN.replace(tzinfo=None)

        # Patch the datetime symbol used *inside* the module under test
        monkeypatch.setattr(mod, "datetime", FixedDateTime, raising=True)

        out = _parse_ts(bad)
        assert out == FROZEN
        assert out.tzinfo is timezone.utc
    def test_string_with_microseconds_and_z(self):
        s = "2025-09-17T12:34:56.123456Z"
        out = _parse_ts(s)
        assert out == datetime(2025, 9, 17, 12, 34, 56, 123456, tzinfo=timezone.utc)
        assert out.tzinfo is timezone.utc


class TestSanitizeStatus:
    @pytest.mark.parametrize("inp, expected", [
        (100, 100), (200, 200), (404, 404), (599, 599),     # in-range ints
        ("100", 100), ("404", 404), (" 599 ", 599),         # numeric strings (trim ok)
        (200.0, 200),                                       # float -> int works if whole number
    ])
    def test_valid_values(self, inp, expected):
        assert _sanitize_status(inp) == expected

    @pytest.mark.parametrize("inp", [
        99, 0, -1, 600, 700,                                # out-of-range ints
        "099", "600", "xyz", "200.1", "200.0",              # strings that parse out or fail
        200.1, float("inf"), float("nan"),                  # non-integer-ish numerics
        None, object(),                                     # other types
        True, False,                                        # bools are ints (1/0) => out of range
    ])
    def test_invalid_values_return_none(self, inp):
        assert _sanitize_status(inp) is None

    @pytest.mark.parametrize("inp", [
        99, 600, "200.1", 200.1, float("inf"), float("nan"),
        None, object(), True, False,
    ])
    def test_invalid_values(self, inp):
        assert _sanitize_status(inp) is None

    def test_large_int_returns_none(self):
        assert _sanitize_status(10_000) is None

    def test_min_max_edges(self):
        assert _sanitize_status(100) == 100
        assert _sanitize_status(599) == 599
        assert _sanitize_status(99) is None
        assert _sanitize_status(600) is None



def _cols(*names):
    """
    Helper to build a fake sqla_inspect(...).columns with .key attrs.
    """
    class C:  # simple carrier for .key
        def __init__(self, k): self.key = k
    class InspectResult:
        def __init__(self, keys): self.columns = [C(k) for k in keys]
    return InspectResult(names)


class TestMakeLoggerMapper:
    def test_maps_core_fields_and_filters_by_model_columns(self, monkeypatch):
        # --- Arrange: fake model + columns we claim exist on it
        class Model:
            pass
        model_cols = (
            "level_code", "level_name", "event", "logger_type",
            "function_name", "request_id", "duration_ms",
            "info", "arguments", "keyword_arguments", "date_time",
            "path", "bytes",  # only some of the optional HTTP-ish columns
        )
        monkeypatch.setattr(mod, "sqla_inspect", lambda _m: _cols(*model_cols), raising=True)

        # Freeze helpers
        monkeypatch.setattr(mod, "_level_name", lambda raw: "warning", raising=True)
        monkeypatch.setattr(mod, "_LEVEL_NAME_TO_CODE", {"warning": 30}, raising=True)

        FROZEN = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        seen_ts = {}
        def _fake_parse_ts(x):
            seen_ts["arg"] = x
            return FROZEN
        monkeypatch.setattr(mod, "_parse_ts", _fake_parse_ts, raising=True)

        # Mapper under test
        mapper = mod.make_logger_mapper(Model, default_logger_type="decorator")

        # Payload includes extras that are NOT in model_cols and should be dropped
        payload = {
            "level": "WARNING",
            "event": "user_login",
            "logger_type": "api",             # overrides default
            "function": "auth.login",
            "request_id": "req-123",
            "duration_ms": 42,
            "result": "ok",                   # info precedence winner
            "info": "ignored-info",
            "args": {"u": "alice"},
            "kwargs": {"force": True},
            "timestamp": "2025-01-02T03:04:05Z",
            "path": "/v1/login",
            "bytes": 512,
            "client_ip": "1.2.3.4",          # not in model_cols → dropped
            "extra_field": "drop-me",         # dropped
        }

        # --- Act
        out = mapper(payload)

        # --- Assert: core mapping
        assert out["level_name"] == "warning"
        assert out["level_code"] == 30
        assert out["event"] == "user_login"
        assert out["logger_type"] == "api"            # payload wins over default
        assert out["function_name"] == "auth.login"
        assert out["request_id"] == "req-123"
        assert out["duration_ms"] == 42
        assert out["arguments"] == {"u": "alice"}
        assert out["keyword_arguments"] == {"force": True}
        assert out["date_time"] == FROZEN
        assert seen_ts["arg"] == "2025-01-02T03:04:05Z"

        # Optional HTTP-ish fields: only those present on the model are included
        assert out["path"] == "/v1/login"
        assert out["bytes"] == 512
        assert "client_ip" not in out  # model didn’t have it

        # Dropped unknowns
        assert "extra_field" not in out

    def test_default_logger_type_used_when_missing_in_payload(self, monkeypatch):
        class Model:
            pass
        cols = (
            "level_code","level_name","event","logger_type",
            "info","arguments","keyword_arguments","date_time"
        )
        monkeypatch.setattr(mod, "sqla_inspect", lambda _m: _cols(*cols), raising=True)
        monkeypatch.setattr(mod, "_level_name", lambda raw: "info", raising=True)
        monkeypatch.setattr(mod, "_LEVEL_NAME_TO_CODE", {"info": 20}, raising=True)

        FROZEN = datetime(2025, 5, 6, 7, 8, 9, tzinfo=timezone.utc)
        monkeypatch.setattr(mod, "_parse_ts", lambda x: FROZEN, raising=True)

        mapper = mod.make_logger_mapper(Model, default_logger_type="decorator")
        out = mapper({"event": "e", "timestamp": "ignored", "args": None, "kwargs": None})

        assert out["logger_type"] == "decorator"
        assert out["level_name"] == "info"
        assert out["level_code"] == 20
        assert out["event"] == "e"
        assert out["info"] == "e"  # falls back to event when no result/info
        assert out["date_time"] == FROZEN

    @pytest.mark.parametrize(
        "payload,expected_info",
        [
            ({"result": "R", "info": "I", "event": "E"}, "R"),  # result wins
            ({"info": "I", "event": "E"}, "I"),                 # info next
            ({"event": "E"}, "E"),                              # then event
            ({}, None),                                           # empty when none present
        ],
    )
    def test_info_field_precedence(self, payload, expected_info, monkeypatch):
        class Model:
            pass
        cols = ("level_code","level_name","event","logger_type","info","date_time")
        monkeypatch.setattr(mod, "sqla_inspect", lambda _m: _cols(*cols), raising=True)
        monkeypatch.setattr(mod, "_level_name", lambda raw: "info", raising=True)
        monkeypatch.setattr(mod, "_LEVEL_NAME_TO_CODE", {"info": 20}, raising=True)
        monkeypatch.setattr(mod, "_parse_ts", lambda x: datetime(2025,1,1,tzinfo=timezone.utc), raising=True)

        mapper = mod.make_logger_mapper(Model)
        out = mapper({**payload, "timestamp": "ignored"})
        # event defaults to "" if missing per implementation
        assert out["info"] == expected_info

    def test_optional_http_fields_only_included_if_model_has_them(self, monkeypatch):
        class Model:
            pass
        # Model only has 'method' and 'client_ip', not others
        cols = ("level_code","level_name","event","logger_type","info","date_time",
                "method","client_ip")
        monkeypatch.setattr(mod, "sqla_inspect", lambda _m: _cols(*cols), raising=True)
        monkeypatch.setattr(mod, "_level_name", lambda raw: "debug", raising=True)
        monkeypatch.setattr(mod, "_LEVEL_NAME_TO_CODE", {"debug": 10}, raising=True)
        monkeypatch.setattr(mod, "_parse_ts", lambda x: datetime(2025,1,1,tzinfo=timezone.utc), raising=True)

        mapper = mod.make_logger_mapper(Model)
        out = mapper({
            "event": "x",
            "timestamp": "ignored",
            "method": "GET",
            "path": "/should-drop",    # not in model
            "client_ip": "1.2.3.4",
            "bytes": 123,              # not in model
        })

        assert out["method"] == "GET"
        assert out["client_ip"] == "1.2.3.4"
        assert "path" not in out
        assert "bytes" not in out
