import json
import logging
import re

import pytest

from observability.logger.levels import LogLevel
from observability.logger.message_logger import (
    message_logger,
    info as log_info,
    debug as log_debug,
    warning as log_warning,
    error as log_error, _to_text,
)


def _last_json(caplog):
    """
    grabs the most recent captured log record via caplog.records[-1],
    assumes the record’s message is JSON, parses it with json.loads(rec.getMessage()),
    and returns a tuple of (parsed_payload, original_record).
    """
    rec = caplog.records[-1]
    return json.loads(rec.getMessage()), rec


class TestToText:
    def test_none_returns_none(self):
        assert _to_text(None) is None

    def test_plain_string_returns_as_is_and_not_json_quoted(self):
        s = _to_text("hello", limit=10)
        assert s == "hello"

    def test_bytes_decodes_utf8(self):
        s = _to_text("héllo".encode("utf-8"), limit=10)
        assert s == "héllo"

    def test_json_serializable_mapping_to_compact_json_string(self):
        val = {"a": 1, "b": "x"}
        s = _to_text(val, limit=100)
        assert isinstance(s, str)
        # compact separators expected (no spaces)
        assert s in ('{"a":1,"b":"x"}', '{"b":"x","a":1}')

    def test_exactly_at_limit_is_not_truncated(self):
        base = "x" * 50
        s = _to_text(base, limit=50)
        assert s == base

    def test_below_limit_is_not_truncated(self):
        base = "y" * 49
        s = _to_text(base, limit=50)
        assert s == base

    def test_above_limit_is_truncated_with_suffix_and_correct_count(self):
        base = "z" * 60
        limit = 50
        s = _to_text(base, limit=limit)
        assert s.startswith("z" * limit)
        assert s[limit:].startswith("...(+")
        assert s.endswith(" chars)")
        import re as _re
        m = _re.search(r"\.\.\.\(\+(\d+) chars\)$", s)
        assert m and int(m.group(1)) == len(base) - limit

    def test_truncation_with_json_stringified_input(self):
        val = {"a": "x" * 120}
        limit = 80
        s = _to_text(val, limit=limit)
        assert isinstance(s, str)
        assert s[:limit]
        assert s[limit:].startswith("...(+")
        assert s.endswith(" chars)")


class TestMessageLogger:
    @staticmethod
    def setup_method():
        # Our structured logger writes under the "plainera" logger
        logging.getLogger("plainera").setLevel(logging.INFO)

    def test_emits_structured_payload_and_redacts_auth(self, caplog):
        caplog.set_level(logging.INFO, logger="plainera")
        message_logger(
            "custom_event",
            level=LogLevel.WARNING,
            args={"authorization": "abc", "other": 1},
        )
        payload, rec = _last_json(caplog)

        assert payload["event"] == "custom_event"
        assert payload["level"] == "warning"         # lower-cased level name
        assert payload["logger_type"] == "inline"    # default passthrough
        assert payload["args"]["authorization"] == "[REDACTED]"
        assert payload["args"]["other"] == 1
        # Caller function auto-captured
        assert payload["function"] == "test_emits_structured_payload_and_redacts_auth"
        # Sanity: we actually logged via the expected logger
        assert rec.name == "plainera"

    def test_maps_details_to_info_and_truncates_long_values(self, caplog):
        caplog.set_level(logging.INFO, logger="plainera")
        big = {"a": "x" * 2100}
        message_logger("evt", details=big)

        payload, _ = _last_json(caplog)
        info_field = payload["info"]

        assert isinstance(info_field, str)
        # be whitespace-tolerant: '"a":   "'
        assert re.search(r'"a":\s*"', info_field)

        # still confirm truncation hint is present
        assert "…(" in info_field or "...(" in info_field
        assert info_field.endswith("chars)")

    def test_details_non_json_serializable_falls_back_to_str(self, caplog):
        caplog.set_level(logging.INFO, logger="plainera")

        class Unserializable:
            pass

        message_logger("evt", details=Unserializable())
        payload, _ = _last_json(caplog)
        assert isinstance(payload["info"], str)
        assert payload["info"]  # non-empty

    @pytest.mark.parametrize(
        "caller,level_name",
        [
            (lambda: log_debug("evt-dbg"), "debug"),
            (lambda: log_info("evt-inf"), "info"),
            (lambda: log_warning("evt-warn"), "warning"),
            (lambda: log_error("evt-err"), "error"),
        ],
    )
    def test_level_helper_shorthands(self, caplog, caller, level_name):
        caplog.set_level(logging.DEBUG, logger="plainera")
        caller()
        payload, _ = _last_json(caplog)
        assert payload["level"] == level_name

    def test_custom_logger_type_passthrough(self, caplog):
        caplog.set_level(logging.INFO, logger="plainera")
        message_logger("evt", level=LogLevel.INFO, logger_type="audit")
        payload, _ = _last_json(caplog)
        assert payload["logger_type"] == "audit"

    def test_args_none_and_details_none_are_handled(self, caplog):
        caplog.set_level(logging.INFO, logger="plainera")
        message_logger("evt")  # defaults only
        payload, _ = _last_json(caplog)
        # args omitted → null; details omitted → info null
        assert payload.get("args") is None
        assert payload.get("info") is None
