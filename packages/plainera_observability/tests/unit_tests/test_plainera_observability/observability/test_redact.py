import copy
import json

import pytest

from observability.observability.redact import _scrub_str, scrub


class Test_ScrubStr:
    def test__scrub_str_redacts_bearer_and_basic_tokens(self):
        assert _scrub_str("Bearer abcdefghijkLMNOP") == "[REDACTED]"
        out = _scrub_str("note=ok Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ== end")
        assert out == "note=ok [REDACTED] end"

class TestScrub:

    def test_scrub_is_idempotent(self):
        data = {"password": "p", "note": "Bearer abcdefghijk"}
        once = scrub(data)
        twice = scrub(once)
        assert once == twice


    def test_scrub_does_not_mutate_input(self):
        original = {"Authorization": "Bearer abcdefghijk", "nested": {"secret": "s"}}
        snapshot = copy.deepcopy(original)
        out = scrub(original)
        # original unchanged
        assert original == snapshot
        # result redacted
        assert out["Authorization"] == "[REDACTED]"
        assert out["nested"]["secret"] == "[REDACTED]"


    def test_scrub_preserves_key_casing_but_matches_case_insensitively(self):
        data = {"X-API-Key": "k", "x-api-key": "k2", "Api_Key": "k3"}
        out = scrub(data)
        # all should be redacted, keys unchanged
        assert out["X-API-Key"] == "[REDACTED]"
        assert out["x-api-key"] == "[REDACTED]"
        assert out["Api_Key"] == "[REDACTED]"


    def test_scrub_handles_lists_and_tuples_and_strings(self):
        data = [
            {"Password": "p1"},
            ("Bearer abcdefghijk", {"token": "t"}),
            "ok",
        ]
        out = scrub(data)
        # list preserved
        assert isinstance(out, list)
        # inner tuple preserved
        assert isinstance(out[1], tuple)
        # redactions
        assert out[0]["Password"] == "[REDACTED]"
        assert out[1][0] == "[REDACTED]"
        assert out[1][1]["token"] == "[REDACTED]"
        assert out[2] == "ok"


    def test_scrub_passes_through_non_string_scalars_and_bytes(self):
        data = {"n": 123, "b": b"bytes", "ok": True, "f": 1.25}
        out = scrub(data)
        assert out == data  # unchanged types/values


    def test_scrub_redacts_token_like_values_even_when_key_is_not_sensitive(self):
        # value contains a token pattern but key is benign -> still scrubbed
        text = "User said: Bearer abcdefghijkLMNOP while testing"
        data = {"message": text}
        out = scrub(data)
        assert out["message"].count("[REDACTED]") == 1
        assert "Bearer" not in out["message"]


    @pytest.mark.parametrize(
        "payload",
        [
            {"Authorization": "[REDACTED]"},
            {"password": "[REDACTED]"},
            {"nested": {"token": "[REDACTED]"}},
        ],
    )
    def test_scrub_leaves_existing_placeholder_intact(self, payload):
        out = scrub(payload)
        # no change to already-redacted sentinel values
        assert json.dumps(out) == json.dumps(payload)
