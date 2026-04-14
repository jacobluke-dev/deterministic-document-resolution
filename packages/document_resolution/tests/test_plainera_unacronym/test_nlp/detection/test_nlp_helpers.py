import math
import re

import pytest
from document_resolution.nlp.common.types import AcronymDetectorConfig, FirstOccurrence
from document_resolution.nlp.detection.nlp_helpers import _round_sig, cfg_fingerprint, top_n_values


class TestCfgFingerprint:
    def test_returns_12_lower_hex(self):
        fp = cfg_fingerprint(AcronymDetectorConfig())
        assert len(fp) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", fp), f"not lower-hex: {fp}"

    def test_stable_for_same_values(self):
        a = AcronymDetectorConfig()
        b = AcronymDetectorConfig()
        assert cfg_fingerprint(a) == cfg_fingerprint(b)

    def test_ignores_unlisted_fields(self):
        base = AcronymDetectorConfig()
        # Change fields NOT included in the snapshot:
        changed = AcronymDetectorConfig(
            min_len=99,  # ignored
            max_len=123,  # ignored
            require_caps_ratio=0.123,  # ignored
            require_caps_ratio_mixed=0.321,  # ignored
            enable_dotted=True,  # ignored
            debug_reasons=True,  # ignored
            enable_mixed_case=False,  # ignored
            locale="en_US",  # ignored
            non_acronym_upper=frozenset({"OK", "PM", "ETC"}),  # ignored
            blacklist=frozenset({"OF", "IN"}),  # ignored
            domain_cfg={"x": 1},  # ignored
        )
        assert cfg_fingerprint(base) == cfg_fingerprint(changed)

    @pytest.mark.parametrize(
        "mut",
        [
            lambda c: c.__class__(allow_chars=c.allow_chars + "_"),
            lambda c: c.__class__(window_chars=c.window_chars + 1),
            lambda c: c.__class__(min_confidence_default=c.min_confidence_default + 0.01),
            lambda c: c.__class__(dotted_display="preserve" if c.dotted_display == "strip" else "strip"),
            lambda c: c.__class__(enabled_domains=frozenset({"a", "b"})),
        ],
        ids=[
            "allow_chars",
            "window_chars",
            "min_confidence_default",
            "dotted_display",
            "enabled_domains",
        ],
    )
    def test_changes_when_relevant_field_changes(self, mut):
        base = AcronymDetectorConfig()
        changed = mut(base)
        assert cfg_fingerprint(base) != cfg_fingerprint(changed)

    def test_domains_order_insensitive(self):
        a = AcronymDetectorConfig(enabled_domains=frozenset({"b", "a", "c"}))
        b = AcronymDetectorConfig(enabled_domains=frozenset({"c", "b", "a"}))
        assert cfg_fingerprint(a) == cfg_fingerprint(b)


class TestRoundSig:
    @pytest.mark.parametrize(
        "x,sig,expected",
        [
            (0.0, 3, 0.0),  # zero stays zero
            (0.5555, 3, 0.556),  # half-up at 4th dp → 3sf
            (0.9999, 3, 1.0),  # carry to next order
            (0.12345, 3, 0.123),  # trim extra digits
            (0.00012345, 3, 0.000123),  # tiny numbers
            (9.995, 3, 10.0),  # half-up across boundary
            (1.005, 3, 1.01),  # classic float trap avoided
            (12345.0, 3, 12300.0),  # large magnitude down-round
            (-0.5555, 3, -0.556),  # negatives half-up (towards +inf in magnitude)
            (-9.995, 3, -10.0),  # negative boundary
        ],
        ids=[
            "zero",
            "half_up_0.5555",
            "carry_to_one",
            "trim_extra",
            "tiny",
            "boundary_up",
            "classic_trap",
            "large_down",
            "neg_half_up",
            "neg_boundary",
        ],
    )
    def test_round_sig_examples(self, x, sig, expected):
        out = _round_sig(x, sig)
        assert math.isclose(out, expected, rel_tol=0, abs_tol=1e-12)

    @pytest.mark.parametrize("sig", [1, 2, 3, 6])
    def test_idempotent_on_already_rounded_values(self, sig):
        # Values already at the target significant figures should not change if rounded again.
        samples = [1.2, 12.0, 1200.0, 0.34, 0.00340, 9.99, -4.56]
        for x in samples:
            once = _round_sig(x, sig)
            twice = _round_sig(once, sig)
            assert once == twice

    def test_order_preserving_near_boundary(self):
        # Near a rounding threshold, rounding must not flip the order (non-decreasing).
        left = _round_sig(0.9949, 3)  # just below the 0.995 edge
        mid = _round_sig(0.9950, 3)  # at the edge
        right = _round_sig(0.9951, 3)  # just above the edge
        assert left <= mid <= right

    def test_sig_parameter_effect(self):
        x = 0.12345
        assert _round_sig(x, 2) == 0.12
        assert _round_sig(x, 3) == 0.123
        assert _round_sig(x, 4) == 0.1235


def fo(key: str, conf: float, start: int = 0, end: int = 1) -> FirstOccurrence:
    # Minimal, valid FirstOccurrence for tests
    return FirstOccurrence(
        acronym=key,
        start_offset=start,
        end_offset=end,
        occurrence_confidence=conf,
        normalized_key=key,
    )


class TestTopNValues:
    def test_empty_input_returns_empty_list(self):
        assert top_n_values({}) == []

    def test_n_le_zero_returns_empty_list(self):
        data: dict[str, FirstOccurrence] = {"GPU": fo("GPU", 0.9)}
        assert top_n_values(data, n=0) == []
        assert top_n_values(data, n=-3) == []

    def test_basic_sorting_and_shape(self):
        data: dict[str, FirstOccurrence] = {
            "GPU": fo("GPU", 0.95),
            "NLP": fo("NLP", 0.87),
            "API": fo("API", 0.91),
            "UK": fo("UK", 0.60),
        }
        out = top_n_values(data, n=2)
        assert out == [
            {"key": "GPU", "conf": 0.95},
            {"key": "API", "conf": 0.91},
        ]
        assert all(set(item.keys()) == {"key", "conf"} for item in out)

    def test_rounding_to_3dp(self):
        data: dict[str, FirstOccurrence] = {
            "A": fo("A", 0.12345),  # -> 0.123
            "B": fo("B", 0.5555),  # -> 0.556
            "C": fo("C", 0.9999),  # -> 1.0
        }
        out = top_n_values(data, n=3)
        assert out == [
            {"key": "C", "conf": 1.0},
            {"key": "B", "conf": 0.556},
            {"key": "A", "conf": 0.123},
        ]

    def test_n_greater_than_length_returns_all_sorted(self):
        data: dict[str, FirstOccurrence] = {
            "X": fo("X", 0.2),
            "Y": fo("Y", 0.8),
            "Z": fo("Z", 0.5),
        }
        out = top_n_values(data, n=10)
        assert out == [
            {"key": "Y", "conf": 0.8},
            {"key": "Z", "conf": 0.5},
            {"key": "X", "conf": 0.2},
        ]

    def test_ties_are_included_order_not_strictly_asserted(self):
        data: dict[str, FirstOccurrence] = {
            "D": fo("D", 0.9),
            "E": fo("E", 0.9),
            "F": fo("F", 0.8),
        }
        out = top_n_values(data, n=2)
        keys = {item["key"] for item in out}
        confs = {item["conf"] for item in out}
        assert keys == {"D", "E"}
        assert confs == {0.9}
