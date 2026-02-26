import pytest

from public_api.core.auth.api_keys import parse_api_key


def test_parse_accepts_valid():
    raw = "uak_live_AbCDef1234_abcdefghijklmnopqrstuvwxyzABCDEFGHijklmnop123456"
    out = parse_api_key(raw, allow_prefixes={"live", "test"})
    assert out is not None
    assert out.prefix == "live"
    assert out.key_id == "AbCDef1234"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "uak__x_y",
        "uak_badprefix_x_y",
        "uak_live_short_short",
        "uak_live_@@@_secret",
    ],
)
def test_parse_rejects_malformed(raw):
    out = parse_api_key(raw, allow_prefixes={"live", "test"})
    assert out is None
