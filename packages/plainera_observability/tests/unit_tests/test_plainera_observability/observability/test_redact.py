from plainera_observability.observability.redact import scrub


def test_scrub_redacts_common_keys_case_insensitive():
    data = {"Authorization": "Bearer abc", "password": "p", "nested": {"X-API-Key": "k"}}
    out = scrub(data)
    assert out["Authorization"] == "[REDACTED]"
    assert out["password"] == "[REDACTED]"
    assert out["nested"]["X-API-Key"] == "[REDACTED]"
