
REQ_ID_HEADER = "X-Request-ID"
SENSITIVE_KEYS = {k.lower() for k in {
    "password", "secret", "token", "authorization", "x-api-key", "api_key", "bearer"
}}
