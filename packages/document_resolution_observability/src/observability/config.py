import re

REQ_ID_HEADER: str = "X-Request-ID"

SENSITIVE_KEYS: set[str] = {
    k.lower() for k in {"password", "secret", "token", "authorization", "x-api-key", "api_key", "bearer"}
}

TOKEN_PATS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{10,}"),
    re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}"),
    re.compile(r"(?i)\b(?:api|access|secret|auth|session)[-_ ]?key\s*[:=]\s*[A-Za-z0-9._-]{10,}"),
]
