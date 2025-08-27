SENSITIVE_KEYS = {k.lower() for k in {"password","secret","token","authorization","x-api-key","api_key","bearer"}}

def scrub(obj):
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if k.lower() in SENSITIVE_KEYS else scrub(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [scrub(v) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj
