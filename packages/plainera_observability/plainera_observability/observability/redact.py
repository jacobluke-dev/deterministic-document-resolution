from plainera_observability.config import SENSITIVE_KEYS


def scrub(obj):
    if isinstance(obj, dict):
        return {
            k: ("[REDACTED]" if k.lower() in SENSITIVE_KEYS else scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        t = [scrub(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj
