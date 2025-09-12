import re
from functools import lru_cache

from plainera_unacronym.domains.bio.config import BIO_GREEK


@lru_cache(maxsize=1)
def bio_pattern() -> re.Pattern[str]:
    camel   = r"(?:[A-Z][a-z]+[A-Z0-9][A-Za-z0-9]{1,6}|[A-Z]{2,6}\d{0,2})"
    cytokine= rf"(?:IL-\d{{1,3}}|TNF-[{BIO_GREEK}]|IFN-[{BIO_GREEK}]|TGF-[{BIO_GREEK}\d])"
    virus   = r"(?:SARS-CoV-2|MERS-CoV|H\dN\d)"
    prime   = r"(?:[35][\'′″]-?\s?UTR)"
    return re.compile(rf"(?P<bio>{cytokine}|{virus}|{prime}|{camel})")
