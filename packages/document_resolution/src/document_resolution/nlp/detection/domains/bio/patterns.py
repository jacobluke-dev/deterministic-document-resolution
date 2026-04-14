import re
from functools import lru_cache

from document_resolution.nlp.detection.domains.bio.config import BIO_GREEK


@lru_cache(maxsize=1)
def bio_pattern() -> re.Pattern[str]:
    # Specific bio forms FIRST so they win before the generic 'camel' branch.
    rna = r"(?:m|t|r|mi|si|sg|g|lnc|sn|sc|pi|sh|nc)RNA"
    dna = r"(?:cDNA|gDNA)"

    camel = r"(?:[A-Z][a-z]+[A-Z0-9][A-Za-z0-9]{1,6}|[A-Z]{2,6}\d{0,2})"
    cytokine = rf"(?:IL-\d{{1,3}}|TNF-[{BIO_GREEK}]|IFN-[{BIO_GREEK}]|TGF-[{BIO_GREEK}\d])"
    virus = r"(?:SARS-CoV-2|MERS-CoV|H\d{1,2}N\d{1,2})"
    prime = r"(?:[35][\'′″]-?\s?UTR)"
    # Whole-token preference: RNA/DNA | cytokine | virus | prime | generic camel
    return re.compile(rf"(?P<bio>{rna}|{dna}|{cytokine}|{virus}|{prime}|{camel})")
