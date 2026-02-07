from plainera_unacronym.nlp.common.types import DetectorResult, DetectorConfig, Occurrence, FirstOccurrence
from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.detection.cleanup.post import post_detect_cleanup


def _fo(cfg: DetectorConfig, acr: str, s: int, e: int, conf: float = 0.9) -> FirstOccurrence:
    k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
    assert k
    return FirstOccurrence(acronym=acr, start_offset=s, end_offset=e, confidence=conf, normalized_key=k)


def _occ(cfg: DetectorConfig, acr: str, s: int, e: int, conf: float = 0.9) -> Occurrence:
    k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
    return Occurrence(
        acronym=acr,
        start_offset=s,
        end_offset=e,
        confidence=conf,
        context_window=(max(0, s - 20), e + 20),
        normalized_key=k,
        reasons=None,
    )


def test_cleanup_drops_rna_inside_mrna_by_contained_suffix():
    cfg = DetectorConfig()
    text = "We measured mRNA expression."

    occ_mrna = _occ(cfg, "mRNA", 12, 16)
    occ_rna  = _occ(cfg, "RNA", 13, 16)  # contained, strict suffix

    det = DetectorResult(
        unique_acronyms={
            occ_mrna.normalized_key: _fo(cfg, "mRNA", 12, 16),
            occ_rna.normalized_key: _fo(cfg, "RNA", 13, 16),
        },
        occurrences=[occ_mrna, occ_rna],
    )

    cleaned, summary, dropped = post_detect_cleanup(text, det, cfg)

    assert len(cleaned.occurrences) == 1
    assert cleaned.occurrences[0].acronym == "mRNA"
    assert "RNA" not in {fo.acronym for fo in cleaned.unique_acronyms.values()}
    assert any(d.acronym == "RNA" and d.rule == "contained_suffix" for d in dropped), dropped


def test_cleanup_does_not_drop_when_not_suffix_even_if_contained():
    cfg = DetectorConfig()
    text = "Token ABCD appears."

    # Contained span, but not suffix relationship -> should be retained
    occ_outer = _occ(cfg, "ABCD", 6, 10)
    occ_inner = _occ(cfg, "BC", 7, 9)  # contained, but "ABCD" does not end with "BC"? (it ends with "CD")

    det = DetectorResult(
        unique_acronyms={
            occ_outer.normalized_key: _fo(cfg, "ABCD", 6, 10),
            occ_inner.normalized_key: _fo(cfg, "BC", 7, 9),
        },
        occurrences=[occ_outer, occ_inner],
    )

    cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

    assert {o.acronym for o in cleaned.occurrences} == {"ABCD", "BC"}
    assert dropped == []


def test_cleanup_drops_token_before_paren_suffix_messenger_rna_mrna():
    cfg = DetectorConfig()
    text = "messenger RNA (mRNA) has been developed,"

    occ_rna = _occ(cfg, "RNA", 10, 13, conf=0.6)
    occ_mrna = _occ(cfg, "mRNA", 15, 19, conf=0.85)

    det = DetectorResult(
        unique_acronyms={
            occ_rna.normalized_key: _fo(cfg, "RNA", 10, 13, conf=0.6),
            occ_mrna.normalized_key: _fo(cfg, "mRNA", 15, 19, conf=0.85),
        },
        occurrences=[occ_rna, occ_mrna],
    )

    cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

    assert {o.acronym for o in cleaned.occurrences} == {"mRNA"}
    assert "RNA" not in {fo.acronym for fo in cleaned.unique_acronyms.values()}
    assert any(d.acronym == "RNA" and d.rule == "token_before_paren_suffix" for d in dropped), dropped
