from plainera_unacronym.nlp import AcronymDetectorConfig
from plainera_unacronym.nlp.plugins.activation import autodetect_domains


def test_autodetect_domains_can_enable_legal_and_structural_reference_together():
    cfg = AcronymDetectorConfig()
    text = 'In this Agreement, "Services" shall mean the services described in Schedule A.'
    auto = autodetect_domains(text, cfg)
    assert "legal" in auto
    assert "structural_reference" in auto

def test_autodetect_domains_can_enable_bio_and_structural_reference_together():
    cfg = AcronymDetectorConfig()
    text = (
        "Methods. Section 1 describes the assay design. "
        "RT-qPCR was used to quantify mRNA expression. Appendix A contains primer sequences."
    )
    auto = autodetect_domains(text, cfg)
    assert "bio" in auto
    assert "structural_reference" in auto

def test_autodetect_domains_can_enable_legal_and_bio_together():
    cfg = AcronymDetectorConfig()
    text = (
        '"Services" shall mean the laboratory analysis services. '
        "The Methods used RT-qPCR to quantify mRNA expression."
    )
    auto = autodetect_domains(text, cfg)
    assert "legal" in auto
    assert "bio" in auto

def test_autodetect_domains_can_enable_legal_bio_and_structural_reference_together():
    cfg = AcronymDetectorConfig()
    text = (
        'In this Agreement, "Services" shall mean the laboratory analysis services '
        'described in Schedule A. Section 2 sets out the workflow. '
        "Methods included RT-qPCR to quantify mRNA expression."
    )
    auto = autodetect_domains(text, cfg)
    assert "legal" in auto
    assert "bio" in auto
    assert "structural_reference" in auto
