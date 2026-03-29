from __future__ import annotations

from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.extraction.defined_terms.execute import detect_and_resolve_terms


def _mention_keys(det_res) -> list[str]:
    return [m.normalized_key for m in det_res.mentions]


class TestDefinedTermUnquotedPolicy:
    def test_unquoted_mentions_detected_by_default_in_legal_text(self):
        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Services" means the software development services described in Schedule A.
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Agreement shall commence on the Effective Date.
        The Services shall be performed with reasonable skill and care.
        """.strip()

        det_res, extr = detect_and_resolve_terms(text)

        mention_keys = _mention_keys(det_res)

        assert "agreement" in mention_keys
        assert "services" in mention_keys
        assert "effective_date" in mention_keys

    def test_unquoted_mentions_remain_conservative_when_policy_is_never(self):
        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Services" means the software development services described in Schedule A.
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Agreement shall commence on the Effective Date.
        The Services shall be performed with reasonable skill and care.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            enabled_domains=frozenset({"legal"}),
            auto_detect_domains=False,
            unquoted_capitalised_terms_policy="never",
        )

        det_res, extr = detect_and_resolve_terms(text, det_cfg=det_cfg)

        mention_keys = _mention_keys(det_res)

        assert "agreement" not in mention_keys
        assert "services" not in mention_keys
        assert "effective_date" not in mention_keys

    def test_unquoted_mentions_do_not_regress_false_positive_guards(self):
        text = """
        "Project" means the migration project described in the statement of work.

        The Team met on Tuesday.
        The Company presented an update.
        The Project moved to the next phase.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            enabled_domains=frozenset(),
            auto_detect_domains=False,
            unquoted_capitalised_terms_policy="legal_only",
        )

        det_res, extr = detect_and_resolve_terms(text, det_cfg=det_cfg)

        mention_keys = _mention_keys(det_res)

        assert ["project"] not in mention_keys
        assert "team" not in mention_keys
        assert "company" not in mention_keys
