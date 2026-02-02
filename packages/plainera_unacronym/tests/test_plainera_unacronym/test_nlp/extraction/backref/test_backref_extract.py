import pytest

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import normalize_definition
from plainera_unacronym.nlp.extraction.backref.spans import best_span_by_initials
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.backref.extract import _candidate_from_prev_sentence, extract_sentence_backrefs
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym


class TestCandidateFromPrevSentenceIntegration:
    def test_happy_path_returns_normalised_candidate(self):
        cand0 = best_span_by_initials("SSO", "We use Single sign-on for authentication", max_chars=200)
        print("cand0:", cand0)

        cand1 = tighten_label_by_acronym(cand0, "SSO", bridges=set(BRIDGES_DEFAULT))
        cand1 = normalize_definition(cand1)
        print("cand1:", cand1)
        cfg = ExtractionConfig()
        prev = "We use Single sign-on for authentication."
        out = _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )
        assert out == "Single sign-on"

    def test_returns_none_for_blank_prev_text(self):
        cfg = ExtractionConfig()
        assert _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text="   \n\t",
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        ) is None

    def test_filters_out_very_long_prev_sentence(self):
        cfg = ExtractionConfig()
        prev = "Word " * 1000  # collapsed length likely > max_chars*3
        assert _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=50,
            require_two_words=True,
        ) is None

    def test_strips_trailing_punctuation_before_span_search(self):
        cfg = ExtractionConfig()
        prev = "We use Single sign-on for authentication...   "
        out = _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )
        assert out == "Single sign-on"

    def test_returns_none_when_no_initials_span_found(self):
        cfg = ExtractionConfig()
        prev = "Nothing matching here."
        assert _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        ) is None

    def test_rejects_candidate_without_letters(self):
        cfg = ExtractionConfig()
        # This sentence can never produce a valid initials span with letters for "A"
        prev = "123 456 789."
        assert _candidate_from_prev_sentence(
            acr_norm="A",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=False,
        ) is None

    def test_rejects_candidate_equal_to_acronym(self):
        cfg = ExtractionConfig()
        prev = "We use SSO for authentication."
        assert _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=False,
        ) is None

    def test_rejects_candidate_over_max_chars(self):
        cfg = ExtractionConfig()
        prev = "We use Single sign-on for authentication."
        assert _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=5,  # far too small for "Single sign-on"
            require_two_words=False,
        ) is None

    def test_initials_match_validation_blocks_false_positive(self):
        cfg = ExtractionConfig()
        # "Lots Of Llamas" gives initials LOL; acronym LLO should not validate
        prev = "We use Lots Of Llamas in testing."
        assert _candidate_from_prev_sentence(
            acr_norm="LLO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        ) is None



def test_sentence_backref_ignores_single_letter_acronyms(self):
    cfg = ExtractionConfig()
    text = "We use Authentication. A is sometimes used as shorthand."
    firsts = {
        "A": FirstOccurrence(acronym="A", start_offset=text.index("A is"), end_offset=text.index("A is") + 1,
                             confidence=0.9, normalized_key="A")
    }

    out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
    assert out == []
