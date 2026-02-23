from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.detection.heuristics.context import blacklist_context_drop


def mk_cfg(**overrides) -> DetectorConfig:
    """
    Build a config with pragmatic defaults for this unit under test.
    We ensure 'IT' and 'AM' are in blacklist (so token-specific rules run),
    and include some common non-acronym uppercase tokens.
    """
    base = {
        "non_acronym_upper": frozenset({"OK", "LTD", "PLC", "NO"}),
    }

    cfg = DetectorConfig()

    object.__setattr__(cfg, "non_acronym_upper", base["non_acronym_upper"])
    object.__setattr__(cfg, "blacklist", frozenset({"IT", "AM"}))
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


class TestBlacklistContextDrop:
    def test_all_caps_heading_drops(self, span, monkeypatch):
        import plainera_unacronym.nlp.detection.heuristics.context as ctx

        monkeypatch.setattr(ctx, "_drop_all_caps_heading", lambda *args, **kwargs: True)
        text = "INTRODUCTION\nWe begin here."
        s, e = span(text, "INTRODUCTION")
        assert blacklist_context_drop("INTRODUCTION", text, s, e, mk_cfg()) is True

    def test_ok_followed_by_capitalised_kept(self, span):
        text = "OK This is fine."
        s, e = span(text, "OK")
        assert blacklist_context_drop("OK", text, s, e, mk_cfg()) is False

    # 0) Definition contexts should NOT drop
    def test_paren_definition_right(self, span):
        text = "IT (Information Technology) leads the team."
        s, e = span(text, "IT")
        assert blacklist_context_drop("IT", text, s, e, mk_cfg()) is False

    def test_stands_for_context(self, span):
        text = "IT stands for Information Technology in most orgs."
        s, e = span(text, "IT")
        assert blacklist_context_drop("IT", text, s, e, mk_cfg()) is False

    # 1) Shouty ALL-CAPS phrase should drop both words
    def test_shouty_phrase_drops(self, span):
        text = "Jacob says, ALRIGHTY THEN! We’ll reconvene."
        s1, e1 = span(text, "ALRIGHTY")
        s2, e2 = span(text, "THEN")
        cfg = mk_cfg()
        assert blacklist_context_drop("ALRIGHTY", text, s1, e1, cfg) is True
        assert blacklist_context_drop("THEN", text, s2, e2, cfg) is True

    # 3) Non-acronym uppercase tokens
    def test_ok_followed_by_punctuation_drops(self, span):
        text = "OK, then let’s go."
        s, e = span(text, "OK")
        assert blacklist_context_drop("OK", text, s, e, mk_cfg()) is True

    def test_ltd_followed_by_lowercase_word_drops(self, span):
        text = "Acme LTD announced results."
        s, e = span(text, "LTD")
        assert blacklist_context_drop("LTD", text, s, e, mk_cfg()) is True

    # 4) Token-specific polysemes
    def test_it_pronoun_at_sentence_start_drops(self, span):
        text = "IT was fine. (Later…) "
        s, e = span(text, "IT")
        assert blacklist_context_drop("IT", text, s, e, mk_cfg()) is True

    def test_it_in_definition_context_kept(self, span):
        text = "The NHS IT (Information Technology) team met."
        s, e = span(text, "IT")
        assert blacklist_context_drop("IT", text, s, e, mk_cfg()) is False

    def test_am_time_of_day_drops(self, span):
        text = "Meeting at 07:30 AM today."
        s, e = span(text, "AM")
        assert blacklist_context_drop("AM", text, s, e, mk_cfg()) is True

    def test_am_after_I_with_sentence_boundary_drops(self, span):
        text = "I AM going now."
        s, e = span(text, "AM")
        assert blacklist_context_drop("AM", text, s, e, mk_cfg()) is True

    def test_am_as_noun_kept(self, span):
        text = "AM radio is still a thing."
        s, e = span(text, "AM")
        # Not time-of-day, not after boundary 'I', so generic fallback unlikely to trigger
        assert blacklist_context_drop("AM", text, s, e, mk_cfg()) is False

    # 5) Generic fallback: sentence-start + next word lowercase
    def test_generic_sentence_start_next_lowercase_drops(self, span):
        text = "NO worries — it’s sorted."
        s, e = span(text, "NO")
        assert blacklist_context_drop("NO", text, s, e, mk_cfg()) is True

    # Control: a normal, real acronym should not be dropped
    def test_real_acronym_kept(self, span):
        text = "We met the NHS team today."
        s, e = span(text, "NHS")
        # Not in blacklist/non_acronym_upper → early return False (keep)
        assert blacklist_context_drop("NHS", text, s, e, mk_cfg()) is False
