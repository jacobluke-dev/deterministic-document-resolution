from types import SimpleNamespace

import pytest

from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.heuristics.general import at_sentence_boundary, blacklist_context_drop, is_all_caps_heading, \
    shouty_phrase_drop


def _extract(text_with_caret: str) -> tuple[str, int]:
    """Turn '...^...' into (text, pos)."""
    pos = text_with_caret.index("^")
    return text_with_caret.replace("^", ""), pos


def _extract_span(s: str) -> tuple[str, int, int]:
    """
    Use [ ... ] to mark (start, end) in the sample.
    Returns: (clean_text, start, end)
    """
    pre, rest = s.split("[", 1)
    inside, post = rest.split("]", 1)
    text = pre + inside + post
    start = len(pre)
    end = start + len(inside)
    return text, start, end

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
    object.__setattr__(cfg, "soft_blacklist", frozenset({"IT", "AM"}))
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg



def make_cfg(allow_chars: str = "-&/._") -> DetectorConfig:
    """
    Build a valid DetectorConfig for tests.
    Adjust defaults if your real class has more fields.
    """
    return DetectorConfig(min_len=2, max_len=12, allow_chars=allow_chars)


def _patch_near(monkeypatch, left_gap: int, right_gap: int):
    """Simulate 'near' using max gap thresholds."""
    def comma_near_left(text: str, s: int) -> bool:
        i = s - 1
        spaces = 0
        while i >= 0 and text[i].isspace():
            spaces += 1
            i -= 1
        return i >= 0 and text[i] == "," and spaces <= left_gap

    def exclam_near_right(text: str, e: int) -> bool:
        # Distance from e to the next '!' (counts every char between, incl. the next word).
        i = e
        dist = 0
        n = len(text)
        while i < n and text[i] != "!":
            i += 1
            dist += 1
        return i < n and dist <= right_gap

    monkeypatch.setattr("plainera_unacronym.nlp.heuristics.general._comma_near_left", comma_near_left, raising=True)
    monkeypatch.setattr("plainera_unacronym.nlp.heuristics.general.exclam_near_right", exclam_near_right, raising=True)


class TestShoutyPhraseDrop:

    @pytest.fixture(autouse=True)
    def _cfg(self):
        return make_cfg()

    @staticmethod
    def _extract_span(s: str):
        """Use [ ... ] to mark (start, end). Return (text, surface, s, e)."""
        pre, rest = s.split("[", 1)
        inside, post = rest.split("]", 1)
        text = pre + inside + post
        s_idx = len(pre)
        e_idx = s_idx + len(inside)
        return text, inside, s_idx, e_idx

    @pytest.mark.parametrize(
        "sample, expected",
        [
            ("Well, [ALRIGHTY] THEN!", True),             # canonical: comma + ALLCAPS + ALLCAPS + !
            ("Well,    [ALRIGHTY]   THEN!", True),        # extra spaces
            ("OK, [MOVE] ALONG!", True),                   # different words
            ("Well, [ALRIGHTY] THEN!!", True),            # multiple !
            ("Well [ALRIGHTY] THEN!", False),             # no comma near left
            ("Well, [ALRIGHTY] THEN.", False),            # no exclamation near right
            ("Well, [Alrighty] THEN!", False),            # first word not ALLCAPS
            ("Well, [ALRIGHTY] Then!", False),            # next word not ALLCAPS
            ("Well, [ALRIGHTY] OK!", False),              # next word too short (<3)
            ("Well, [ALRIGHTY] X!", False),               # length 1 next word
            ("Well, [ALRIGHTY] 123!", False),             # next token not letters
        ],
    )
    def test_various(self, sample, expected, _cfg):
        text, surface, s, e = self._extract_span(sample)
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is expected

    def test_handles_unicode_caps_next_word(self, _cfg):
        # Next word with Unicode uppercase letters should count
        sample = "Well, [BRAVO] ÉTUDE!"
        text, surface, s, e = self._extract_span(sample)
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is True

    def test_variants_any_two_words(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=4, right_gap=40)  # permissive for shape tests
        for sample in [
            "Well, [ALRIGHTY] THEN!",
            "Fine, [MOVE] ALONG!",
            "Okay, [RIGHT] NOW!",
            "Listen,   [STOP]   THAT!",
            "Hey, [PLEASE] CLAP!",
        ]:
            text, surface, s, e = self._extract_span(sample)
            assert shouty_phrase_drop(surface, text, s, e, _cfg) is True

    def test_rejects_if_next_word_not_all_caps_or_too_short(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=4, right_gap=40)
        # Next word not ALL CAPS
        t1 = "Well, [ALRIGHTY] Then!"
        # Next word length < 3
        t2 = "Well, [ALRIGHTY] OK!"
        for sample in [t1, t2]:
            text, surface, s, e = self._extract_span(sample)
            assert shouty_phrase_drop(surface, text, s, e, _cfg) is False

    @pytest.mark.parametrize("right_gap,sample,expected", [
        (3, "Well, [GO] UP!", True),    # distance e->'!' = 1(space)+2(UP) = 3
        (3, "Well, [GO] NOW!", False),  # 1 + 3 = 4 > 3
        (4, "Well, [GO] NOW!", True),   # 4 allowed
    ])
    def test_right_gap_enforced(self, monkeypatch, right_gap, sample, expected, _cfg):
        _patch_near(monkeypatch, left_gap=3, right_gap=right_gap)
        text, surface, s, e = self._extract_span(sample)
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is expected

    @pytest.mark.parametrize("left_gap,sample,expected", [
        (3, "Well,   [GO] UP!", True),   # 3 spaces after comma OK
        (3, "Well,    [GO] UP!", False), # 4 spaces > 3
        (4, "Well,    [GO] UP!", True),  # 4 allowed
    ])
    def test_left_gap_enforced(self, monkeypatch, left_gap, sample, expected, _cfg):
        _patch_near(monkeypatch, left_gap=left_gap, right_gap=10)
        text, surface, s, e = self._extract_span(sample)
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is expected

    def test_both_gaps_tight_fail(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=2, right_gap=2)
        text, surface, s, e = self._extract_span("Well,   [GO] UP!")
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is False

    def test_allowed_internal_separators_pass(self, monkeypatch):
        _patch_near(monkeypatch, left_gap=3, right_gap=40)
        # ampersand and slash allowed
        for sample in [
            "Well, [R&D] TEAM!",      # &
            "Well, [GPU/CPU] CLUB!",  # /
            "Well, [MOVE-ON] THEN!",  # -
            "Well, [A_B] TEST!",      # _
            "Well, [A.B] TEST!",      # .
        ]:
            text, surface, s, e = self._extract_span(sample)
            _cfg = make_cfg(allow_chars="-&/._")
            assert shouty_phrase_drop(surface, text, s, e, _cfg) is True

    def test_disallowed_separator_fails(self, monkeypatch):
        _patch_near(monkeypatch, left_gap=3, right_gap=40)
        # '-' not allowed here -> fails ALL-CAPS word check
        text, surface, s, e = self._extract_span("Well, [MOVE-ON] THEN!")
        _cfg = make_cfg(allow_chars="&/._")
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is False

    def test_unicode_all_caps_next_word_ok(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=3, right_gap=40)
        text, surface, s, e = self._extract_span("Well, [BRAVO] ÉTUDE!")
        assert shouty_phrase_drop(surface, text, s, e, _cfg) is True

    @pytest.mark.parametrize("sample, expected", [
        ("\n[INTRODUCTION]\nBody", True),                      # simple all-caps, >=6 letters
        ("\n   [   HEADING   ]   \n", True),                   # leading/trailing spaces
        ("\n[API V2 OVERVIEW]\n", True),                       # digits ignored, letters all caps
        ("\n[End. The]\n", False),                             # mixed case -> False
        ("\n[FAQ]\n", False),                                  # <6 letters -> False
        ("\n[----]\n", False),                                 # no letters -> False
        ("\n[CHAPÉU]\n", True),                                # Unicode uppercase letters
        ("\n[ΠΡΟΛΟΓΟΣ]\n", True),                              # Greek uppercase
        ("Prev\n.. [NOT all CAPS]\nNext\n", False),            # mixed case long enough
        ("# Intro\n[INTRODUCTION line continues]\n", False),   # selection mid-line still but contains lowercase
        ("Last line with \n[OVERVIEW]\n(no trailing newlne)", True), # \n \n
    ])
    def test_various(self, sample: str, expected: bool):
        text, start, end = _extract_span(sample)
        assert is_all_caps_heading(text, start, end) is expected

    def test_selection_inside_line_not_full_line(self):
        # Start/end are in the middle of the line; function should still consider the whole line.
        sample = "##   PRE [INTRODUCTION]  POST"
        text, start, end = _extract_span(sample)
        assert is_all_caps_heading(text, start + 2, end - 2) is True  # even a subspan on that line


class TestAtSentenceBoundary:
    @pytest.mark.parametrize(
        "sample, expected",
        [
            ("^Hello", True),                                   # start of doc
            ("Hello. ^World", True),                            # period + space
            ("Hello.^World", True),                             # period, no space
            ('He said: "Go." ^Then', True),                     # period + closing quote
            ("Do it now.) ^Then", True),                        # period + ) closer
            ('"Go?!" ^Next', True),                             # mixed ?!
            ("Wait… ^Go", True),                                # unicode ellipsis
            ("Hello ^brave world", False),                      # mid-sentence
            ("Wait—^no", False),                                # em dash is not a terminator
            ("Hello\n^World", False),                           # newline alone isn't a boundary
            ('He said "hello" ^and left.', False),              # closer without terminator
            ('He said: “Go.”^Then', True),                      # curly closer right before next
            ('Done! ”^Next', True),                             # space + curly quote closer
            ('Done! »^Next', True),                             # guillemet closer
            ("U.S.^Policy", True),  # dotted initialism, no space
            ("Hello\t\t^World", False),  # tabs as whitespace
            ("Dog.     ^The", True)
        ],
    )
    def test_various(self, sample: str, expected: bool):
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos) is expected

    def test_whitespace_then_closers_then_terms(self):
        # e.g. space(s) -> closers -> terminator cluster -> next token
        sample = 'He: "Done?!"   ^Next'
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos) is True

    def test_many_spaces_after_terminator(self):
        sample = "End.     ^Next"
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos) is True

    @pytest.mark.parametrize("sample", [
        "End. ^The",
        "End! ^The",
        "End? ^The",
        "End?! ^The",
        "End!   ^The",
        'End! ”^The',
        'End? ) ”  ^The',
    ])
    def test_true_cases(self, sample):
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos)


class TestBlacklistContextDrop:
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
        assert blacklist_context_drop("THEN",     text, s2, e2, cfg) is True

    # 1b) ALL-CAPS heading drop (if your is_all_caps_heading() recognizes it)
    def test_all_caps_heading_drops(self, span):
        text = "INTRODUCTION\nWe begin here."
        s, e = span(text, "INTRODUCTION")
        # If your is_all_caps_heading() returns True for a standalone caps line,
        # this will be True. If not, adjust or skip this test.
        assert blacklist_context_drop("INTRODUCTION", text, s, e, mk_cfg()) in (True,)

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
