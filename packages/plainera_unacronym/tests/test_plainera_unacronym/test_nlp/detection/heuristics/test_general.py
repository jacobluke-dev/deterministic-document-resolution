import pytest
from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.common.types import TextSpanTuple
from plainera_unacronym.nlp.detection.heuristics.general import (
    _alpha_len,
    _comma_near_left,
    at_sentence_boundary,
    exclam_near_right,
    is_all_caps_heading,
    is_all_caps_word,
    is_in_caps_interjection_context,
    is_in_caps_interjection_context_prev,
    strip_terminal_plural,
)


def _extract(text_with_caret: str) -> tuple[str, int]:
    """Turn '...^...' into (text, pos)."""
    pos = text_with_caret.index("^")
    return text_with_caret.replace("^", ""), pos


def _extract_span(s: str) -> TextSpanTuple:
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


def make_cfg(allow_chars: str = "-&/._") -> DetectorConfig:
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

    monkeypatch.setattr("plainera_unacronym.nlp.detection.heuristics.general._comma_near_left",
                        comma_near_left,
                        raising=True)
    monkeypatch.setattr("plainera_unacronym.nlp.detection.heuristics.general.exclam_near_right",
                        exclam_near_right,
                        raising=True)


@pytest.fixture(autouse=True)
def _cfg():
    return make_cfg()


def _extract_from_brackets(s: str) -> tuple[str, int]:
    """
    Mark the token start with [ ... ] and return (clean_text, start_index).
    """
    pre, rest = s.split("[", 1)
    inside, post = rest.split("]", 1)
    text = pre + inside + post
    start = len(pre)  # index of the first char of the token
    return text, start

class TestAlphaLen:
    @pytest.mark.parametrize(
        "s, expected",
        [
            ("", 0),
            ("ABC", 3),
            ("A1B2C3", 3),
            ("Hello, world!", 10),   # commas/space/exclamation ignored
            ("R&D", 2),              # ampersand ignored
            ("GPU/CPU", 6),          # slash ignored
            ("MOVE-ON", 6),          # hyphen ignored
            ("\n\t ", 0),            # whitespace only
            ("🙂", 0),               # emoji not alpha
            ("É", 1),                # composed accented letter
            ("e\u0301", 1),          # 'e' + combining acute → only 'e' counts
            ("CHAPÉU", 6),           # accented capitals
            ("ΠΡΟΛΟΓΟΣ", 8),         # Greek
            ("добро", 5),            # Cyrillic
            ("漢字", 2),             # CJK ideographs are letters
            ("ß", 1),                # sharp s
            ("abc—def", 6),          # em dash ignored
        ],
    )
    def test_various_strings(self, s, expected):
        assert _alpha_len(s) == expected


class TestStripTerminalPlural:
    @pytest.mark.parametrize("surface,expected", [
        ("GPUs", "GPU"),          # simple plural
        ("CPU's", "CPU"),         # straight apostrophe
        ("CPU’s", "CPU"),         # curly apostrophe
        ("NDA’s", "NDA"),         # another curly case
        ("X’s", "X"),             # single-letter stem still OK
    ])
    def test_strips_when_stem_is_all_caps(self, surface, expected):
        assert strip_terminal_plural(surface) == expected

    @pytest.mark.parametrize("surface", [
        "Apis",                   # stem not all-caps → unchanged
        "cats",                   # lowercase word → unchanged
        "123’s",                  # no cased letters in stem → unchanged
        "GPU’s,",                 # trailing punctuation prevents match → unchanged
        "GPUS",                   # endswith('S') (upper S) → unchanged
        "CATS",                   # same: upper 'S' is not matched
    ])
    def test_does_not_strip_in_other_cases(self, surface):
        assert strip_terminal_plural(surface) == surface

    def test_mixed_punctuation_no_strip(self):
        # Apostrophe in middle but no terminal plural suffix
        surface = "O’BRIEN"
        assert strip_terminal_plural(surface) == surface

    def test_trailing_space_prevents_strip(self):
        assert strip_terminal_plural("GPUs ") == "GPUs "

    def test_dotted_initialism_plural_is_not_stripped(self):
        assert strip_terminal_plural("U.S.A.s") == "U.S.A.s"


class TestIsAllCapsWord:
    @pytest.mark.parametrize("surface,allow,expected", [
        ("ALPHA", "-&/._", True),          # simple all-caps, >=4 letters
        ("GPU", "-&/._", False),           # too short (alpha_len=3)
        ("Alpha", "-&/._", False),         # mixed case -> False
        ("ALPHA1", "-&/._", False),        # digits disqualify
        ("A.BCD", "-&/._", False),         # '.' in allow_chars -> disqualify
        ("A.BCD", "", True),               # '.' not in allow_chars -> OK
        ("MOVE-ON", "-&/._", False),       # '-' in allow_chars -> disqualify
        ("MOVE-ON", "", True),             # '-' not listed -> OK (letters are all upper)
        ("R&DCPU", "-&/._", False),        # '&' in allow_chars -> disqualify
        ("RNDCPU", "-&/._", True),         # no separators, all upper, >=4
        ("A_BCD", "-&/._", False),         # '_' in allow_chars -> disqualify
        ("A_BCD", "", True),               # '_' not listed -> OK
        ("ÉTUDE", "-&/._", True),          # Unicode uppercase letters
        ("ÜBER", "-&/._", True),           # Unicode uppercase (German Umlauts)
        ("CHAPÉU", "-&/._", True),         # Accented capitals
        ("ßABC", "-&/._", False),          # 'ß' is not uppercase -> False
    ])
    def test_various(self, surface, allow, expected):
        assert is_all_caps_word(surface, allow) is expected

    def test_alpha_len_boundary(self):
        # Exactly 4 alphabetic chars required
        assert is_all_caps_word("ABCD", "-&/._") is True
        assert is_all_caps_word("ABCd", "-&/._") is False  # mixed case


class TestExclamNearRight:
    @pytest.mark.parametrize("sample,expected", [
        ("[IT]!", True),                # immediate
        ("[IT]   !", True),             # spaces then !
        ("[IT]!!", True),               # multiple !
        ("[IT]！", True),               # full-width
        ("[IT]‼ boom", True),           # double exclam glyph
        ("[IT]. !", False),             # '.' before '!' blocks
        ("[IT]? later!", False),        # '?' before '!' blocks
        ("[IT]   , then !", True),      # comma doesn't block
        ("[IT]", False),                # no exclam at all
    ])
    def test_basics(self, sample, expected):
        text, end = _extract_from_brackets(sample)
        assert exclam_near_right(text, end) is expected

    def test_newline_stops_by_default(self):
        text, end = _extract_from_brackets("[IT]\n!")
        assert exclam_near_right(text, end, stop_at_newline=True) is False

    def test_newline_opt_out_allows_scan(self):
        text, end = _extract_from_brackets("[IT]\n!")
        assert exclam_near_right(text, end, stop_at_newline=False) is True

    @pytest.mark.parametrize("max_scan,sample,expected", [
        # distance = number of characters from end to first '!'
        # "[IT] ab!" -> distance 3 (' ', 'a', 'b')
        (5, "[IT] ab!", True),     # equal to max_scan passes (<=)
        (2, "[IT] ab!", False),    # over budget
        (2, "[IT]!", True),        # distance 0
        (3, "[IT] !", True),       # distance 1 (space)
        (1, "[IT]  !", False),     # distance 2
    ])
    def test_max_scan(self, max_scan, sample, expected):
        text, end = _extract_from_brackets(sample)
        assert exclam_near_right(text, end, max_scan=max_scan) is expected

    def test_stops_before_period_even_if_within_budget(self):
        # There's a '.' before '!' within max_scan — must return False.
        text, end = _extract_from_brackets("[IT] . !")
        assert exclam_near_right(text, end, max_scan=10) is False


class TestCommaNearLeft:
    @pytest.mark.parametrize("sample,expected", [
        ("Well, [ALRIGHTY] THEN!", True),              # direct comma
        ("Well,    [ALRIGHTY] THEN!", True),           # many spaces after comma
        ("Well ,  [ALRIGHTY] THEN!", True),            # spaces around comma still ok
        ("Well,\n   [ALRIGHTY] THEN!", True),          # newline is whitespace → ok
        ("Well [ALRIGHTY] THEN!", False),              # no comma at all
        ("[ALRIGHTY] THEN!", False),                   # start of text
        ("Well,) [ALRIGHTY] THEN!", False),            # closer immediately left → not skipped
        ('Well, " [ALRIGHTY] THEN!', False),           # quote immediately left → not skipped
    ])
    def test_various(self, sample, expected):
        text, start = _extract_from_brackets(sample)
        assert _comma_near_left(text, start) is expected

# TODO write _has_upper_after_with_fillers tests

class TestIsInCapsInterjectionContext:

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
            ("Well, [ALRIGHTY] THEN!", True),  # canonical: comma + ALLCAPS + ALLCAPS + !
            ("Well,    [ALRIGHTY]   THEN!", True),  # extra spaces
            ("OK, [MOVE] ALONG!", True),  # different words
            ("Well, [ALRIGHTY] THEN!!", True),  # multiple !
            ("Well [ALRIGHTY] THEN!", False),  # no comma near left
            ("Well, [ALRIGHTY] THEN.", False),  # no exclamation near right
            ("Well, [Alrighty] THEN!", False),  # first word not ALLCAPS
            ("Well, [ALRIGHTY] Then!", False),  # next word not ALLCAPS
            ("Well, [ALRIGHTY] OK!", False),  # next word too short (<3)
            ("Well, [ALRIGHTY] X!", False),  # length 1 next word
            ("Well, [ALRIGHTY] 123!", False),  # next token not letters
        ],
    )
    def test_various(self, sample, expected, _cfg):
        text, surface, s, e = self._extract_span(sample)
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is expected

    def test_handles_unicode_caps_next_word(self, _cfg):
        # Next word with Unicode uppercase letters should count
        sample = "Well, [BRAVO] ÉTUDE!"
        text, surface, s, e = self._extract_span(sample)
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is True

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
            assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is True

    def test_rejects_if_next_word_not_all_caps_or_too_short(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=4, right_gap=40)
        # Next word not ALL CAPS
        t1 = "Well, [ALRIGHTY] Then!"
        # Next word length < 3
        t2 = "Well, [ALRIGHTY] OK!"
        for sample in [t1, t2]:
            text, surface, s, e = self._extract_span(sample)
            assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is False

    @pytest.mark.parametrize("right_gap,sample,expected", [
        (4, "Well, [MOVE] AYE!", True),  # 1 space + 3 letters = 4
        (3, "Well, [MOVE] NOW!", False),  # dist = 1 + 3 = 4 > 3 → False
        (4, "Well, [MOVE] NOW!", True),  # dist = 4 == right_gap → True
    ])
    def test_right_gap_enforced(self, monkeypatch, right_gap, sample, expected, _cfg):
        _patch_near(monkeypatch, left_gap=3, right_gap=right_gap)
        text, surface, s, e = self._extract_span(sample)
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is expected

    @pytest.mark.parametrize("left_gap,sample,expected", [
        (3, "Well,   [MOVE] NOW!", True),  # 3 spaces after comma OK
        (3, "Well,    [MOVE] NOW!", False),  # 4 spaces > 3
        (4, "Well,    [MOVE] NOW!", True),  # 4 allowed
    ])
    def test_left_gap_enforced(self, monkeypatch, left_gap, sample, expected, _cfg):
        _patch_near(monkeypatch, left_gap=left_gap, right_gap=10)
        text, surface, s, e = self._extract_span(sample)
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is expected

    def test_both_gaps_tight_fail(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=2, right_gap=2)
        text, surface, s, e = self._extract_span("Well,   [GO] UP!")
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is False

    def test_disallowed_separator_fails(self, monkeypatch):
        _patch_near(monkeypatch, left_gap=3, right_gap=40)
        # '-' not allowed here -> fails ALL-CAPS word check
        text, surface, s, e = self._extract_span("Well, [MOVE-ON] THEN!")
        _cfg = make_cfg(allow_chars="&/._-")
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is False

    def test_unicode_all_caps_next_word_ok(self, monkeypatch, _cfg):
        _patch_near(monkeypatch, left_gap=3, right_gap=40)
        text, surface, s, e = self._extract_span("Well, [BRAVO] ÉTUDE!")
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is True

    @pytest.mark.parametrize("sample, expected", [
        ("Well, [HELLO] I AM COOL!", True),
        ("Well, [HELLO] I AM!", False),
        ("Well, [ALRIGHTY] THEN!", True),
        ("Well, [HELLO] YOU ARE cool!", True),  # mixed case breaks
    ])
    def test_multi_word_shout(self, sample, expected, _cfg, monkeypatch):
        _patch_near(monkeypatch, left_gap=4, right_gap=40)
        text, surface, s, e = self._extract_span(sample)
        assert is_in_caps_interjection_context(surface, text, s, e, _cfg) is expected


class TestIsInCapsInterjectionContextPrev:

    @staticmethod
    def _extract_span(s: str) -> tuple[str, int, int, str]:
        """
        Use [ ... ] to mark the second (surface) word span.
        Returns: (clean_text, s, e, surface)
        """
        pre, rest = s.split("[", 1)
        inside, post = rest.split("]", 1)
        text = pre + inside + post
        s = len(pre)
        e = s + len(inside)
        return text, s, e, inside

    @pytest.mark.parametrize("sample", [
        "Well, ALRIGHTY [THEN]!",  # canonical
        "Well,   ALRIGHTY   [THEN]!",  # extra spaces
        "Well, ALRIGHTY [THEN]!!",  # multiple exclamations
    ])
    def test_positive_cases(self, sample, _cfg):
        text, s, e, surface = self._extract_span(sample)
        assert is_in_caps_interjection_context_prev(surface, text, s, e, _cfg) is True

    @pytest.mark.parametrize("sample", [
        "Well ALRIGHTY [THEN]!",  # no comma near left of prev word
        "Well, Alrighty [THEN]!",  # prev not ALL-CAPS
        "Well, AHA [NOW]!",  # prev len < 4
        "Well, ALRIGHTY [UP]!",  # surface len < 3 (second word too short)
        "Well, ALRIGHTY [THEN]. Bang!",  # '.' before '!' → exclam_near_right False
        "Well, ) ALRIGHTY [THEN]!",  # closer before prev (comma not nearest non-space)
    ])
    def test_negative_cases(self, sample, _cfg):
        text, s, e, surface = self._extract_span(sample)
        assert is_in_caps_interjection_context_prev(surface, text, s, e, _cfg) is False

    def test_prev_word_detected_when_tight_to_surface(self, _cfg):
        # Ensure the backward scan finds the previous ALL-CAPS word with a single space.
        sample = "Well, ALRIGHTY [THEN]!"
        text, s, e, surface = self._extract_span(sample)
        assert is_in_caps_interjection_context_prev(surface, text, s, e, _cfg) is True

    def test_prev_word_with_space_before_comma_still_counts(self, _cfg):
        # Comma is still the nearest non-space char to the left of prev word.
        sample = "Well ,  ALRIGHTY  [THEN]!"
        text, s, e, surface = self._extract_span(sample)
        assert is_in_caps_interjection_context_prev(surface, text, s, e, _cfg) is True


class TestIsAllCapsHeading:
    @pytest.mark.parametrize("sample, expected", [
        ("\n[INTRODUCTION]\nBody", True),  # simple all-caps, >=6 letters
        ("\n   [   HEADING   ]   \n", True),  # leading/trailing spaces
        ("\n[API V2 OVERVIEW]\n", True),  # digits ignored, letters all caps
        ("\n[End. The]\n", False),  # mixed case -> False
        ("\n[FAQ]\n", False),  # <6 letters -> False
        ("\n[----]\n", False),  # no letters -> False
        ("\n[CHAPÉU]\n", True),  # Unicode uppercase letters
        ("\n[ΠΡΟΛΟΓΟΣ]\n", True),  # Greek uppercase
        ("Prev\n.. [NOT all CAPS]\nNext\n", False),  # mixed case long enough
        ("# Intro\n[INTRODUCTION line continues]\n", False),  # selection mid-line still but contains lowercase
        ("Last line with \n[OVERVIEW]\n(no trailing newlne)", True),  # \n \n
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
            ("^Hello", True),  # start of doc
            ("Hello. ^World", True),  # period + space
            ("Hello.^World", True),  # period, no space
            ('He said: "Go." ^Then', True),  # period + closing quote
            ("Do it now.) ^Then", True),  # period + ) closer
            ('"Go?!" ^Next', True),  # mixed ?!
            ("Wait… ^Go", True),  # unicode ellipsis
            ("Hello ^brave world", False),  # mid-sentence
            ("Wait—^no", False),  # em dash is not a terminator
            ("Hello\n^World", False),  # newline alone isn't a boundary
            ('He said "hello" ^and left.', False),  # closer without terminator
            ('He said: “Go.”^Then', True),  # curly closer right before next
            ('Done! ”^Next', True),  # space + curly quote closer
            ('Done! »^Next', True),  # guillemet closer
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
