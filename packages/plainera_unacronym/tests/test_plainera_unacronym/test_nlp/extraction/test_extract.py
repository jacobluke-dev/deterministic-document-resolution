import re
from types import SimpleNamespace as NS
import pytest

from plainera_unacronym.nlp.extraction.extract import _acr_pat, _def_pat, _compile_parenthetical, _compile_inline


def _cfg(**overrides):
    base = dict(
        min_acr_len=2,
        max_acr_len=10,
        acr_allowed=r"A-Z0-9&./-",
        max_phrase_chars=200,
    )
    base.update(overrides)
    return NS(**base)


@pytest.fixture
def cfg_default():
    return _cfg()


class TestAcrPat:
    def test_accepts_common_acronyms_and_specials(self, cfg_default):
        pat = re.compile(rf"^{_acr_pat(cfg_default)}$")
        assert pat.fullmatch("PDF")
        assert pat.fullmatch("R&D")
        assert pat.fullmatch("C/A")
        assert pat.fullmatch("A-10")
        assert pat.fullmatch("3D")

    def test_rejects_length_outside_bounds(self):
        cfg = _cfg(min_acr_len=2, max_acr_len=5)
        pat = re.compile(rf"^{_acr_pat(cfg)}$")
        assert pat.fullmatch("P") is None         # too short
        assert pat.fullmatch("TOOLONG") is None   # too long
        assert pat.fullmatch("SME")               # within bounds

    def test_rejects_chars_not_in_allowed_set(self):
        cfg = _cfg(acr_allowed=r"A-Z")  # only letters A-Z
        pat = re.compile(rf"^{_acr_pat(cfg)}$")
        assert pat.fullmatch("PDF")
        assert pat.fullmatch("R&D") is None   # '&' not allowed now
        assert pat.fullmatch("C/A") is None   # '/' not allowed
        assert pat.fullmatch("A_1") is None   # '_' and digits disallowed here


class TestDefPat:
    def test_accepts_plain_text_within_limit(self, cfg_default):
        pat = re.compile(rf"^{_def_pat(cfg_default)}$")
        m = pat.fullmatch("Portable Document Format")
        assert m
        assert m.group("def") == "Portable Document Format"

    def test_rejects_forbidden_paren_and_braces(self):
        cfg = _cfg(max_phrase_chars=100)
        pat = re.compile(rf"^{_def_pat(cfg)}$")
        assert pat.fullmatch("Alpha Beta")            # ok
        assert pat.fullmatch("Has)Paren") is None     # ')' forbidden
        assert pat.fullmatch("Has{Brace") is None     # '{' forbidden
        assert pat.fullmatch("Has}Brace") is None     # '}' forbidden

    def test_enforces_max_phrase_chars(self):
        cfg = _cfg(max_phrase_chars=5)
        pat = re.compile(rf"^{_def_pat(cfg)}$")
        assert pat.fullmatch("ABCDE")                 # exactly 5
        assert pat.fullmatch("ABCDEF") is None        # 6 → too long

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("GPU", True),
            ("Cost per Acquisition", True),
            ("Trailing space ok ", True),
        ],
    )
    def test_named_group_def_is_present(self, cfg_default, text, expected):
        # just sanity-check presence of the named group "def"
        pat = re.compile(rf"{_def_pat(cfg_default)}")
        m = pat.search(text)
        assert (m is not None) == expected
        if m:
            assert "def" in m.groupdict()


class TestCompileParenthetical:
    def test_returns_two_patterns_with_flags(self):
        cfg = _cfg()
        fwd, rev = _compile_parenthetical(cfg)
        assert isinstance(fwd, re.Pattern) and isinstance(rev, re.Pattern)
        assert (fwd.flags & re.IGNORECASE) and (fwd.flags & re.MULTILINE)
        assert (rev.flags & re.IGNORECASE) and (rev.flags & re.MULTILINE)

    def test_forward_and_reverse_match(self):
        cfg = _cfg()
        fwd, rev = _compile_parenthetical(cfg)
        text = (
            "Portable Document Format (PDF) is widely used.\n"
            "Also PDF (Portable Document Format) appears later."
        )
        # Forward: should match the first sentence
        m1 = fwd.search(text)
        assert m1
        assert m1.group("acr").lower() == "pdf"
        assert "Portable" in m1.group("def")

        # Reverse: there should EXIST a match where acr == PDF and def contains the long form
        rev_hits = list(rev.finditer(text))
        assert any(m.group("acr").lower() == "pdf" and "Portable" in m.group("def") for m in rev_hits)

    def test_max_phrase_chars_limits_definition(self):
        cfg = _cfg(max_phrase_chars=5)
        fwd, rev = _compile_parenthetical(cfg)
        # Forward: long phrase present, but regex can still match the SHORT tail right before parens
        text_fwd = "Incredibly long name (PDF)"
        m = fwd.search(text_fwd)
        assert m is not None
        assert len(m.group("def")) <= 5  # the definition capture is bounded

        # Reverse: DEF inside parens longer than 5 → no match
        text_rev = "PDF (Incredibly long)"
        assert rev.search(text_rev) is None

    def test_forbids_paren_and_braces_inside_def(self):
        cfg = _cfg()
        fwd, rev = _compile_parenthetical(cfg)

        # Forward: the DEF itself contains a forbidden char → must fail
        assert fwd.search("Bad) (PDF)") is None  # ')' forbidden in DEF
        assert fwd.search("Bad{ (PDF)") is None  # '{' forbidden in DEF
        assert fwd.search("Bad} (PDF)") is None  # '}' forbidden in DEF

        # Reverse: test braces *inside the parentheses* → must fail
        assert rev.search("PDF (Bad{)") is None
        assert rev.search("PDF (Bad})") is None

        # And clarify behavior: extra text after a valid '(DEF)' still yields a match
        # (engine matches the earliest valid 'PDF (DEF)' and ignores the trailing ' token)')
        assert rev.search("PDF (Bad) token)") is not None

    def test_special_char_acronyms_are_escaped(self):
        cfg = _cfg()
        fwd, rev = _compile_parenthetical(cfg)
        text = (
            "Research and Development (R&D) fuels innovation. "
            "We track cost of acquisition: C/A (Cost per Acquisition)."
        )
        m_rnd = fwd.search(text)
        assert m_rnd and m_rnd.group("acr") == "R&D" and "Research and Development" in m_rnd.group("def")
        rev_hits = list(rev.finditer(text))
        assert any(m.group("acr") == "C/A" and "Cost per Acquisition" in m.group("def") for m in rev_hits)

class TestCompileInline:
    def _cfg(self, **overrides):
        return _cfg(**overrides)

    def test_compiles_one_pattern_per_cue(self):
        cfg = _cfg()
        cues = (r"short\s+for", r"stands?\s+for")
        pats = _compile_inline(cfg, cues)
        assert len(pats) == len(cues)
        for p in pats:
            assert isinstance(p, re.Pattern)
            assert (p.flags & re.IGNORECASE) and (p.flags & re.MULTILINE)

    def test_inline_matches_all_cues_and_optional_comma(self):
        cfg = _cfg()
        cues = (r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for")
        pats = _compile_inline(cfg, cues)
        text = (
            "PDF, short for Portable Document Format.\n"
            "PDF stands for Portable Document Format.\n"
            "PDF is an acronym for Portable Document Format."
        )
        hits = 0
        for p in pats:
            for m in p.finditer(text):
                assert m.group("acr").lower() == "pdf"
                assert m.group("def").strip() != ""
                hits += 1
        assert hits >= 3

    def test_inline_respects_max_phrase_chars(self):
        cfg = _cfg(max_phrase_chars=10)
        pats = _compile_inline(cfg, (r"stands?\s+for",))
        assert all(p.search("PDF stands for a very very long definition here") is None for p in pats)
        assert any(p.search("PDF stands for format") for p in pats)

    def test_inline_escapes_special_char_acronyms(self):
        cfg = _cfg()
        pats = _compile_inline(cfg, (r"stands?\s+for",))
        text = "C/A stands for Cost per Acquisition"
        m = None
        for p in pats:
            m = p.search(text) or m
        assert m and m.group("acr") == "C/A" and "Cost per Acquisition" in m.group("def")
