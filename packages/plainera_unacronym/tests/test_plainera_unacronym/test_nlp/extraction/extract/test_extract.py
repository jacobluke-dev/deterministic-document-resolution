import re
import sys
from types import SimpleNamespace as NS, ModuleType
import pytest

from plainera_unacronym.nlp.extraction.extract import _acr_pat, _def_pat, _compile_parenthetical, _compile_inline, \
    _has_letters, _two_words, _initials_match, _build_plan, _parenthetical_allowed


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


class TestHasLetters:
    def test_empty_and_nonalpha_only(self):
        assert _has_letters("") is False
        assert _has_letters("12345") is False
        assert _has_letters("!!!") is False
        assert _has_letters("  \t\n ") is False

    def test_ascii_letters_present(self):
        assert _has_letters("a") is True
        assert _has_letters("Z") is True
        assert _has_letters("123abc") is True
        assert _has_letters("abc-123") is True

    def test_unicode_letters_not_counted_by_pattern(self):
        # Regex is [A-Za-z], so characters like 'é'/'Ä' won't match
        assert _has_letters("é") is False
        assert _has_letters("Ä") is False


class TestTwoWords:
    def test_false_for_zero_or_one_word_with_letters(self):
        assert _two_words("") is False
        assert _two_words("a") is False
        assert _two_words("123") is False
        assert _two_words("123 !@#") is False

    def test_true_for_two_or_more_words_with_letters(self):
        assert _two_words("foo bar") is True
        assert _two_words("GPU 3D") is True  # '3D' contains a letter, so counts
        assert _two_words("alpha beta gamma") is True

    def test_ignores_purely_numeric_or_symbol_words(self):
        # Only words containing [A-Za-z] count
        assert _two_words("alpha 123") is False
        assert _two_words("123 beta") is False
        assert _two_words("alpha 456 beta") is True

    def test_whitespace_variations(self):
        assert _two_words(" foo   bar ") is True
        assert _two_words("\tfoo\nbar") is True


class TestInitialsMatch:
    def test_positive_simple(self):
        assert _initials_match("PDF", "Portable Document Format") is True
        assert _initials_match("PTO", "Please Turn Over") is True
        assert _initials_match("HTTP", "Hyper Text Transfer Protocol") is True

    def test_positive_with_bridges_and_case(self):
        # ‘per’ is a bridge between C and A; order must be preserved
        assert _initials_match("C/A", "Cost per Acquisition") is True
        # case-insensitive for both sides
        assert _initials_match("pdf", "portable document format") is True

    def test_positive_ignores_nonalpha_in_acronym(self):
        # Non-alpha chars in acronym should be ignored by the matcher
        assert _initials_match("R&D", "Research and Development") is True
        assert _initials_match("A/B/C", "Alpha Beta Charlie") is True

    def test_skips_tokens_with_nonalpha_initials(self):
        # Token '3M' starts with digit and is skipped when building initials
        # Initials from phrase become ['H','P'] — 'HP' — which should match
        assert _initials_match("HP", "3M Hewlett Packard") is True
        # But '3M Hewlett' provides initials ['H'] → 'HP' should NOT match
        assert _initials_match("HP", "3M Hewlett") is False

    def test_negative_when_order_not_preserved(self):
        assert _initials_match("ABC", "Alpha Charlie Beta") is False
        assert _initials_match("CPU", "Central Unit Processing") is False

    def test_negative_when_missing_letters(self):
        assert _initials_match("ABC", "Alpha Beta") is False
        assert _initials_match("PDF", "Portable Format") is False



class TestBuildPlan:
    def test_no_plugins_uses_cfg_inline_cues_only(self):
        cfg = _cfg(inline_cues=(r"short\s+for", r"stands?\s+for"), plugins=())
        plan = _build_plan(cfg)

        assert tuple(plan.inline_cues) == cfg.inline_cues
        assert isinstance(plan.parenthetical_allows, tuple)
        assert len(plan.parenthetical_allows) == 0

    def test_plugins_extend_extraction(self, monkeypatch):
        """
        Simulate plainera_unacronym.nlp.plugins.registry.get returning two plugins,
        each calling builder hooks to extend cues and parenthetical allows.
        """
        # Create a fake registry module at the dotted path that the relative import resolves to:
        # extract_first_occ.py does: from ..plugins.registry import get
        # That resolves to: plainera_unacronym.nlp.plugins.registry
        registry_mod_name = "plainera_unacronym.nlp.plugins.registry"
        fake_registry = ModuleType(registry_mod_name)

        class PluginA:
            def extend_extraction(self, builder):
                builder.add_inline_cues([r"also\s+known\s+as"])
                builder.add_parenthetical_allow(lambda acr, df: acr.isupper())

        class PluginB:
            def extend_extraction(self, builder):
                builder.add_inline_cues([r"aka"])
                builder.add_parenthetical_allow(lambda acr, df: " " in df)

        def fake_get(_names):
            # names could be ("pA","pB"), but we ignore and return instances
            return [PluginA(), PluginB()]

        fake_registry.get = fake_get

        # Install the fake module into sys.modules so the relative import succeeds
        monkeypatch.setitem(sys.modules, registry_mod_name, fake_registry)

        base_cues = (r"short\s+for", r"stands?\s+for")
        cfg = _cfg(inline_cues=base_cues, plugins=("pA", "pB"))

        plan = _build_plan(cfg)

        # Inline cues should be base + extras (order preserved: base first, then plugin extras)
        assert tuple(plan.inline_cues) == base_cues + (r"also\s+known\s+as", r"aka")

        # Parenthetical allows should include both plugin-provided callables
        assert isinstance(plan.parenthetical_allows, tuple)
        assert len(plan.parenthetical_allows) == 2

        # Sanity: the allows behave as advertised
        allow1, allow2 = plan.parenthetical_allows
        assert allow1("PDF", "Portable Document Format") is True   # A: requires acr.isupper()
        assert allow1("Pdf", "Portable Document Format") is False
        assert allow2("PDF", "Cost per Acquisition") is True       # B: requires space in def
        assert allow2("PDF", "Acquisition") is False

    def test_registry_exception_is_swallowed(self, monkeypatch):
        """
        If registry.get raises (or import fails), _build_plan must not crash and
        should fall back to config-only plan.
        """
        registry_mod_name = "plainera_unacronym.nlp.plugins.registry"
        fake_registry = ModuleType(registry_mod_name)

        def raising_get(_names):
            raise RuntimeError("boom")

        fake_registry.get = raising_get
        monkeypatch.setitem(sys.modules, registry_mod_name, fake_registry)

        base_cues = (r"short\s+for", r"stands?\s+for")
        cfg = _cfg(inline_cues=base_cues, plugins=("anything",))

        plan = _build_plan(cfg)

        # Should gracefully fall back: no extras added
        assert tuple(plan.inline_cues) == base_cues
        assert isinstance(plan.parenthetical_allows, tuple)
        assert len(plan.parenthetical_allows) == 0


class TestParentheticalAllowed:
    def test_returns_true_when_no_attribute(self):
        cfg = NS()  # no _parenthetical_allows attr
        assert _parenthetical_allowed(cfg, "Portable Document Format", "PDF") is True

    def test_returns_true_when_empty_list(self):
        cfg = NS(_parenthetical_allows=[])
        assert _parenthetical_allowed(cfg, "Portable Document Format", "PDF") is True

    def test_single_predicate_true(self):
        # allow only if acronym is uppercase
        cfg = NS(_parenthetical_allows=[lambda definition, acronym: acronym.isupper()])
        assert _parenthetical_allowed(cfg, "Portable Document Format", "PDF") is True

    def test_single_predicate_false(self):
        cfg = NS(_parenthetical_allows=[lambda definition, acronym: acronym.isupper()])
        assert _parenthetical_allowed(cfg, "Portable Document Format", "Pdf") is False

    def test_multiple_predicates_all_must_pass(self):
        f1 = lambda definition, acronym: acronym.isupper()
        f2 = lambda definition, acronym: " " in definition  # requires multi-word def
        cfg = NS(_parenthetical_allows=[f1, f2])

        assert _parenthetical_allowed(cfg, "Portable Document Format", "PDF") is True
        assert _parenthetical_allowed(cfg, "Acquisition", "PDF") is False  # fails f2
        assert _parenthetical_allowed(cfg, "Portable Document Format", "Pdf") is False  # fails f1

    def test_argument_order_is_definition_then_acronym(self):
        # capture the received args to ensure correct ordering
        received = {}
        def spy(definition, acronym):
            received["definition"] = definition
            received["acronym"] = acronym
            return True

        cfg = NS(_parenthetical_allows=[spy])
        ok = _parenthetical_allowed(cfg, "Cost per Acquisition", "C/A")
        assert ok is True
        assert received["definition"] == "Cost per Acquisition"
        assert received["acronym"] == "C/A"

    def test_exception_in_predicate_bubbles_up(self):
        def boom(definition, acronym):
            raise RuntimeError("predicate failed")

        cfg = NS(_parenthetical_allows=[boom])
        with pytest.raises(RuntimeError):
            _parenthetical_allowed(cfg, "Any", "ACR")
