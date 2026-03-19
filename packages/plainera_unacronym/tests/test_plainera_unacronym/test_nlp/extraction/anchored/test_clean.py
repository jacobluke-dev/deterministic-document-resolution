import pytest
from plainera_unacronym.nlp.common.types import INLINE
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.anchored.clean import _strip_leading_determiner, clean_definition


class TestStripLeadingDeterminer:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("the Portable Document Format", "Portable Document Format"),
            ("a Portable Document Format", "Portable Document Format"),
            ("an Umbrella Term", "Umbrella Term"),
            ("  the Portable Document Format", "Portable Document Format"),  # leading whitespace ignored
            ("THE Portable Document Format", "Portable Document Format"),  # case-insensitive
            ("An Umbrella Term", "Umbrella Term"),  # case-insensitive
            # Only strips one determiner
            ("the the Portable Document Format", "the Portable Document Format"),
            ("a an Umbrella Term", "an Umbrella Term"),
            # Not at start => unchanged
            ("Portable Document Format", "Portable Document Format"),
            ("in the Portable Document Format", "in the Portable Document Format"),
            # Must be whole token "a"/"an"/"the" (word boundary)
            ("theatre mode", "theatre mode"),
            ("another thing", "another thing"),
            ("anvil", "anvil"),
            # Punctuation immediately after determiner: not matched by this regex (needs whitespace)
            ("the, Portable Document Format", "the, Portable Document Format"),
        ],
    )
    def test_cases(self, inp, expected):
        assert _strip_leading_determiner(inp) == expected

    def test_empty_string(self):
        assert _strip_leading_determiner("") == ""
        assert _strip_leading_determiner("   ") == "   "


class _FakeTokenRe:
    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    def findall(self, _s: str):
        return self._tokens


class TestCleanDefinitionUnit:
    def test_inline_tightens_span(self, _patch):
        cfg = ExtractionConfig()
        calls = {"tighten_span": 0, "strip_det": 0, "tighten_label": [], "norm": []}

        def fake_tighten_span(s):
            calls["tighten_span"] += 1
            return f"TSPAN[{s}]"

        def fake_strip_det(s):
            calls["strip_det"] += 1
            return f"STRIP[{s}]"

        def fake_tighten_label(base, acr):
            calls["tighten_label"].append((base, acr))
            return f"TL[{base}|{acr}]"

        def fake_norm(s):
            calls["norm"].append(s)
            return "CLEAN_OK"

        _patch(
            clean_definition,
            tighten_definition_span=fake_tighten_span,
            _strip_leading_determiner=fake_strip_det,
            tighten_label_by_acronym=fake_tighten_label,
            normalize_definition=fake_norm,
            has_letter=lambda s: True,
        )

        cfg = ExtractionConfig(require_two_words=False)
        out = clean_definition("Raw Def", acr_norm="SSO", cfg=cfg, kind=INLINE)
        assert out == "CLEAN_OK"
        assert calls["tighten_span"] == 1
        assert calls["strip_det"] == 0
        assert calls["tighten_label"] == [("TSPAN[Raw Def]", "SSO")]

    def test_inline_before_strips_determiner(self, _patch):
        cfg = ExtractionConfig()
        calls = {"tighten_span": 0, "strip_det": 0}

        _patch(
            clean_definition,
            tighten_definition_span=lambda s: (_ for _ in ()).throw(AssertionError("should not be called")),
            _strip_leading_determiner=lambda s: (calls.__setitem__("strip_det", calls["strip_det"] + 1) or f"S[{s}]"),
            tighten_label_by_acronym=lambda base, acr: base,  # passthrough
            normalize_definition=lambda s: s,  # passthrough
            has_letter=lambda s: True,
        )

        out = clean_definition("the Portable Document Format", acr_norm="PDF", cfg=cfg, kind="inline_before")
        assert out == "S[the Portable Document Format]"
        assert calls["strip_det"] == 1

    def test_non_inline_kind_uses_orig_as_base(self, _patch):
        cfg = ExtractionConfig()

        seen = {}

        def fake_tighten_label(base, acr):
            seen["base"] = base
            seen["acr"] = acr
            return base

        _patch(
            clean_definition,
            tighten_definition_span=lambda s: (_ for _ in ()).throw(AssertionError("nope")),
            _strip_leading_determiner=lambda s: (_ for _ in ()).throw(AssertionError("nope")),
            tighten_label_by_acronym=fake_tighten_label,
            normalize_definition=lambda s: s,
            has_letter=lambda s: True,
        )

        out = clean_definition("ORIG", acr_norm="X", cfg=cfg, kind="parenthetical")
        assert out == "ORIG"
        assert seen == {"base": "ORIG", "acr": "X"}

    def test_inline_raw_length_gate_rejects_over_max_phrase_chars(self, _patch):
        cfg = ExtractionConfig(max_phrase_chars=10)

        _patch(
            clean_definition,
            # if we pass the early gate, these might be called; ensure they aren't
            tighten_definition_span=lambda s: (_ for _ in ()).throw(AssertionError("should not be called")),
            tighten_label_by_acronym=lambda *_: (_ for _ in ()).throw(AssertionError("should not be called")),
            normalize_definition=lambda *_: (_ for _ in ()).throw(AssertionError("should not be called")),
            has_letter=lambda *_: True,
        )

        orig = "a" * 50
        out = clean_definition(orig, acr_norm="PDF", cfg=cfg, kind=INLINE)
        assert out is None

    def test_rejects_when_normalised_empty_or_no_letters_or_too_long(self, _patch):
        cfg = ExtractionConfig(max_phrase_chars=10)

        # Case 1: empty
        _patch(
            clean_definition,
            tighten_definition_span=lambda s: s,
            tighten_label_by_acronym=lambda base, acr: base,
            normalize_definition=lambda s: "",
            has_letter=lambda s: True,
        )
        assert clean_definition("X", acr_norm="A", cfg=cfg, kind=INLINE) is None

        # Case 2: no letters
        _patch(
            clean_definition,
            normalize_definition=lambda s: "123-456",
            has_letter=lambda s: False,
        )
        assert clean_definition("X", acr_norm="A", cfg=cfg, kind="inline") is None

        # Case 3: too long after normalisation
        _patch(
            clean_definition,
            normalize_definition=lambda s: "X" * 100,
            has_letter=lambda s: True,
        )
        assert clean_definition("X", acr_norm="A", cfg=cfg, kind=INLINE) is None

    def test_two_word_gate_only_applies_to_inline_kinds(self, _patch):
        cfg = ExtractionConfig(require_two_words=True)

        _patch(
            clean_definition,
            tighten_definition_span=lambda s: s,
            tighten_label_by_acronym=lambda base, acr: base,
            normalize_definition=lambda s: s,
            has_letter=lambda s: True,
        )

        # Patch TOKEN_RE in the function's globals to simulate 1 token vs 2 tokens
        _patch(clean_definition, TOKEN_RE=_FakeTokenRe(tokens=["one"]))
        assert clean_definition("Authentication", acr_norm="A", cfg=cfg, kind=INLINE) is None

        _patch(clean_definition, TOKEN_RE=_FakeTokenRe(tokens=["one", "two"]))
        assert clean_definition("Single sign-on", acr_norm="SSO", cfg=cfg, kind="inline") == "Single sign-on"

        # Non-inline kinds should not be gated by token count
        _patch(clean_definition, TOKEN_RE=_FakeTokenRe(tokens=["one"]))
        assert clean_definition("Authentication", acr_norm="A", cfg=cfg, kind="parenthetical") == "Authentication"


class TestCleanDefinitionIntegration:
    def test_inline_before_strips_article_and_normalises(self):
        cfg = ExtractionConfig(require_two_words=True)
        out = clean_definition("the Portable Document Format", acr_norm="PDF", cfg=cfg, kind="inline_before")
        assert out == "Portable Document Format"

    def test_inline_clean_definition_trims_trailing_punct(self):
        cfg = ExtractionConfig(require_two_words=True)
        out = clean_definition("Single sign-on...", acr_norm="SSO", cfg=cfg, kind="inline")
        assert out == "Single sign-on"
