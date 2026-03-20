from plainera_unacronym.nlp.extraction.acronyms.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.anchored.patterns import compile_anchored_for_surface
from plainera_unacronym.nlp.extraction.acronyms.anchored.spans import resolve_def_span


def _by_kind(specs, kind: str):
    return [s for s in specs if s.kind == kind]


def _first_match(specs, text: str):
    for s in specs:
        m = s.pat.search(text)
        if m:
            return s, m
    return None, None


class TestCompileAnchoredExactContracts:
    def test_returns_non_empty_and_has_expected_kinds(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("PDF", cfg)

        kinds = {s.kind for s in specs}
        # Parenthetical shapes
        assert "def_before" in kinds
        assert "def_after" in kinds
        assert "before_acr_paren" in kinds
        # Inline shapes
        assert "inline" in kinds
        assert "inline_before" in kinds

        # Sanity: every spec must have compiled regex and base_conf
        assert all(hasattr(s.pat, "search") for s in specs)
        assert all(0 < s.base_conf <= 1.0 for s in specs)

    def test_forward_paren_captures_groups(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("PTO", cfg)
        text = "Please turn over (PTO)."

        spec, m = _first_match(specs, text)
        assert m is not None

        assert m.group("acr") == "PTO"
        assert m.group("def") == "Please turn over"

    def test_reverse_paren_captures_groups(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("SSO", cfg)
        text = "SSO (Single sign-on) is enabled."

        spec, m = _first_match(specs, text)
        assert m is not None

        assert m.group("acr") == "SSO"
        assert m.group("def") == "Single sign-on"

    def test_brackets_variants_work(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("GPU", cfg)

        # Forward bracket
        text1 = "Graphics Processing Unit [GPU]"
        spec1, m1 = _first_match(specs, text1)
        assert m1 is not None
        assert m1.group("acr") == "GPU"
        assert m1.group("def") == "Graphics Processing Unit"

        # Reverse bracket
        text2 = "GPU [Graphics Processing Unit]"
        spec2, m2 = _first_match(specs, text2)
        assert m2 is not None
        assert m2.group("acr") == "GPU"
        assert m2.group("def") == "Graphics Processing Unit"

    def test_optional_trailing_dot_is_outside_acr_group(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("U.S", cfg)

        text = "United States (U.S.)."
        spec, m = _first_match(specs, text)
        assert m is not None

        # Acr group should be EXACT "U.S" (dot-optional is outside group)
        assert m.group("acr") == "U.S"
        assert m.group("def") == "United States"

    def test_optional_quotes_do_not_enter_acr_group(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("NHS", cfg)

        text = 'National Health Service ("NHS")'
        spec, m = _first_match(specs, text)
        assert m is not None
        assert m.group("acr") == "NHS"
        assert m.group("def") == "National Health Service"

    def test_tail_inside_wrapper_does_not_pollute_acr_group(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("PPE", cfg)

        text = "Personal Protective Equipment (PPE, including masks)."
        spec, m = _first_match(specs, text)
        assert m is not None

        assert m.group("acr") == "PPE"
        assert m.group("def") == "Personal Protective Equipment"

    def test_possessive_surface_allowed_but_acr_group_remains_plain(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("PDF", cfg)

        text1 = "PDF's (Portable Document Format) is common."
        spec1, m1 = _first_match(specs, text1)
        assert m1 is not None
        assert m1.group("acr") == "PDF"
        assert m1.group("def") == "Portable Document Format"

        text2 = "PDF’s (Portable Document Format) is common."
        spec2, m2 = _first_match(specs, text2)
        assert m2 is not None
        assert m2.group("acr") == "PDF"
        assert m2.group("def") == "Portable Document Format"

    def test_inline_after_resolves_definition_span(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("NLP", cfg)

        text = "NLP stands for Natural language processing."
        seg = text

        # Pick an inline-after spec that matches
        hit = None
        for spec in specs:
            if spec.kind != "inline":
                continue
            m = spec.pat.search(seg)
            if m:
                hit = (spec, m)
                break

        assert hit is not None
        spec, m = hit

        a0, a1 = m.span("acr")
        span = resolve_def_span(spec.strategy, seg=seg, m=m, acr_key="NLP", a1_local=a1, cfg=cfg)
        assert span is not None

        d0, d1 = span
        got = seg[d0:d1]
        assert "Natural language processing" in got

    def test_inline_before_resolves_definition_span(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("NLP", cfg)

        text = "Natural language processing stands for NLP."
        seg = text

        hit = None
        for spec in specs:
            if spec.kind != "inline_before":
                continue
            m = spec.pat.search(seg)
            if m:
                hit = (spec, m)
                break

        assert hit is not None
        spec, m = hit

        a0, a1 = m.span("acr")
        span = resolve_def_span(spec.strategy, seg=seg, m=m, acr_key="NLP", a1_local=a1, cfg=cfg)
        assert span is not None

        d0, d1 = span
        got = seg[d0:d1]
        assert "Natural language processing" in got

    def test_inline_before_matches_one_cue(self):
        cfg = ExtractionConfig()
        specs = compile_anchored_for_surface("NLP", cfg)
        text = "Natural language processing stands for NLP"
        seg = text

        # Find an inline_before match
        spec = next(s for s in specs if s.kind == "inline_before" and s.pat.search(seg))
        m = spec.pat.search(seg)
        assert m is not None
        assert m.group("acr").upper() == "NLP"

        a0, a1 = m.span("acr")
        span = resolve_def_span(spec.strategy, seg=seg, m=m, acr_key="NLP", a1_local=a1, cfg=cfg)
        assert span is not None

        d0, d1 = span
        assert seg[d0:d1] == "Natural language processing"
