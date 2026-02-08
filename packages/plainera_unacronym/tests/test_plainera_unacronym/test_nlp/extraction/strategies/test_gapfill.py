from types import SimpleNamespace as NS

import pytest
from plainera_unacronym.nlp.extraction.strategies.gapfill import fill_missing_from_defs

# ----------------------------
# helpers
# ----------------------------

def _fo(start: int) -> object:
    # fill_missing_from_defs only reads .start_offset
    return NS(start_offset=start)


def _ed(
    *,
    acronym: str,
    definition: str = "DEF",
    acr_start: int,
    acr_end: int,
    def_start: int = 0,
    def_end: int = 1,
    confidence: float = 0.5,
    original_definition: str = "orig",
) -> object:
    # fill_missing_from_defs reads these attrs
    return NS(
        acronym=acronym,
        definition=definition,
        acr_start=acr_start,
        acr_end=acr_end,
        def_start=def_start,
        def_end=def_end,
        confidence=confidence,
        original_definition=original_definition,
    )


# =====================================================================
# UNIT TESTS (patch normalisation to isolate selection logic)
# =====================================================================

class TestFillMissingFromDefsUnit:
    def test_returns_none_when_no_defs_for_key(self, _patch):
        # Force defs to normalise under a different key than firsts ("PDF")
        _patch(
            fill_missing_from_defs,
            normalize_acronym_key=lambda surface, allow_chars, dotted_mode: f"{surface}__X",
        )

        det_cfg = NS(allow_chars="", dotted_display="strip")
        firsts = {"PDF": _fo(10)}
        defs = [_ed(acronym="PDF", acr_start=12, acr_end=15)]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        assert out == {"PDF": None}

    def test_selects_closest_then_higher_confidence_then_earliest(self,  _patch):
        _patch(
            fill_missing_from_defs,
            normalize_acronym_key=lambda surface, allow_chars, dotted_mode: surface,  # identity
        )

        det_cfg = NS(allow_chars="", dotted_display="strip")
        firsts = {"PDF": _fo(100)}

        # Distance to FO is abs(acr_start - 100)
        # A and B same distance (5), then choose higher confidence (B)
        # C same distance and confidence as B, but earlier acr_start → should win over B
        defs = [
            _ed(acronym="PDF", definition="A", acr_start=95, acr_end=98, confidence=0.80),
            _ed(acronym="PDF", definition="B", acr_start=105, acr_end=108, confidence=0.90),
            _ed(acronym="PDF", definition="C", acr_start=95, acr_end=98, confidence=0.90),
        ]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        pick = out["PDF"]
        assert pick is not None
        assert pick.definition == "C"
        assert pick.acr_span == (95, 98)
        assert pick.confidence == pytest.approx(0.90)

    def test_handles_multiple_firsts_independently(self,  _patch):
        _patch(
            fill_missing_from_defs,
            normalize_acronym_key=lambda surface, allow_chars, dotted_mode: surface,  # identity
        )

        det_cfg = NS(allow_chars="", dotted_display="strip")
        firsts = {"PDF": _fo(10), "GPU": _fo(200)}

        defs = [
            _ed(acronym="PDF", definition="Portable Document Format", acr_start=12, acr_end=15, confidence=0.6),
            _ed(acronym="GPU", definition="Graphics Processing Unit", acr_start=190, acr_end=193, confidence=0.7),
        ]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        assert out["PDF"].definition == "Portable Document Format"
        assert out["GPU"].definition == "Graphics Processing Unit"

    def test_skips_defs_that_normalise_to_empty_key(self, _patch):
        _patch(
            fill_missing_from_defs,
            normalize_acronym_key=lambda surface, allow_chars, dotted_mode: "" if surface == "BAD" else surface,
        )

        det_cfg = NS(allow_chars="", dotted_display="strip")
        firsts = {"OK": _fo(0), "BAD": _fo(0)}
        defs = [
            _ed(acronym="OK", definition="Okay", acr_start=0, acr_end=2),
            _ed(acronym="BAD", definition="Nope", acr_start=0, acr_end=3),
        ]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        assert out["OK"] is not None
        assert out["BAD"] is None


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestFillMissingFromDefsIntegration:
    def test_normalises_spaces_around_allowed_separators(self):
        # "R & D" should normalise to "R&D" when allow_chars includes "&"
        det_cfg = NS(allow_chars="&", dotted_display="strip")

        firsts = {"R&D": _fo(50)}
        defs = [
            _ed(
                acronym="R & D",
                definition="Research and Development",
                acr_start=55,
                acr_end=60,
                def_start=0,
                def_end=10,
                confidence=0.9,
                original_definition="Research and Development",
            )
        ]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        pick = out["R&D"]
        assert pick is not None
        assert pick.definition == "Research and Development"

    def test_dotted_mode_strip_indexes_under_undotted_key(self):
        det_cfg = NS(allow_chars="&-/", dotted_display="strip")

        # def acronym includes dots, but firsts key is undotted
        firsts = {"USA": _fo(10)}
        defs = [_ed(acronym="U.S.A.", definition="United States of America", acr_start=12, acr_end=18)]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        assert out["USA"] is not None
        assert out["USA"].definition == "United States of America"

    def test_dotted_mode_preserve_requires_dotted_key(self):
        det_cfg = NS(allow_chars="&-/", dotted_display="preserve")

        firsts = {"U.S.A.": _fo(10)}
        defs = [_ed(acronym="U.S.A.", definition="United States of America", acr_start=12, acr_end=18)]

        out = fill_missing_from_defs("x", firsts=firsts, det_cfg=det_cfg, defs=defs)
        assert out["U.S.A."] is not None
        assert out["U.S.A."].definition == "United States of America"

        # And the undotted key should *not* match in preserve mode
        out2 = fill_missing_from_defs("x", firsts={"USA": _fo(10)}, det_cfg=det_cfg, defs=defs)
        assert out2["USA"] is None
