from types import SimpleNamespace as NS

import pytest
from plainera_unacronym.nlp.common.types import ExtractedDefinition
from plainera_unacronym.nlp.extraction.util import picks_from_global


class TestPicksFromGlobalUnit:
    @pytest.mark.unit
    def test_empty_defs_yields_none(self, monkeypatch):
        monkeypatch.setattr('plainera_unacronym.nlp.extraction.util.extract_iter', lambda t,cfg: iter([]))
        det_cfg = NS(allow_chars=set('-'), dotted_display='strip')
        out = picks_from_global("text", {"pdf": NS(start_offset=10)}, det_cfg)
        assert out == {"pdf": None}

    def test_nearest_then_confidence_then_position(self, monkeypatch):
        # Two defs for same key; same distance; different confidence
        defs = [
            ExtractedDefinition("PDF", "Portable Document Format", "in_text", 0.90, 100,103,  50,  75, "Portable Document Format"),
            ExtractedDefinition("Pdf", "Portable Doc Format",      "in_text", 0.95, 100,103, 150, 175, "Portable Doc Format"),
        ]
        monkeypatch.setattr('plainera_unacronym.nlp.extraction.util.extract_iter', lambda t,cfg: iter(defs))
        normal = lambda acr, allowed, dotted_mode='strip': acr.lower()
        monkeypatch.setattr('plainera_unacronym.nlp.extraction.util.normalize_acronym_key', normal)

        det_cfg = NS(allow_chars=set(), dotted_mode='strip')
        firsts = {"pdf": NS(start_offset=120)}
        out = picks_from_global("text", firsts, det_cfg)
        pick = out["pdf"]
        assert pick.definition == "Portable Doc Format"   # higher confidence wins
        assert pick.acr_span == (100, 103)
        assert pick.def_span == (150, 175)

    def test_missing_key_not_included(self, monkeypatch):
        defs = [
            ExtractedDefinition(
                "GPU", "Graphics Processing Unit", "in_text", 0.9,
                10, 13, 0, 5, "Graphics Processing Unit"
            )
        ]
        monkeypatch.setattr(
            'plainera_unacronym.nlp.extraction.util.extract_iter',
            lambda t, cfg: iter(defs)
        )
        # IMPORTANT: accept dotted_mode kwarg
        monkeypatch.setattr(
            'plainera_unacronym.nlp.extraction.util.normalize_acronym_key',
            lambda a, allowed, dotted_mode='strip': a.lower()
        )

        det_cfg = NS(allow_chars=set(), dotted_display='strip')
        out = picks_from_global("text", firsts={"pdf": NS(start_offset=0)}, det_cfg=det_cfg)
        assert out == {"pdf": None}
