from dataclasses import dataclass

import pytest
from document_resolution.nlp.extraction.acronyms.meanings.meaning_build import _slug, build_meanings


@dataclass(frozen=True)
class DummyDef:
    acronym: str
    definition: str
    def_start: int
    def_end: int
    definition_confidence: float = 0.9


# ----------------------------
# Unit tests: _slug
# ----------------------------


class TestSlugUnit:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("European Medicines Agency", "european_medicines_agency"),
            ("  spaced   out ", "spaced_out"),
            ("A.B.C", "a_b_c"),
            ("HTTP/2", "http_2"),
            ("___", "x"),
            ("", "x"),
            ("Already_sluggy", "already_sluggy"),  # underscore preserved (non-alnum => _ runs collapse)
            ("naïve café", "na_ve_caf"),  # non-ascii splits to underscores
        ],
    )
    def test_slugging(self, s, expected):
        assert _slug(s) == expected


class TestBuildMeaningsUnit:
    def test_groups_by_upper_acronym_and_merges_same_sid(self, _patch):
        # Patch only dependencies to keep this unit test deterministic.
        _patch(
            build_meanings,
            dedupe_defs=lambda xs: xs,  # identity
            tighten_label=lambda s: s.strip(),  # deterministic label
        )

        defs = [
            DummyDef("ema", "European Medicines Agency", 10, 20),
            DummyDef("EMA", "European Medicines Agency", 30, 40),  # same label => same sid
        ]

        out = build_meanings(defs)

        assert set(out.keys()) == {"EMA"}
        assert len(out["EMA"]) == 1

        meaning = out["EMA"][0]
        assert meaning.acronym == "EMA"
        assert meaning.definition == "European Medicines Agency"
        assert meaning.meaning_id == "ema|european_medicines_agency"
        assert meaning.def_spans == [(10, 20), (30, 40)]
        assert meaning.support == 2

    def test_creates_multiple_meanings_per_acronym_when_labels_differ(self, _patch):
        _patch(
            build_meanings,
            dedupe_defs=lambda xs: xs,
            tighten_label=lambda s: s,  # no change
        )

        defs = [
            DummyDef("NLP", "Natural Language Processing", 0, 10),
            DummyDef("nlp", "Nice Lovely Plants", 20, 30),
        ]

        out = build_meanings(defs)

        assert set(out.keys()) == {"NLP"}
        assert {s.meaning_id for s in out["NLP"]} == {
            "nlp|natural_language_processing",
            "nlp|nice_lovely_plants",
        }
        by_id = {s.meaning_id: s for s in out["NLP"]}
        assert by_id["nlp|natural_language_processing"].support == 1
        assert by_id["nlp|nice_lovely_plants"].support == 1

    def test_dedupe_defs_is_called_with_list_copy(self, _patch):
        seen = {}

        def fake_dedupe(xs):
            # build_meanings should pass a list (because it does list(defs))
            seen["is_list"] = isinstance(xs, list)
            seen["len"] = len(xs)
            return xs

        _patch(
            build_meanings,
            dedupe_defs=fake_dedupe,
            tighten_label=lambda s: s,
        )

        defs = (DummyDef("A", "Alpha", 1, 2), DummyDef("A", "Alpha", 3, 4))  # tuple input
        out = build_meanings(defs)

        assert seen["is_list"] is True
        assert seen["len"] == 2
        assert "A" in out

    def test_build_meanings_prefers_highest_confidence_duplicate(self, _patch):
        # Use real dedupe_defs or patch it to pick max confidence.
        defs = [
            DummyDef("PDF", "Portable Document Format", 0, 10, definition_confidence=0.60),
            DummyDef("pdf", "And, which the Portable Document Format", 20, 30, definition_confidence=0.90),
        ]
        meanings = build_meanings(defs)
        pdf = meanings["PDF"][0]
        assert pdf.meaning_confidence == 0.90
        assert pdf.support == 1  # because dedupe collapses into one def before meaning building


# ----------------------------
# Integration tests: build_meanings + real tighten_label + real slugging
# (patch dedupe_defs only, to avoid depending on dedupe semantics here)
# ----------------------------


class TestBuildMeaningsIntegration:
    def test_real_tighten_label_affects_sid(self, _patch):
        # Keep real tighten_label and _slug; only bypass dedupe behavior.
        _patch(build_meanings, dedupe_defs=lambda xs: xs)

        defs = [
            DummyDef("EMA", "The European Medicines Agency", 5, 15, 0),
            DummyDef("ema", "EMA stands for European Medicines Agency", 50, 80, 0),
        ]

        out = build_meanings(defs)

        assert set(out.keys()) == {"EMA"}
        # Depending on tighten_label rules, both should tighten to "European Medicines Agency"
        assert len(out["EMA"]) == 1

        s = out["EMA"][0]
        assert s.meaning_id == "ema|european_medicines_agency"
        assert s.support == 2
        assert s.def_spans == [(5, 15), (50, 80)]
