from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from public_api.core.services.resolve_mapper import span_start_end, build_definitions_by_acronym


@dataclass(frozen=True)
class _SpanObj:
    start: int
    end: int

@dataclass(frozen=True)
class _Definition:
    acronym: str
    definition: str
    def_start: int
    def_end: int
    definition_confidence: float



@dataclass(frozen=True)
class _Pick:
    definition: str
    def_span: object
    definition_confidence: float



class _IterableSpan:
    def __init__(self, start: int, end: int) -> None:
        self._start = start
        self._end = end

    def __iter__(self):
        yield self._start
        yield self._end


class TestSpanStartEnd:
    @pytest.mark.parametrize(
        ("span", "expected"),
        [
            ((1, 4), (1, 4)),
            (_SpanObj(5, 9), (5, 9)),
            (_IterableSpan(10, 12), (10, 12)),
        ],
    )
    def test_span_start_end_supports_multiple_shapes(self, span, expected):
        assert span_start_end(span) == expected

    def test_span_start_end_rejects_unrecognised_shape(self):

        with pytest.raises(TypeError, match="Unrecognised span type"):
            span_start_end(object())


class TestBuildDefinitionsByAcronym:
    def test_build_definitions_by_acronym_places_pick_first_then_sorts_remaining(self, opts_factory):

        extr = SimpleNamespace(
            picks={
                "ABC": _Pick(
                    definition="winner pick",
                    def_span=(100, 111),
                    definition_confidence=0.20,
                )
            },
            definitions=[
                _Definition("ABC", "zulu ledger", 20, 30, 0.80),
                _Definition("ABC", "alpha ledger", 10, 19, 0.80),
                _Definition("ABC", "mid ledger", 31, 40, 0.50),
            ],
        )

        out = build_definitions_by_acronym(extr=extr, opts=opts_factory(min_confidence=0.0))

        assert out["ABC"] == [
            {
                "text": "winner pick",
                "start": 100,
                "end": 111,
                "confidence": 0.20,
                "source": "extracted",
            },
            {
                "text": "alpha ledger",
                "start": 10,
                "end": 19,
                "confidence": 0.80,
                "source": "extracted",
            },
            {
                "text": "zulu ledger",
                "start": 20,
                "end": 30,
                "confidence": 0.80,
                "source": "extracted",
            },
            {
                "text": "mid ledger",
                "start": 31,
                "end": 40,
                "confidence": 0.50,
                "source": "extracted",
            },
        ]

    def test_build_definitions_by_acronym_dedupes_duplicate_pick_and_ledger_entry(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": _Pick(
                    definition="same text",
                    def_span=(10, 20),
                    definition_confidence=0.95,
                )
            },
            definitions=[
                _Definition("ABC", "same text", 10, 20, 0.95),
                _Definition("ABC", "same text", 10, 20, 0.95),
                _Definition("ABC", "different text", 21, 30, 0.80),
            ],
        )

        out = build_definitions_by_acronym(extr=extr, opts=opts_factory(min_confidence=0.0))

        assert out["ABC"] == [
            {
                "text": "same text",
                "start": 10,
                "end": 20,
                "confidence": 0.95,
                "source": "extracted",
            },
            {
                "text": "different text",
                "start": 21,
                "end": 30,
                "confidence": 0.80,
                "source": "extracted",
            },
        ]

    def test_build_definitions_by_acronym_filters_low_confidence_and_trims(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": _Pick(
                    definition="too low pick",
                    def_span=(1, 5),
                    definition_confidence=0.19,
                )
            },
            definitions=[
                _Definition("ABC", "high one", 10, 20, 0.90),
                _Definition("ABC", "high two", 21, 30, 0.85),
                _Definition("ABC", "too low ledger", 31, 40, 0.10),
            ],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(min_confidence=0.20, max_definitions_per_acronym=1),
        )

        assert out["ABC"] == [
            {
                "text": "high one",
                "start": 10,
                "end": 20,
                "confidence": 0.90,
                "source": "extracted",
            }
        ]
