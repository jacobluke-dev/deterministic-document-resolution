from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import public_api.core.services.resolve_service as resolve_service_mod
import pytest
from fastapi import status
from public_api.core.services.resolve_service import ResolveError, ResolveService, _lang_from_locale
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolveRequest


@dataclass(frozen=True)
class _Pick:
    definition: str
    def_span: object
    definition_confidence: float


@dataclass(frozen=True)
class _Definition:
    acronym: str
    definition: str
    def_start: int
    def_end: int
    definition_confidence: float


@dataclass(frozen=True)
class _Occurrence:
    acronym: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class _SpanObj:
    start: int
    end: int


class _IterableSpan:
    def __init__(self, start: int, end: int) -> None:
        self._start = start
        self._end = end

    def __iter__(self):
        yield self._start
        yield self._end


class _LockedSemaphore:
    def locked(self) -> bool:
        return True


class _UnlockedSemaphore:
    def locked(self) -> bool:
        return False


@pytest.fixture
def opts_factory():
    def make(**overrides) -> ResolveOptions:
        return ResolveOptions.model_validate(overrides)

    return make


@pytest.fixture
def service_factory():
    def make(*, meanings: list[dict] | None = None, semaphore=None) -> tuple[ResolveService, Mock]:
        repo = Mock()
        repo.list_meanings.return_value = meanings or []
        svc = ResolveService(
            glossary_repo=repo,
            semaphore=semaphore,
            request_timeout_ms=1000,
            tier2_model=None,
        )
        return svc, repo

    return make


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("en-GB", "en"),
        ("en-US", "en"),
        ("", "en"),
    ],
)
def test_lang_from_locale(locale: str, expected: str):
    assert _lang_from_locale(locale) == expected


class DummyGlossaryRepo:
    def __init__(self, meanings_by_acronym):
        self._meanings_by_acronym = meanings_by_acronym

    def list_meanings(self, *, acronym: str):
        return self._meanings_by_acronym.get(acronym, [])


@pytest.fixture
def opts():
    return ResolveOptions.model_validate(
        {
            "max_definitions_per_acronym": 5,
            "include_glossary_enrichment": True,
        }
    )


@pytest.fixture
def make_service():
    def _make(meanings_by_acronym):
        return ResolveService(
            glossary_repo=DummyGlossaryRepo(meanings_by_acronym),
            semaphore=None,
            request_timeout_ms=1000,
            tier2_model=None,
        )

    return _make


class TestValidateAndPrepare:
    def test_validate_and_prepare_rejects_whitespace_only_text(self, service_factory):
        svc, _ = service_factory()
        payload = ResolveRequest(text="   \n\t   ", options=None)

        with pytest.raises(ResolveError) as exc:
            svc._validate_and_prepare(payload)

        assert exc.value.http_status == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert exc.value.code == ErrorCode.UNPROCESSABLE_ENTITY
        assert exc.value.message == "Text must not be empty."
        assert exc.value.details == {"hint": "Provide non-empty 'text'"}

    def test_validate_and_prepare_rejects_oversized_text(self, monkeypatch, service_factory):
        svc, _ = service_factory()
        monkeypatch.setattr(resolve_service_mod, "TEXT_MAX_LEN", 5)

        payload = ResolveRequest(text="abcdef", options=None)

        with pytest.raises(ResolveError) as exc:
            svc._validate_and_prepare(payload)

        assert exc.value.http_status == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert exc.value.code == ErrorCode.PAYLOAD_TOO_LARGE
        assert exc.value.message == "Body/text too large."
        assert exc.value.details == {"limit": 5, "actual": 6}

    def test_validate_and_prepare_supplies_default_options(self, service_factory):
        svc, _ = service_factory()
        payload = ResolveRequest(text="ABC means Alpha Beta Company.", options=None)

        opts, lang = svc._validate_and_prepare(payload)

        assert opts.model_dump() == ResolveOptions.model_validate({}).model_dump()
        assert lang == _lang_from_locale(opts.locale)

    def test_validate_and_prepare_extracts_lang_from_locale(self, service_factory, opts_factory):
        svc, _ = service_factory()
        payload = ResolveRequest(
            text="ABC means Alpha Beta Company.",
            options=opts_factory(locale="en-GB"),
        )

        opts, lang = svc._validate_and_prepare(payload)

        assert opts.locale == "en-GB"
        assert lang == "en"


class TestSpanStartEnd:
    @pytest.mark.parametrize(
        ("span", "expected"),
        [
            ((1, 4), (1, 4)),
            (_SpanObj(5, 9), (5, 9)),
            (_IterableSpan(10, 12), (10, 12)),
        ],
    )
    def test_span_start_end_supports_multiple_shapes(self, service_factory, span, expected):
        svc, _ = service_factory()
        assert svc._span_start_end(span) == expected

    def test_span_start_end_rejects_unrecognised_shape(self, service_factory):
        svc, _ = service_factory()

        with pytest.raises(TypeError, match="Unrecognised span type"):
            svc._span_start_end(object())


class TestRaiseIfOverloaded:
    def test_raise_if_overloaded_raises_503_when_locked(self, service_factory):
        svc, _ = service_factory(semaphore=_LockedSemaphore())

        with pytest.raises(ResolveError) as exc:
            svc._raise_if_overloaded()

        assert exc.value.http_status == status.HTTP_503_SERVICE_UNAVAILABLE
        assert exc.value.code == ErrorCode.SERVICE_UNAVAILABLE
        assert exc.value.details == {"reason": "OVERLOADED"}


def test_raise_if_overloaded_noops_when_unlocked(service_factory):
    svc, _ = service_factory(semaphore=_UnlockedSemaphore())

    svc._raise_if_overloaded()  # no exception


class TestMaybeGlossaryBlock:
    def test_maybe_glossary_block_returns_none_when_enrichment_disabled(self, service_factory, opts_factory):
        svc, repo = service_factory(
            meanings=[
                {"definition": "Alpha", "domain": "general", "is_active": True},
            ]
        )

        out = svc._maybe_glossary_block("ABC", "en", opts_factory(include_glossary_enrichment=False))

        assert out is None
        repo.list_meanings.assert_not_called()

    def test_maybe_glossary_block_returns_none_when_no_meanings(self, service_factory, opts_factory):
        svc, repo = service_factory(meanings=[])

        out = svc._maybe_glossary_block("ABC", "en", opts_factory(include_glossary_enrichment=True))

        assert out is None
        repo.list_meanings.assert_called_once_with(acronym="ABC")

    def test_maybe_glossary_block_returns_only_active_meanings_in_deterministic_order(
        self,
        service_factory,
        opts_factory,
    ):
        svc, _ = service_factory(
            meanings=[
                {"definition": "Zulu meaning", "domain": "zeta", "is_active": True},
                {"definition": "Ignored inactive", "domain": "general", "is_active": False},
                {"definition": "Apple meaning", "domain": None, "is_active": True},
                {"definition": "Beta meaning", "domain": "alpha", "is_active": True},
                {"definition": "   ", "domain": "general", "is_active": True},
            ]
        )

        out = svc._maybe_glossary_block("ABC", "en", opts_factory(include_glossary_enrichment=True))

        assert out == {
            "matches": [
                {
                    "definition": "Apple meaning",
                    "domain": None,
                    "lang": "en",
                    "confidence": 1.0,
                    "source": "system",
                },
                {
                    "definition": "Beta meaning",
                    "domain": "alpha",
                    "lang": "en",
                    "confidence": 1.0,
                    "source": "system",
                },
                {
                    "definition": "Zulu meaning",
                    "domain": "zeta",
                    "lang": "en",
                    "confidence": 1.0,
                    "source": "system",
                },
            ]
        }


class TestBuildDefinitionsByAcronym:
    def test_build_definitions_by_acronym_places_pick_first_then_sorts_remaining(self, service_factory, opts_factory):
        svc, _ = service_factory()
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

        out = svc._build_definitions_by_acronym(extr=extr, opts=opts_factory(min_confidence=0.0))

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

    def test_build_definitions_by_acronym_dedupes_duplicate_pick_and_ledger_entry(self, service_factory, opts_factory):
        svc, _ = service_factory()
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

        out = svc._build_definitions_by_acronym(extr=extr, opts=opts_factory(min_confidence=0.0))

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

    def test_build_definitions_by_acronym_filters_low_confidence_and_trims(self, service_factory, opts_factory):
        svc, _ = service_factory()
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

        out = svc._build_definitions_by_acronym(
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


class TestMapPipelineToBlocks:
    def test_map_pipeline_to_blocks_orders_blocks_and_occurrences_and_omits_glossary_when_disabled(
        self,
        service_factory,
        opts_factory,
    ):
        svc, repo = service_factory()
        det_res = SimpleNamespace(
            occurrences=[
                _Occurrence("ABC", 30, 33),
                _Occurrence("XYZ", 5, 8),
                _Occurrence("ABC", 10, 13),
            ]
        )
        extr = SimpleNamespace(
            picks={
                "ABC": _Pick("Alpha Beta Company", (40, 58), 0.95),
                "XYZ": _Pick("Xylophone Yard Zone", (60, 79), 0.90),
            },
            definitions=[],
        )

        out = svc._map_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(include_glossary_enrichment=False, return_occurrences=True),
            lang="en",
        )

        assert [b["acronym"] for b in out] == ["XYZ", "ABC"]

        assert out[0]["first_occurrence"] == {"start": 5, "end": 8}
        assert out[0]["occurrences"] == [{"start": 5, "end": 8}]
        assert "glossary" not in out[0]

        assert out[1]["first_occurrence"] == {"start": 10, "end": 13}
        assert out[1]["occurrences"] == [
            {"start": 10, "end": 13},
            {"start": 30, "end": 33},
        ]
        assert "glossary" not in out[1]

        repo.list_meanings.assert_not_called()

    def test_map_pipeline_to_blocks_includes_glossary_when_enabled(self, service_factory, opts_factory):
        svc, repo = service_factory(
            meanings=[
                {"definition": "Alpha Beta Company", "domain": "general", "is_active": True},
            ]
        )
        det_res = SimpleNamespace(
            occurrences=[_Occurrence("ABC", 10, 13)]
        )
        extr = SimpleNamespace(
            picks={"ABC": _Pick("Alpha Beta Company", (20, 38), 0.95)},
            definitions=[],
        )

        out = svc._map_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(include_glossary_enrichment=True, return_occurrences=False),
            lang="en",
        )

        assert out == [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 10, "end": 13},
                "definitions": [
                    {
                        "text": "Alpha Beta Company",
                        "start": 20,
                        "end": 38,
                        "confidence": 0.95,
                        "source": "extracted",
                    }
                ],
                "glossary": {
                    "matches": [
                        {
                            "definition": "Alpha Beta Company",
                            "domain": "general",
                            "lang": "en",
                            "confidence": 1.0,
                            "source": "system",
                        }
                    ]
                },
            }
        ]
        repo.list_meanings.assert_called_once_with(acronym="ABC")


class TestAttachResolutionMetaData:

    def test_attach_resolution_metadata_prefers_document_definition(self, make_service, opts):
        svc = make_service(
            {
                "GP": [
                    {
                        "meaning_id": 1,
                        "definition": "General Partner",
                        "domain": "finance",
                        "provenance": "test",
                        "is_active": True,
                    }
                ]
            }
        )

        blocks = [
            {
                "acronym": "GP",
                "first_occurrence": {"start": 28, "end": 30},
                "definitions": [
                    {
                        "text": "General Practitioner",
                        "start": 4,
                        "end": 24,
                        "confidence": 0.99,
                        "source": "extracted",
                    }
                ],
            }
        ]

        out = svc._attach_resolution_metadata(blocks=blocks, opts=opts)
        block = out[0]

        assert block["selected"]["definition"] == "General Practitioner"
        assert block["selected"]["reason"] == "in_document_definition"
        assert block["candidates"][0]["definition"] == "General Practitioner"
        assert block["conflict"] is True
        assert block["conflict_count"] == 2

    def test_attach_resolution_metadata_prefers_document_definition_and_dedupes_same_glossary_definition(
        self,
        service_factory,
        opts_factory,
    ):
        svc, _ = service_factory(
            meanings=[
                {"meaning_id": 1, "definition": "Alpha Beta Company", "domain": "general", "is_active": True},
                {"meaning_id": 2, "definition": "Another meaning", "domain": "finance", "is_active": True},
                {"meaning_id": 3, "definition": "Inactive meaning", "domain": "legal", "is_active": False},
            ]
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [
                    {
                        "text": "Alpha Beta Company",
                        "start": 10,
                        "end": 28,
                        "confidence": 0.99,
                        "source": "extracted",
                    }
                ],
            }
        ]

        out = svc._attach_resolution_metadata(blocks=blocks, opts=opts_factory(max_definitions_per_acronym=10))
        block = out[0]

        assert block["selected"] == {
            "domain": None,
            "definition": "Alpha Beta Company",
            "reason": "in_document_definition",
        }
        assert block["conflict"] is True
        assert block["conflict_count"] == 2
        assert block["selection"] == {
            "policy_used": None,
            "filtered_inactive_count": 1,
        }

        assert block["candidates"] == [
            {
                "domain": None,
                "definition": "Alpha Beta Company",
                "score": 1.0,
                "provenance": "document",
                "source_ref": "text_span:10-28",
            },
            {
                "domain": "finance",
                "definition": "Another meaning",
                "score": 0.0,
                "provenance": "glossary",
                "source_ref": "meaning:2",
            },
        ]

    def test_attach_resolution_metadata_prefers_general_domain_when_multiple_glossary_candidates(
        self,
        service_factory,
        opts_factory,
    ):
        svc, _ = service_factory(
            meanings=[
                {"meaning_id": 10, "definition": "Specific meaning", "domain": "finance", "is_active": True},
                {"meaning_id": 11, "definition": "General meaning", "domain": "general", "is_active": True},
            ]
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = svc._attach_resolution_metadata(blocks=blocks, opts=opts_factory(max_definitions_per_acronym=10))
        block = out[0]

        assert block["selected"] == {
            "domain": "general",
            "definition": "General meaning",
            "reason": "fallback_general",
        }
        assert block["conflict"] is True
        assert block["conflict_count"] == 2
        assert block["candidates"][0] == {
            "domain": "general",
            "definition": "General meaning",
            "score": 1.0,
            "provenance": "glossary",
            "source_ref": "meaning:11",
        }

    def test_attach_resolution_metadata_uses_highest_score_fallback_when_no_general_candidate(
        self,
        service_factory,
        opts_factory,
    ):
        svc, _ = service_factory(
            meanings=[
                {"meaning_id": 20, "definition": "Alpha meaning", "domain": "zeta", "is_active": True},
                {"meaning_id": 21, "definition": "Beta meaning", "domain": "alpha", "is_active": True},
            ]
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = svc._attach_resolution_metadata(
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=10),
        )
        block = out[0]

        # Current behaviour: when there is no general-domain candidate,
        # selection falls back to deterministic candidate ordering, which
        # sorts by definition text before domain.
        assert block["selected"] == {
            "domain": "zeta",
            "definition": "Alpha meaning",
            "reason": "highest_score",
        }
        assert block["candidates"] == [
            {
                "domain": "zeta",
                "definition": "Alpha meaning",
                "score": 1.0,
                "provenance": "glossary",
                "source_ref": "meaning:20",
            },
            {
                "domain": "alpha",
                "definition": "Beta meaning",
                "score": 0.0,
                "provenance": "glossary",
                "source_ref": "meaning:21",
            },
        ]

    def test_attach_resolution_metadata_respects_max_definitions_per_acronym_for_candidates(
        self,
        service_factory,
        opts_factory,
    ):
        svc, _ = service_factory(
            meanings=[
                {"meaning_id": 1, "definition": "One", "domain": "a", "is_active": True},
                {"meaning_id": 2, "definition": "Two", "domain": "b", "is_active": True},
                {"meaning_id": 3, "definition": "Three", "domain": "c", "is_active": True},
            ]
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = svc._attach_resolution_metadata(blocks=blocks, opts=opts_factory(max_definitions_per_acronym=2))
        block = out[0]

        assert len(block["candidates"]) == 2
