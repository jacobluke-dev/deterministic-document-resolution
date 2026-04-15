from types import SimpleNamespace

import pytest
from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import (
    build_definitions_by_acronym,
    map_acronym_pipeline_to_blocks,
    maybe_glossary_block,
)
from public_api.core.services.resolve_service import ResolveService
from public_api.schemas.resolve import ResolutionMode, ResolveOptions


class DummyGlossaryRepo:
    def __init__(self, meanings_by_acronym):
        self._meanings_by_acronym = meanings_by_acronym

    def list_meanings(self, *, acronym: str):
        return self._meanings_by_acronym.get(acronym, [])


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




@pytest.fixture
def opts():
    return ResolveOptions.model_validate(
        {
            "max_definitions_per_acronym": 5,
            "include_glossary_enrichment": True,
        }
    )



class TestMapPipelineToBlocks:
    def test_map_pipeline_to_blocks_returns_empty_when_no_occurrences(self, opts_factory):
        det_res = SimpleNamespace(occurrences=[])
        extr = SimpleNamespace(picks={}, definitions=[])
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == []

    def test_map_pipeline_to_blocks_orders_by_first_occurrence_then_acronym(self, opts_factory):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ZZZ", start_offset=20, end_offset=23),
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
                SimpleNamespace(acronym="DEF", start_offset=10, end_offset=13),
            ]
        )
        extr = SimpleNamespace(picks={}, definitions=[])
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(return_occurrences=False),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 10, "end": 13},
                "definitions": [],
            },
            {
                "acronym": "DEF",
                "first_occurrence": {"start": 10, "end": 13},
                "definitions": [],
            },
            {
                "acronym": "ZZZ",
                "first_occurrence": {"start": 20, "end": 23},
                "definitions": [],
            },
        ]

    def test_map_pipeline_to_blocks_uses_earliest_occurrence_as_first_occurrence(self, opts_factory):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ABC", start_offset=40, end_offset=43),
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
                SimpleNamespace(acronym="ABC", start_offset=25, end_offset=28),
            ]
        )
        extr = SimpleNamespace(picks={}, definitions=[])
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(return_occurrences=True),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 10, "end": 13},
                "definitions": [],
                "occurrences": [
                    {"start": 10, "end": 13},
                    {"start": 25, "end": 28},
                    {"start": 40, "end": 43},
                ],
            }
        ]

    def test_map_pipeline_to_blocks_includes_definitions_from_extraction(self, opts_factory):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
            ]
        )
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Alpha Beta Corp",
                    definition_confidence=0.95,
                    def_span=(20, 35),
                ),
            },
            definitions=[],
        )
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [{'acronym': 'ABC',
                        'definitions': [{'confidence': 0.95,
                                         'end': 35,
                                         'source': 'extracted',
                                         'start': 20,
                                         'text': 'Alpha Beta Corp'}],
                        'first_occurrence': {'end': 13, 'start': 10},
                        'occurrences': [{'end': 13, 'start': 10}]}
                       ]

    def test_map_pipeline_to_blocks_includes_occurrences_only_when_enabled(self, opts_factory):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
                SimpleNamespace(acronym="ABC", start_offset=20, end_offset=23),
            ]
        )
        extr = SimpleNamespace(picks={}, definitions=[])
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(return_occurrences=False),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 10, "end": 13},
                "definitions": [],
            }
        ]

    def test_map_pipeline_to_blocks_includes_glossary_block_when_present(self, opts_factory):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
            ]
        )
        extr = SimpleNamespace(picks={}, definitions=[])
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                    {"definition": "Alpha meaning", "domain": "general", "is_active": True},
                ]
            }
        )

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(include_glossary_enrichment=True),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [{
            'acronym': 'ABC',
            'definitions': [],
            'first_occurrence': {
                'end': 13, 'start': 10},
            'glossary': {
                'matches': [
                    {
                        'confidence': 1.0,
                        'definition': 'Alpha meaning',
                        'domain': 'general',
                        'lang': 'en',
                        'source': 'system'
                    }]},
            'occurrences': [{
                'end': 13, 'start': 10
            }]
        }]

    def test_map_pipeline_to_blocks_omits_glossary_block_when_none(self, opts_factory):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
            ]
        )
        extr = SimpleNamespace(picks={}, definitions=[])
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={"ABC": []})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(include_glossary_enrichment=True),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [{'acronym': 'ABC',
                        'definitions': [],
                        'first_occurrence': {'end': 13, 'start': 10},
                        'occurrences': [{'end': 13, 'start': 10}]}]

    def test_map_pipeline_to_blocks_includes_acronym_with_occurrences_even_without_definitions(
        self,
        opts_factory,
    ):
        det_res = SimpleNamespace(
            occurrences=[
                SimpleNamespace(acronym="ABC", start_offset=10, end_offset=13),
                SimpleNamespace(acronym="DEF", start_offset=20, end_offset=23),
            ]
        )
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Alpha Beta Corp",
                    definition_confidence=0.95,
                    def_span=(30, 45),
                ),
            },
            definitions=[],
        )
        glossary_repo = DummyGlossaryRepo(meanings_by_acronym={})

        out = map_acronym_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts_factory(),
            lang="en",
            glossary_repo=glossary_repo,
        )

        assert out == [{'acronym': 'ABC',
                        'definitions': [{'confidence': 0.95,
                                         'end': 45,
                                         'source': 'extracted',
                                         'start': 30,
                                         'text': 'Alpha Beta Corp'}],
                        'first_occurrence': {'end': 13, 'start': 10},
                        'occurrences': [{'end': 13, 'start': 10}]},
                       {'acronym': 'DEF',
                        'definitions': [],
                        'first_occurrence': {'end': 23, 'start': 20},
                        'occurrences': [{'end': 23, 'start': 20}]}]


class TestBuildDefinitionsByAcronym:
    def test_build_definitions_by_acronym_returns_empty_when_no_candidates(self, opts_factory):
        extr = SimpleNamespace(
            picks={},
            definitions=[],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(),
        )

        assert out == {}

    def test_build_definitions_by_acronym_ignores_none_picks_and_below_threshold(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": None,
                "DEF": SimpleNamespace(
                    definition="Too low",
                    definition_confidence=0.4,
                    def_span=(10, 20),
                ),
                "GHI": SimpleNamespace(
                    definition="Accepted pick",
                    definition_confidence=0.9,
                    def_span=(30, 40),
                ),
            },
            definitions=[],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(min_confidence=0.5),
        )

        assert out == {
            "GHI": [
                {
                    "text": "Accepted pick",
                    "start": 30,
                    "end": 40,
                    "confidence": 0.9,
                    "source": "extracted",
                }
            ]
        }

    def test_build_definitions_by_acronym_dedupes_definition_against_pick_by_text_and_span(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Alpha Beta Corp",
                    definition_confidence=0.95,
                    def_span=(5, 21),
                ),
            },
            definitions=[
                SimpleNamespace(
                    acronym="ABC",
                    definition="Alpha Beta Corp",
                    definition_confidence=0.7,
                    def_start=5,
                    def_end=21,
                ),
            ],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(),
        )

        assert out == {
            "ABC": [
                {
                    "text": "Alpha Beta Corp",
                    "start": 5,
                    "end": 21,
                    "confidence": 0.95,
                    "source": "extracted",
                }
            ]
        }

    def test_build_definitions_by_acronym_keeps_non_duplicate_definition_with_same_text_different_span(
        self,
        opts_factory,
    ):
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Alpha Beta Corp",
                    definition_confidence=0.95,
                    def_span=(5, 21),
                ),
            },
            definitions=[
                SimpleNamespace(
                    acronym="ABC",
                    definition="Alpha Beta Corp",
                    definition_confidence=0.7,
                    def_start=100,
                    def_end=116,
                ),
            ],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(),
        )

        assert out == {
            "ABC": [
                {
                    "text": "Alpha Beta Corp",
                    "start": 5,
                    "end": 21,
                    "confidence": 0.95,
                    "source": "extracted",
                },
                {
                    "text": "Alpha Beta Corp",
                    "start": 100,
                    "end": 116,
                    "confidence": 0.7,
                    "source": "extracted",
                },
            ]
        }

    def test_build_definitions_by_acronym_orders_pick_first_then_confidence_then_text(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Zulu pick",
                    definition_confidence=0.6,
                    def_span=(1, 10),
                ),
            },
            definitions=[
                SimpleNamespace(
                    acronym="ABC",
                    definition="Beta definition",
                    definition_confidence=0.8,
                    def_start=20,
                    def_end=35,
                ),
                SimpleNamespace(
                    acronym="ABC",
                    definition="Alpha definition",
                    definition_confidence=0.8,
                    def_start=40,
                    def_end=56,
                ),
                SimpleNamespace(
                    acronym="ABC",
                    definition="Lower confidence",
                    definition_confidence=0.7,
                    def_start=60,
                    def_end=76,
                ),
            ],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(),
        )

        assert out == {
            "ABC": [
                {
                    "text": "Zulu pick",
                    "start": 1,
                    "end": 10,
                    "confidence": 0.6,
                    "source": "extracted",
                },
                {
                    "text": "Alpha definition",
                    "start": 40,
                    "end": 56,
                    "confidence": 0.8,
                    "source": "extracted",
                },
                {
                    "text": "Beta definition",
                    "start": 20,
                    "end": 35,
                    "confidence": 0.8,
                    "source": "extracted",
                },
                {
                    "text": "Lower confidence",
                    "start": 60,
                    "end": 76,
                    "confidence": 0.7,
                    "source": "extracted",
                },
            ]
        }

    def test_build_definitions_by_acronym_respects_max_definitions_per_acronym(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Chosen pick",
                    definition_confidence=0.9,
                    def_span=(1, 11),
                ),
            },
            definitions=[
                SimpleNamespace(
                    acronym="ABC",
                    definition="Alpha definition",
                    definition_confidence=0.8,
                    def_start=20,
                    def_end=36,
                ),
                SimpleNamespace(
                    acronym="ABC",
                    definition="Beta definition",
                    definition_confidence=0.7,
                    def_start=40,
                    def_end=55,
                ),
            ],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(max_definitions_per_acronym=2),
        )

        assert out == {
            "ABC": [
                {
                    "text": "Chosen pick",
                    "start": 1,
                    "end": 11,
                    "confidence": 0.9,
                    "source": "extracted",
                },
                {
                    "text": "Alpha definition",
                    "start": 20,
                    "end": 36,
                    "confidence": 0.8,
                    "source": "extracted",
                },
            ]
        }

    def test_build_definitions_by_acronym_removes_internal_is_pick_marker(self, opts_factory):
        extr = SimpleNamespace(
            picks={
                "ABC": SimpleNamespace(
                    definition="Alpha Beta Corp",
                    definition_confidence=0.95,
                    def_span=(5, 21),
                ),
            },
            definitions=[],
        )

        out = build_definitions_by_acronym(
            extr=extr,
            opts=opts_factory(),
        )

        assert "_is_pick" not in out["ABC"][0]


class TestMaybeGlossaryBlock:
    def test_maybe_glossary_block_returns_none_when_enrichment_disabled(self, opts_factory):
        dummy_glossary = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                    {"definition": "Alpha", "domain": "general", "is_active": True},
                ]
            }
        )

        out = maybe_glossary_block(
            glossary_repo=dummy_glossary,
            acronym="ABC",
            lang="en",
            opts=opts_factory(include_glossary_enrichment=False),
        )

        assert out is None

    def test_maybe_glossary_block_returns_none_when_no_meanings(self, opts_factory):
        dummy_glossary = DummyGlossaryRepo(
            meanings_by_acronym={"ABC": []}
        )

        out = maybe_glossary_block(
            glossary_repo=dummy_glossary,
            acronym="ABC",
            lang="en",
            opts=opts_factory(include_glossary_enrichment=True),
        )

        assert out is None

    def test_maybe_glossary_block_returns_only_active_meanings_in_deterministic_order(
        self,
        opts_factory,
    ):
        dummyGlossary = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                    {"definition": "Zulu meaning", "domain": "zeta", "is_active": True},
                    {"definition": "Ignored inactive", "domain": "general", "is_active": False},
                    {"definition": "Apple meaning", "domain": None, "is_active": True},
                    {"definition": "Beta meaning", "domain": "alpha", "is_active": True},
                    {"definition": "   ", "domain": "general", "is_active": True},
                ]
            }
        )

        out = maybe_glossary_block(
            glossary_repo=dummyGlossary,
            acronym="ABC",
            lang="en",
            opts=opts_factory(include_glossary_enrichment=True),
        )

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


class TestAttachResolutionMetaData:

    def test_attach_resolution_metadata_prefers_document_definition(self, opts_factory):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
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

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(),
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )
        block = out[0]

        assert block["selected"]["definition"] == "General Practitioner"
        assert block["selected"]["reason"] == "in_document_definition"
        assert block["candidates"][0]["definition"] == "General Practitioner"
        assert block["conflict"] is True
        assert block["conflict_count"] == 2

    def test_attach_resolution_metadata_prefers_document_definition_and_dedupes_same_glossary_definition(
        self,
        opts_factory,
    ):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                    {"meaning_id": 1, "definition": "Alpha Beta Company", "domain": "general", "is_active": True},
                    {"meaning_id": 2, "definition": "Another meaning", "domain": "finance", "is_active": True},
                    {"meaning_id": 3, "definition": "Inactive meaning", "domain": "legal", "is_active": False},
                ]
            }
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

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=10),
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )
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
        opts_factory,
    ):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                {"meaning_id": 10, "definition": "Specific meaning", "domain": "finance", "is_active": True},
                {"meaning_id": 11, "definition": "General meaning", "domain": "general", "is_active": True},
            ]
            }
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=10),
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )
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
        opts_factory,
    ):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                {"meaning_id": 20, "definition": "Alpha meaning", "domain": "zeta", "is_active": True},
                {"meaning_id": 21, "definition": "Beta meaning", "domain": "alpha", "is_active": True},
            ]
            }
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=10),
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
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
        opts_factory,
    ):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                {"meaning_id": 1, "definition": "One", "domain": "a", "is_active": True},
                {"meaning_id": 2, "definition": "Two", "domain": "b", "is_active": True},
                {"meaning_id": 3, "definition": "Three", "domain": "c", "is_active": True},
            ]
            }
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=2),
            resolution_mode=ResolutionMode.DOMAIN_PRIORITY,
        )
        block = out[0]

        assert len(block["candidates"]) == 2

    def test_attach_resolution_metadata_strict_leaves_unresolved_when_multiple_glossary_candidates(
        self,
        opts_factory,
    ):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                {"meaning_id": 10, "definition": "Specific meaning", "domain": "finance", "is_active": True},
                {"meaning_id": 11, "definition": "General meaning", "domain": "general", "is_active": True},
            ]
            }
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=10),
            resolution_mode=ResolutionMode.STRICT,
        )

        assert out[0]["selected"] is None
        assert out[0]["conflict"] is True
        assert out[0]["conflict_count"] == 2

    def test_attach_resolution_metadata_fallback_general_prefers_general_candidate(
        self,
        opts_factory,
    ):
        glossary_repo = DummyGlossaryRepo(
            meanings_by_acronym={
                "ABC": [
                {"meaning_id": 10, "definition": "Specific meaning", "domain": "finance", "is_active": True},
                {"meaning_id": 11, "definition": "General meaning", "domain": "general", "is_active": True},
            ]
            }
        )

        blocks = [
            {
                "acronym": "ABC",
                "first_occurrence": {"start": 0, "end": 3},
                "definitions": [],
            }
        ]

        out = attach_resolution_metadata(
            glossary_repo=glossary_repo,
            blocks=blocks,
            opts=opts_factory(max_definitions_per_acronym=10),
            resolution_mode=ResolutionMode.FALLBACK_GENERAL,
        )

        assert out[0]["selected"] == {
            "domain": "general",
            "definition": "General meaning",
            "reason": "fallback_general",
        }
        assert out[0]["conflict"] is True
        assert out[0]["conflict_count"] == 2
