import threading

import pytest
from public_api.db.models import GlossaryAcronym, GlossaryMeaning, GlossaryVariant
from public_api.db.repos import SqlAlchemyAcronymRepo


class TestSqlAlchemyAcronymRepo:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.session() as s:
            s.query(GlossaryVariant).delete()
            s.query(GlossaryMeaning).delete()
            s.query(GlossaryAcronym).delete()
            s.commit()

            pdf = GlossaryAcronym(
                tenant_id=None,
                acronym="PDF",
                normalized="pdf",
                is_active=True,
            )
            s.add(pdf)
            s.flush()

            s.add_all(
                [
                    GlossaryMeaning(
                        acronym_id=pdf.id,
                        domain="general",
                        definition="Portable Document Format.",
                        provenance="seed",
                        source_ref="seed:pdf:general",
                        is_active=True,
                    ),
                    GlossaryMeaning(
                        acronym_id=pdf.id,
                        domain="statistics",
                        definition="Probability Density Function.",
                        provenance="seed",
                        source_ref="seed:pdf:statistics",
                        is_active=True,
                    ),
                ]
            )

            s.add_all(
                [
                    GlossaryVariant(acronym_id=pdf.id, variant="Portable Document Format"),
                    GlossaryVariant(acronym_id=pdf.id, variant="pdf"),
                    GlossaryVariant(acronym_id=pdf.id, variant="P.D.F."),
                ]
            )

            old = GlossaryAcronym(
                tenant_id=None,
                acronym="OLD",
                normalized="old",
                is_active=False,
            )
            s.add(old)
            s.flush()
            s.add(
                GlossaryMeaning(
                    acronym_id=old.id,
                    domain="general",
                    definition="Should not be returned.",
                    provenance="seed",
                    source_ref="seed:old",
                    is_active=True,
                )
            )

            nod = GlossaryAcronym(
                tenant_id=None,
                acronym="NOD",
                normalized="nod",
                is_active=True,
            )
            s.add(nod)
            s.flush()
            s.add(
                GlossaryMeaning(
                    acronym_id=nod.id,
                    domain="general",
                    definition="Inactive meaning.",
                    provenance="seed",
                    source_ref="seed:nod",
                    is_active=False,
                )
            )

            s.commit()

        yield

    def test_get_definitions_is_case_insensitive(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_definitions("pdf")

        assert len(out) == 1
        assert out[0].acronym == "PDF"
        assert out[0].definition == "Portable Document Format."
        assert out[0].domain == "general"
        assert out[0].provenance == "seed"
        assert out[0].source_ref == "seed:pdf:general"
        assert out[0].is_active is True

    def test_get_definitions_respects_domain(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_definitions("PDF", domain="statistics")

        assert len(out) == 1
        assert out[0].definition == "Probability Density Function."
        assert out[0].domain == "statistics"

    def test_get_definitions_defaults_none_domain_to_general(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_definitions("PDF", domain=None)

        assert len(out) == 1
        assert out[0].definition == "Portable Document Format."
        assert out[0].domain == "general"

    def test_get_definitions_excludes_inactive_acronyms(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_definitions("OLD")

        assert out == []

    def test_get_definitions_excludes_inactive_meanings(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_definitions("NOD")

        assert out == []

    def test_get_by_alias_resolves_variant_case_insensitive(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_by_alias("portable document format")

        assert len(out) == 1
        assert out[0].acronym == "PDF"
        assert out[0].definition == "Portable Document Format."
        assert out[0].domain == "general"

    def test_get_by_alias_respects_domain(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.get_by_alias("pdf", domain="statistics")

        assert len(out) == 1
        assert out[0].definition == "Probability Density Function."
        assert out[0].domain == "statistics"

    def test_list_acronyms_returns_distinct_sorted_matches(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        out = repo.list_acronyms("p")

        assert out == ["PDF"]

    def test_upsert_entry_inserts_when_missing(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        item = repo.upsert_entry(
            "GPU",
            "Graphics Processing Unit",
            domain=None,
            provenance="manual",
            source_ref="seed:gpu",
            is_active=True,
        )

        assert item.acronym == "GPU"
        assert item.definition == "Graphics Processing Unit"
        assert item.domain == "general"
        assert item.provenance == "manual"
        assert item.source_ref == "seed:gpu"
        assert item.is_active is True

        out = repo.get_definitions("gpu")
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_upsert_entry_updates_existing_without_duplicate(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        inserted = repo.upsert_entry(
            "GPU",
            "Graphics Processing Unit",
            domain=None,
            provenance="manual",
            source_ref="seed:gpu:v1",
            is_active=True,
        )
        updated = repo.upsert_entry(
            "GPU",
            "Graphics Processing Unit",
            domain=None,
            provenance="import",
            source_ref="seed:gpu:v2",
            is_active=False,
        )

        assert updated.id == inserted.id
        assert updated.provenance == "import"
        assert updated.source_ref == "seed:gpu:v2"
        assert updated.is_active is False

        with dbm.session() as s:
            acronym = (
                s.query(GlossaryAcronym)
                .filter(GlossaryAcronym.normalized == "gpu")
                .one()
            )
            meanings = (
                s.query(GlossaryMeaning)
                .filter(GlossaryMeaning.acronym_id == acronym.id)
                .all()
            )

        assert len(meanings) == 1

    def test_deactivate_entry_is_idempotent(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        item = repo.get_definitions("PDF")[0]

        repo.deactivate_entry(item.id)
        repo.deactivate_entry(item.id)

        out = repo.get_definitions("PDF")
        assert out == []

    def test_get_definitions_raises_for_blank_acronym(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        with pytest.raises(ValueError, match="acronym must not be blank"):
            repo.get_definitions("   ")

    def test_upsert_entry_raises_for_blank_definition(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        with pytest.raises(ValueError, match="definition must not be blank"):
            repo.upsert_entry(
                "GPU",
                "   ",
                domain=None,
                provenance="manual",
            )

    def test_deactivate_entry_raises_for_non_positive_id(self, dbm):
        repo = SqlAlchemyAcronymRepo(dbm=dbm)

        with pytest.raises(ValueError, match="entry_id must be positive"):
            repo.deactivate_entry(0)


class TestSqlAlchemyAcronymRepoConcurrency:
    def test_upsert_entry_concurrent_calls_do_not_create_duplicates(self, dbm):
        barrier = threading.Barrier(2)
        results = [None, None]
        errors: list[BaseException | None] = [None, None]

        def worker(index: int) -> None:
            try:
                repo = SqlAlchemyAcronymRepo(dbm=dbm)
                barrier.wait()
                results[index] = repo.upsert_entry(
                    "GPU",
                    "Graphics Processing Unit",
                    domain=None,
                    provenance=f"worker-{index}",
                    source_ref=f"test:worker-{index}",
                    is_active=True,
                )
            except BaseException as exc:
                errors[index] = exc

        threads = [
            threading.Thread(target=worker, args=(0,)),
            threading.Thread(target=worker, args=(1,)),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == [None, None], f"worker errors: {errors!r}"

        left = results[0]
        right = results[1]

        assert left is not None
        assert right is not None
        assert left.definition == "Graphics Processing Unit"
        assert right.definition == "Graphics Processing Unit"

        with dbm.session() as s:
            acronyms = (
                s.query(GlossaryAcronym)
                .filter(GlossaryAcronym.normalized == "gpu")
                .all()
            )
            assert len(acronyms) == 1

            meanings = (
                s.query(GlossaryMeaning)
                .filter(
                    GlossaryMeaning.acronym_id == acronyms[0].id,
                    GlossaryMeaning.domain == "general",
                )
                .all()
            )

        assert len(meanings) == 1
