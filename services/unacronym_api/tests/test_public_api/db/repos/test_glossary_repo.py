import pytest
from public_api.db.models import GlossaryAcronym, GlossaryMeaning, GlossaryVariant
from public_api.db.repos import GlossaryRepository


class TestGlossaryRepositoryGet:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        # Hard reset the specific data we use in this test class
        with dbm.session() as s:
            # delete in FK order
            s.query(GlossaryVariant).delete()
            s.query(GlossaryMeaning).delete()
            s.query(GlossaryAcronym).delete()
            s.commit()

            # Acronym: PDF
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
                        is_active=True,
                    ),
                    GlossaryMeaning(
                        acronym_id=pdf.id,
                        domain="statistics",
                        definition="Probability Density Function.",
                        provenance="seed",
                        is_active=True,
                    ),
                ]
            )

            s.add_all(
                [
                    GlossaryVariant(acronym_id=pdf.id, variant="Portable Document Format"),
                    GlossaryVariant(acronym_id=pdf.id, variant="pdf"),  # case-insensitive check
                ]
            )

            # Acronym: OLD (inactive identity)
            old = GlossaryAcronym(tenant_id=None, acronym="OLD", normalized="old", is_active=False)
            s.add(old)
            s.flush()
            s.add(
                GlossaryMeaning(
                    acronym_id=old.id,
                    domain="general",
                    definition="Should not be returned.",
                    provenance="seed",
                    is_active=True,
                )
            )

            # Acronym: NOD (meaning inactive)
            nod = GlossaryAcronym(tenant_id=None, acronym="NOD", normalized="nod", is_active=True)
            s.add(nod)
            s.flush()
            s.add(
                GlossaryMeaning(
                    acronym_id=nod.id,
                    domain="general",
                    definition="Inactive meaning.",
                    provenance="seed",
                    is_active=False,
                )
            )

            s.commit()

        yield

    def test_get_returns_general_meaning_by_default(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="PDF")

        assert out == {
            "acronym": "PDF",
            "definition": "Portable Document Format.",
            "provenance": "seed",
        }

    def test_get_returns_requested_domain_when_provided(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="PDF", domain="statistics")

        assert out["acronym"] == "PDF"
        assert out["definition"] == "Probability Density Function."
        assert out["provenance"] == "seed"

    def test_get_falls_back_to_general_when_domain_missing(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="PDF", domain="does_not_exist")

        assert out["definition"] == "Portable Document Format."

    def test_get_resolves_by_variant_case_insensitive(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="portable document format")

        assert out is not None
        assert out["acronym"] == "PDF"
        assert out["definition"] == "Portable Document Format."

    def test_get_returns_none_when_identity_is_inactive(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="OLD")

        assert out is None

    def test_get_returns_none_when_meaning_is_inactive(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="NOD")

        assert out is None

    def test_get_returns_none_when_not_found(self, dbm):
        repo = GlossaryRepository(dbm=dbm)

        out = repo.get(acronym="DOES_NOT_EXIST")

        assert out is None
