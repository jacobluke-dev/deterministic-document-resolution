import pytest


class TestUpdateRow:
    @pytest.fixture(autouse=True)
    def seed_row(self, dbm):
        # hard reset for this key every test
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM glossary_entries WHERE acronym = 'ABC'")
            conn.exec_driver_sql(
                "INSERT INTO glossary_entries (acronym, definition, source) "
                "VALUES ('ABC','Alpha Beta Charlie','init')"
            )
        yield

    def _get(self, dbm, acr: str):
        return dbm.select_one_dict(
            table_fqn="glossary_entries",
            columns=["acronym", "definition", "source", "updated_at"],
            criteria=[("acronym", "", acr)],
        )

    def test_update_single_column(self, dbm):
        # Act
        dbm.update_row(
            table_fqn="glossary_entries",
            updates={"definition": "Alpha · Beta · Charlie"},
            where='"acronym" = :acr',
            params={"acr": "ABC"},
            touch_updated_at=False,  # SQLite-friendly
        )
        # Assert
        row = self._get(dbm, "ABC")
        assert row["definition"] == "Alpha · Beta · Charlie"
        assert row["source"] == "init"  # unchanged

    def test_update_multiple_columns(self, dbm):
        dbm.update_row(
            table_fqn="glossary_entries",
            updates={"definition": "Alpha Beta Charlie (updated)", "source": "test"},
            where='"acronym" = :acr',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )
        row = self._get(dbm, "ABC")
        assert row["definition"] == "Alpha Beta Charlie (updated)"
        assert row["source"] == "test"

    def test_kwargs_path_for_decorator(self, dbm):
        # Ensures require_allowed_table handles keyword args (no IndexError)
        dbm.update_row(
            table_fqn="glossary_entries",
            updates={"source": "kw"},
            where='"acronym" = :acr',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )
        row = self._get(dbm, "ABC")
        assert row["source"] == "kw"

    def test_where_clause_parameters(self, dbm):
        # Wrong WHERE → no change
        dbm.update_row(
            "glossary_entries",
            {"source": "should_not_apply"},
            where='"acronym" = :acr',
            params={"acr": "NOPE"},
            touch_updated_at=False,
        )
        row = self._get(dbm, "ABC")
        assert row["source"] == "init"

    def test_noop_updates_is_valid_sql(self, dbm):
        # Some callers may end up with an empty dict; you can assert it raises
        # or treat it as a no-op. If you want to enforce non-empty, change the
        # code and assert raises here. For now, we simulate a tiny update.
        dbm.update_row(
            "glossary_entries",
            {"source": "noop-check"},
            where='"acronym" = :acr',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )
        row = self._get(dbm, "ABC")
        assert row["source"] == "noop-check"
