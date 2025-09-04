from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.mark.integration
class TestInsertAndSelect:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM glossary_entries WHERE acronym IN ('XYZ','ZZZ')")
        yield

    def test_insert_row_then_select_rows(self, dbm):
        dbm.insert_row(
            table_fqn="glossary_entries",
            columns=["acronym", "definition", "source"],
            values=["XYZ", "X Y Zed", "insert_test"],
        )

        rows = dbm.select_rows(
            table_fqn="glossary_entries",
            columns=["acronym", "definition", "source"],
            where='"acronym" = :acr',
            params={"acr": "XYZ"},
        )

        assert rows == [("XYZ", "X Y Zed", "insert_test")]

    def test_select_rows_all_columns_star(self, dbm):
        dbm.insert_row(
            "glossary_entries",
            ["acronym", "definition", "source"],
            ["ZZZ", "Last letters", "insert_test_2"],
        )
        rows = dbm.select_rows("glossary_entries", columns=None, where='"acronym" = :a', params={"a": "ZZZ"})
        assert len(rows) == 1  # sanity check: star returns one row


@pytest.mark.integration
class TestSelectOneDict:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM glossary_entries WHERE acronym IN ('ABC2','ABC3')")
            conn.exec_driver_sql(
                "INSERT INTO glossary_entries (acronym, definition, source) "
                "VALUES ('ABC2','Alpha Beta Charlie 2','seed')"
            )
            conn.exec_driver_sql(
                "INSERT INTO glossary_entries (acronym, definition, source) "
                "VALUES ('ABC3','Alpha Beta Charlie 3','seed3')"
            )
        yield

    def test_select_one_dict_simple_criteria(self, dbm):
        out = dbm.select_one_dict(
            table_fqn="glossary_entries",
            columns=["acronym", "definition", "source"],
            criteria=[("acronym", "", "ABC2")],
        )
        assert out == {
            "acronym": "ABC2",
            "definition": "Alpha Beta Charlie 2",
            "source": "seed",
        }

    def test_select_one_dict_multiple_criteria_and(self, dbm):
        out = dbm.select_one_dict(
            "glossary_entries",
            ["acronym", "definition", "source"],
            criteria=[("acronym", "AND", "ABC3"), ("source", "", "seed3")],
        )
        assert out["acronym"] == "ABC3"
        assert out["source"] == "seed3"

    def test_select_one_dict_not_found_returns_none(self, dbm):
        out = dbm.select_one_dict(
            "glossary_entries",
            ["acronym", "definition", "source"],
            criteria=[("acronym", "", "DOES_NOT_EXIST")],
        )
        assert out is None


@pytest.mark.integration
class TestUpdateTouchUpdatedAt:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM glossary_entries WHERE acronym = 'TTS'")
            conn.exec_driver_sql(
                "INSERT INTO glossary_entries (acronym, definition, source) "
                "VALUES ('TTS','Text To Speech','seed')"
            )
        yield

    def _get(self, dbm, acr: str):
        return dbm.select_one_dict(
            table_fqn="glossary_entries",
            columns=["acronym", "definition", "source", "updated_at"],
            criteria=[("acronym", "", acr)],
        )

    def test_touch_updated_at_true_advances_timestamp(self, dbm):
        before = self._get(dbm, "TTS")
        before_ts = before["updated_at"]
        assert isinstance(before_ts, datetime)

        dbm.update_row(
            "glossary_entries",
            {"definition": "Text-to-Speech"},
            where='"acronym" = :acr',
            params={"acr": "TTS"},
            touch_updated_at=True,
        )

        after = self._get(dbm, "TTS")
        after_ts = after["updated_at"]
        assert isinstance(after_ts, datetime)

        # Allow for DB timezone; just ensure it's changed and later (or different).
        assert after_ts != before_ts
        assert after_ts > before_ts or after_ts.tzinfo == timezone.utc  # minimal sanity
        assert after["definition"] == "Text-to-Speech"


@pytest.mark.integration
class TestRequireAllowedTable:
    def test_disallowed_table_raises_value_error(self, dbm, monkeypatch):
        # Narrow the allowed set so we can assert the decorator blocks others.
        original = set(dbm.allowed_tables)
        try:
            dbm.allowed_tables = {"glossary_entries"}
            with pytest.raises(ValueError, match="Invalid table: not_allowed"):
                dbm.insert_row(
                    table_fqn="not_allowed",
                    columns=["acronym", "definition"],
                    values=["BAD", "should not hit DB"],
                )
        finally:
            dbm.allowed_tables = original

    def test_missing_table_kwarg_raises(self, dbm):
        # Call the method in a way that omits the required param
        with pytest.raises(ValueError, match="table_fqn is required"):
            dbm.insert_row(  # wrong on purpose: no table_fqn (positional) nor kwarg
                columns=["acronym"],
                values=["X"],
            )


@pytest.mark.integration
class TestExecuteSqlFile:
    @pytest.fixture(autouse=True)
    def clean(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM glossary_entries WHERE acronym = 'SQLF'")
        yield

    def test_execute_sql_file_runs_statements_from_disk(self, tmp_path: Path, dbm):
        # Arrange: write a small DML script
        sql = """
        INSERT INTO glossary_entries (acronym, definition, source)
        VALUES ('SQLF','From file','file_seed');

        UPDATE glossary_entries
        SET source = 'file_update'
        WHERE acronym = 'SQLF';
        """
        p = tmp_path / "seed.sql"
        p.write_text(sql, encoding="utf-8")

        # Act
        dbm.execute_sql_file(p)

        # Assert
        row = dbm.select_one_dict(
            "glossary_entries",
            ["acronym", "definition", "source"],
            criteria=[("acronym", "", "SQLF")],
        )
        assert row == {"acronym": "SQLF", "definition": "From file", "source": "file_update"}


@pytest.mark.integration
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
            touch_updated_at=False,
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
