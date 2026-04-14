from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from document_resolution_core.db_manager.connection import DBManager


class TestSessionAndDDL:
    def _mk_dbm(self):
        # minimal fakes
        fake_engine = SimpleNamespace(begin=lambda: nullcontext(), connect=lambda: nullcontext())

        # session_factory() returns an object with commit/rollback/close
        class _S:
            def __init__(self): self.committed = self.rolled = self.closed = False

            def commit(self): self.committed = True

            def rollback(self): self.rolled = True

            def close(self): self.closed = True

        sf = mock.Mock()
        sf.return_value = _S()
        return DBManager(fake_engine, sf)

    def test_session_commit_and_close(self):
        dbm = self._mk_dbm()
        with dbm.session() as s:
            assert hasattr(s, "commit")
        # session manager should have committed and closed
        sess = dbm.session_factory.return_value
        assert sess.committed is True
        assert sess.closed is True
        assert sess.rolled is False

    def test_session_rollback_on_exception(self):
        dbm = self._mk_dbm()
        with pytest.raises(RuntimeError), dbm.session() as _:
            raise RuntimeError("boom")
        sess = dbm.session_factory.return_value
        assert sess.rolled is True
        assert sess.closed is True

    def test_create_schema_executes_sql(self):
        fake_conn = mock.MagicMock()
        fake_ctx = mock.MagicMock()
        fake_ctx.__enter__.return_value = fake_conn
        engine = SimpleNamespace(begin=lambda: fake_ctx)
        dbm = DBManager(engine, mock.Mock())
        dbm.create_schema("analytics")
        # validate the exact statement text (quoted name)
        sql_arg = fake_conn.execute.call_args[0][0]
        assert 'CREATE SCHEMA IF NOT EXISTS "analytics"' in str(sql_arg)

    def test_execute_sql_file_raises_file_not_found(self, tmp_path: Path):
        dbm = DBManager(SimpleNamespace(begin=lambda: nullcontext()), mock.Mock())
        with pytest.raises(FileNotFoundError):
            dbm.execute_sql_file(tmp_path / "missing.sql")


class TestSelectRowsColumns:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as c:
            c.exec_driver_sql("DELETE FROM glossary_acronyms WHERE normalized = lower('QWE') AND tenant_id IS NULL")
            c.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'QWE', lower('QWE'), TRUE)"
            )
        yield

    def test_select_specific_columns_are_quoted(self, dbm):
        rows = dbm.select_rows(
            "glossary_acronyms",
            columns=["acronym", "normalized"],
            where='"acronym" = :a AND "tenant_id" IS NULL',
            params={"a": "QWE"},
        )
        assert rows == [("QWE", "qwe")]


class TestInsertAndSelect:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as conn:
            # Clean up any prior runs for the test acronym
            conn.exec_driver_sql("""
                DELETE FROM glossary_meanings
                WHERE acronym_id IN (
                    SELECT id FROM glossary_acronyms
                    WHERE normalized IN (lower('XYZ'), lower('ZZZ')) AND tenant_id IS NULL
                )
            """)
            conn.exec_driver_sql(
                "DELETE FROM glossary_acronyms WHERE normalized IN (lower('XYZ'), lower('ZZZ')) AND tenant_id IS NULL"
            )
        yield

    def test_insert_row_then_select_rows(self, dbm):
        # Create the acronym identity (using raw SQL is fine here; DBManager is what we're testing below)
        with dbm.engine.begin() as conn:
            acronym_id = conn.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'XYZ', lower('XYZ'), TRUE) "
                "RETURNING id"
            ).scalar_one()

        # Insert meaning row via DBManager helper (this is the unit under test)
        dbm.insert_row(
            table_fqn="glossary_meanings",
            columns=["acronym_id", "definition", "domain", "provenance", "is_active"],
            values=[acronym_id, "X Y Zed", "general", "insert_test", True],
        )

        rows = dbm.select_rows(
            table_fqn="glossary_meanings",
            columns=["acronym_id", "definition", "provenance"],
            where='"acronym_id" = :aid AND "domain" = :d',
            params={"aid": acronym_id, "d": "general"},
        )

        assert rows == [(acronym_id, "X Y Zed", "insert_test")]

    def test_select_rows_all_columns_star(self, dbm):
        # seed row (use raw SQL so we’re only testing select_rows() here)
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM glossary_acronyms WHERE normalized = lower('ZZZ') AND tenant_id IS NULL"
            )
            conn.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'ZZZ', lower('ZZZ'), TRUE)"
            )

        rows = dbm.select_rows(
            "glossary_acronyms",
            columns=None,
            where='"acronym" = :a AND "tenant_id" IS NULL',
            params={"a": "ZZZ"},
        )
        assert len(rows) == 1  # sanity check: star returns one row


class TestSelectOneDict:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM glossary_acronyms WHERE normalized IN (lower('ABC2'), lower('ABC3')) AND tenant_id IS NULL"
            )
            conn.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'ABC2', lower('ABC2'), TRUE)"
            )
            conn.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'ABC3', lower('ABC3'), TRUE)"
            )
        yield

    def test_select_one_dict_simple_criteria(self, dbm):
        out = dbm.select_one_dict(
            table_fqn="glossary_acronyms",
            columns=["acronym", "normalized"],
            criteria=[("acronym", "", "ABC2")],
        )
        assert out == {
            "acronym": "ABC2",
            "normalized": "abc2",
        }

    def test_select_one_dict_multiple_criteria_and(self, dbm):
        out = dbm.select_one_dict(
            "glossary_acronyms",
            ["acronym", "normalized"],
            criteria=[("acronym", "AND", "ABC3"), ("normalized", "", "abc3")],
        )
        assert out["acronym"] == "ABC3"
        assert out["normalized"] == "abc3"

    def test_select_one_dict_not_found_returns_none(self, dbm):
        out = dbm.select_one_dict(
            "glossary_acronyms",
            ["acronym", "normalized"],
            criteria=[("acronym", "", "DOES_NOT_EXIST")],
        )
        assert out is None


class TestUpdateTouchUpdatedAt:
    @pytest.fixture(autouse=True)
    def seed(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM glossary_acronyms WHERE normalized = lower('TTS') AND tenant_id IS NULL"
            )
            conn.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'TTS', lower('TTS'), TRUE)"
            )
        yield

    def _get(self, dbm, acr: str):
        return dbm.select_one_dict(
            table_fqn="glossary_acronyms",
            columns=["acronym", "normalized", "updated_at"],
            criteria=[("acronym", "", acr)],
        )

    def test_touch_updated_at_true_advances_timestamp(self, dbm):
        before = self._get(dbm, "TTS")
        before_ts = before["updated_at"]
        assert isinstance(before_ts, datetime)

        dbm.update_row(
            "glossary_acronyms",
            {"acronym": "Text-to-Speech"},  # update a real column on this table
            where='"acronym" = :acr AND "tenant_id" IS NULL',
            params={"acr": "TTS"},
            touch_updated_at=True,
        )

        after = self._get(dbm, "Text-to-Speech")
        after_ts = after["updated_at"]
        assert isinstance(after_ts, datetime)

        assert after_ts != before_ts
        assert after_ts > before_ts or after_ts.tzinfo == timezone.utc  # minimal sanity
        assert after["acronym"] == "Text-to-Speech"


class TestRequireAllowedTable:
    def test_disallowed_table_raises_value_error(self, dbm, monkeypatch):
        # Narrow the allowed set so we can assert the decorator blocks others.
        original = set(dbm.allowed_tables)
        try:
            dbm.allowed_tables = {"glossary_acronyms"}
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


class TestExecuteSqlFile:
    @pytest.fixture(autouse=True)
    def clean(self, dbm):
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM glossary_acronyms WHERE normalized = lower('SQLF') AND tenant_id IS NULL"
            )
        yield

    def test_execute_sql_file_runs_statements_from_disk(self, tmp_path: Path, dbm):
        # Arrange: write a small DML script
        sql = """
        INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active)
        VALUES (NULL, 'SQLF', lower('SQLF'), TRUE);

        UPDATE glossary_acronyms
        SET acronym = 'SQLF_UPDATED'
        WHERE acronym = 'SQLF' AND tenant_id IS NULL;
        """
        p = tmp_path / "seed.sql"
        p.write_text(sql, encoding="utf-8")

        # Act
        dbm.execute_sql_file(p)

        # Assert
        row = dbm.select_one_dict(
            "glossary_acronyms",
            ["acronym", "normalized"],
            criteria=[("normalized", "", "sqlf")],
        )
        assert row == {"acronym": "SQLF_UPDATED", "normalized": "sqlf"}


class TestUpdateRow:
    @pytest.fixture(autouse=True)
    def seed_row(self, dbm):
        # hard reset for this key every test
        with dbm.engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM glossary_acronyms WHERE normalized = lower('ABC') AND tenant_id IS NULL"
            )
            conn.exec_driver_sql(
                "INSERT INTO glossary_acronyms (tenant_id, acronym, normalized, is_active) "
                "VALUES (NULL, 'ABC', lower('ABC'), TRUE)"
            )
        yield

    def _get(self, dbm, acr: str):
        return dbm.select_one_dict(
            table_fqn="glossary_acronyms",
            columns=["acronym", "normalized", "is_active", "updated_at"],
            criteria=[("acronym", "", acr)],
        )

    def test_update_single_column(self, dbm):
        dbm.update_row(
            table_fqn="glossary_acronyms",
            updates={"acronym": "AlphaBetaCharlie"},
            where='"acronym" = :acr AND "tenant_id" IS NULL',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )

        row = self._get(dbm, "AlphaBetaCharlie")
        assert row["acronym"] == "AlphaBetaCharlie"
        assert row["normalized"] == "abc"  # unchanged

    def test_update_multiple_columns(self, dbm):
        dbm.update_row(
            table_fqn="glossary_acronyms",
            updates={"acronym": "ABC_UPDATED", "is_active": False},
            where='"acronym" = :acr AND "tenant_id" IS NULL',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )

        row = self._get(dbm, "ABC_UPDATED")
        assert row["acronym"] == "ABC_UPDATED"
        assert row["is_active"] is False

    def test_kwargs_path_for_decorator(self, dbm):
        dbm.update_row(
            table_fqn="glossary_acronyms",
            updates={"is_active": False},
            where='"acronym" = :acr AND "tenant_id" IS NULL',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )

        row = self._get(dbm, "ABC")
        assert row["is_active"] is False

    def test_where_clause_parameters(self, dbm):
        dbm.update_row(
            "glossary_acronyms",
            {"is_active": False},
            where='"acronym" = :acr AND "tenant_id" IS NULL',
            params={"acr": "NOPE"},
            touch_updated_at=False,
        )

        row = self._get(dbm, "ABC")
        assert row["is_active"] is True  # unchanged

    def test_noop_updates_is_valid_sql(self, dbm):
        dbm.update_row(
            "glossary_acronyms",
            {"is_active": False},
            where='"acronym" = :acr AND "tenant_id" IS NULL',
            params={"acr": "ABC"},
            touch_updated_at=False,
        )

        row = self._get(dbm, "ABC")
        assert row["is_active"] is False
