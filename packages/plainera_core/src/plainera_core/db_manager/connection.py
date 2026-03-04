from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar, cast

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

P = ParamSpec("P")
R = TypeVar("R")

def _validate_ident(ident: str) -> str:
    """Allow only simple SQL identifiers: letters, numbers, underscore."""
    ident = ident.strip()
    if not ident:
        raise ValueError("Empty identifier")
    if not ident.replace("_", "").isalnum():
        raise ValueError(f"Invalid identifier: {ident!r}")
    return ident


def _quote_ident(ident: str) -> str:
    return f'"{_validate_ident(ident)}"'


def _quote_table_fqn(table_fqn: str) -> str:
    """
    Quote schema-qualified table like: schema.table or table.
    Reject anything else.
    """
    parts = [p.strip() for p in table_fqn.split(".") if p.strip()]
    if not parts or len(parts) > 2:
        raise ValueError(f"Invalid table name: {table_fqn!r}")
    return ".".join(_quote_ident(p) for p in parts)


def require_allowed_table(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to enforce allowed-table checks on DBManager methods.

    This decorator assumes the first argument after ``self`` is a table
    name (``table_fqn``). If the DBManager instance has a non-empty
    ``allowed_tables`` set, and the provided table name is not in that set,
    a ``ValueError`` is raised before calling the wrapped method.

    Args:
        func (Callable[P, R]): The method to wrap. Must have a signature
            like ``(self, table_fqn: str, *args, **kwargs)``.

    Returns:
        Callable[P, R]: The wrapped method that validates the table name
        before invoking the original function.

    Raises:
        ValueError: If ``table_fqn`` is not in ``self.allowed_tables``.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = cast(Any, args[0])  # bound instance

        if len(args) > 1:
            table_fqn = cast(str, args[1])
        else:
            raw = kwargs.get("table_fqn")
            if raw is None:
                raise ValueError("table_fqn is required")
            table_fqn = str(raw)

        if not table_fqn:
            raise ValueError("table_fqn is required")

        if self.allowed_tables and table_fqn not in self.allowed_tables:
            raise ValueError(f"Invalid table: {table_fqn}")

        return func(*args, **kwargs)

    return cast(Callable[P, R], wrapper)


class DBManager:
    """
    Db Manager, SQLAlchemy engine/session for utility ops.
    """

    def __init__(self,
                 engine: Engine,
                 session_factory: sessionmaker[Session],
                 allowed_tables: set[str] | None = None):
        """Initialize a DBManager.

        Args:
            engine (Engine): SQLAlchemy engine used for database connections.
            session_factory (sessionmaker[Session]): A factory for creating new SQLAlchemy Session objects.
            allowed_tables (set[str] | None): Optional whitelist of fully qualified table names
                that this manager is allowed to operate on. If None or empty, all tables are permitted.
        """
        self.engine = engine
        self.session_factory = session_factory
        self.allowed_tables = allowed_tables or set()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations.

        This context manager yields a SQLAlchemy Session and ensures proper
        commit, rollback, and close behavior.

        Yields:
            Iterator[Session]: An active SQLAlchemy Session object.

        Raises:
            Exception: Re-raises any exception that occurs within the
                context block after rolling back the transaction.
        """
        s = self.session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --- Utilities ---------------------------------------------------------

    def create_schema(self, schema_name: str) -> None:
        """Create a new database schema if it does not already exist.

        Executes a `CREATE SCHEMA IF NOT EXISTS` statement using the
        configured SQLAlchemy engine.

        Args:
            schema_name (str): The name of the schema to create.

        Returns:
            None

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database engine encounters
                an error while executing the statement.
        """
        stmt = text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def execute_sql_file(self, file_path: str | Path) -> None:
        """Execute the SQL statements contained in a file.

        Reads the contents of a `.sql` file and executes them against the
        database using the configured SQLAlchemy engine.

        Args:
            file_path (str | Path): Path to the SQL file. Can be a string
                or a pathlib.Path object.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file does not exist.
            UnicodeDecodeError: If the file cannot be decoded with UTF-8.
            sqlalchemy.exc.SQLAlchemyError: If execution of the SQL fails.

        Examples:
            # >>> dbm.execute_sql_file("migrations/init.sql")
            # Executes all statements in migrations/init.sql
        """
        sql = Path(file_path).read_text(encoding="utf-8")
        with self.engine.begin() as conn:
            conn.exec_driver_sql(sql)

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Run an arbitrary SELECT and return rows.

        Use sparingly.
        """
        with self.engine.connect() as conn:
            res = conn.execute(text(sql), params or {})
            return [tuple(row) for row in res]

    # --- Simple table helpers (keep generic/on purpose) --------------------

    @require_allowed_table
    def insert_row(self, table_fqn: str, columns: Sequence[str], values: Sequence[Any]) -> None:
        """Insert a row into a specified table.

       Constructs an INSERT statement dynamically from the provided
       column names and values, and executes it using the engine.

       Args:
           table_fqn (str): Fully qualified table name (e.g. "public.users").
           columns (Sequence[str]): list or tuple of column names.
           values (Sequence[Any]): Values to insert, in the same order
               as `columns`.

       Returns:
           None

       Raises:
           ValueError: If the table is not in `allowed_tables`.
           sqlalchemy.exc.SQLAlchemyError: If the SQL execution fails.

       Examples:
           >>> dbm.insert_row(
           ...     "glossary_entries",
           ...     ["acronym", "definition"],
           ...     ["NHS", "National Health Service"]
           ... )
        """
        cols = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f":v{i}" for i in range(len(values)))
        params = {f"v{i}": v for i, v in enumerate(values)}
        table_sql = _quote_table_fqn(table_fqn)
        stmt = text(f'INSERT INTO {table_sql} ({cols}) VALUES ({placeholders})')
        with self.engine.begin() as conn:
            conn.execute(stmt, params)

    @require_allowed_table
    def select_rows(
        self,
        table_fqn: str,
        columns: Sequence[str] | None = None,
        where: str | None = None,
        params: dict[str, Any] | None = None,
        order_by: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[tuple[Any, ...]]:
        """Select multiple rows from a specified table.

        Args:
            table_fqn (str): Fully qualified table name.
            columns (Sequence[str] | None): list of column names to select.
                If None, selects all columns ("*").
            where (str | None): Optional WHERE clause (without the "WHERE"
                keyword).
            params (dict[str, Any] | None): Parameters to bind in the WHERE
                clause.
            order_by (Sequence[str] | None): Optional ordering clause
            limit (int | None): Optional limit clause.

        Returns:
            list[tuple]: list of rows returned by the query.

        Raises:
            ValueError: If the table is not in `allowed_tables`.
            sqlalchemy.exc.SQLAlchemyError: If execution fails.
        """
        if not columns:
            col_str = "*"
        else:
            rendered: list[str] = []
            for c in columns:
                c = c.strip()
                if c == "*":
                    rendered.append("*")
                else:
                    rendered.append(_quote_ident(c))
            col_str = ", ".join(rendered)

        table_sql = _quote_table_fqn(table_fqn)

        stmt = f"SELECT {col_str} FROM {table_sql}"
        if where:
            stmt += f" WHERE {where}"

        if order_by:
            ob = ", ".join(_quote_ident(c) for c in order_by)
            stmt += f" ORDER BY {ob}"

        bound = dict(params or {})
        if limit is not None:
            if int(limit) < 0:
                raise ValueError("limit must be >= 0")
            stmt += " LIMIT :__limit"
            bound["__limit"] = int(limit)

        with self.engine.connect() as conn:
            res = conn.execute(text(stmt), bound)
            return [tuple(row) for row in res]

    @require_allowed_table
    def select_one_dict(
        self,
        table_fqn: str,
        columns: Sequence[str],
        criteria: Iterable[tuple[str, str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Select a single row and return it as a dictionary.

        Args:
            table_fqn (str): Fully qualified table name.
            columns (Sequence[str]): Column names to include in the result.
            criteria (Iterable[tuple[str, str, Any]] | None): Optional
                criteria for filtering. Each tuple is of the form
                (column_name, operator, value), where `operator` is either
                "AND" or "OR". The operator is ignored for the last criterion.

        Returns:
            dict[str, Any] | None: A dictionary mapping column names to
            values if a row is found, otherwise None.

        Raises:
            ValueError: If the table is not in `allowed_tables`.
            sqlalchemy.exc.SQLAlchemyError: If execution fails.

        Examples:
            >>> dbm.select_one_dict(
            ...     "glossary_entries",
            ...     ["acronym", "definition"],
            ...     criteria=[("acronym", "", "NHS")]
            ... )
            {'acronym': 'NHS', 'definition': 'National Health Service'}
        """
        where = None
        params: dict[str, Any] = {}

        crit = list(criteria) if criteria else []
        if crit:
            parts: list[str] = []
            for i, (col, joiner, val) in enumerate(crit):
                key = f"p{i}"
                parts.append(f'"{col}" = :{key}')
                params[key] = val

                if i < len(crit) - 1:
                    j = (joiner or "AND").strip().upper()
                    if j not in {"AND", "OR"}:
                        raise ValueError(f"Invalid criteria joiner: {joiner!r} (expected AND/OR)")
                    parts.append(j)

            where = " ".join(parts)

        rows = self.select_rows(table_fqn, columns, where=where, params=params)
        if not rows:
            return None
        return dict(zip(columns, rows[0]))

    @require_allowed_table
    def update_row(
        self,
        table_fqn: str,
        updates: dict[str, Any],
        where: str,
        params: dict[str, Any],
        touch_updated_at: bool = True,
    ) -> None:
        """Update rows in a specified table.

        Args:
            table_fqn (str): Fully qualified table name.
            updates (dict[str, Any]): Column-value pairs to update.
            where (str): WHERE clause (without the "WHERE" keyword).
            params (dict[str, Any]): Parameters to bind in the WHERE clause.
            touch_updated_at (bool): If True, automatically set the
                "updated_at" column to NOW().

        Returns:
            None

        Raises:
            ValueError: If the table is not in `allowed_tables`.
            sqlalchemy.exc.SQLAlchemyError: If execution fails.

        Examples:
            >>> dbm.update_row(
            ...     "glossary_entries",
            ...     {"definition": "Nat. Health Service"},
            ...     where='"acronym" = :acr',
            ...     params={"acr": "NHS"}
            ... )
        """
        sets = [f'"{k}" = :u_{k}' for k in updates]
        update_params = {f"u_{k}": v for k, v in updates.items()}
        if touch_updated_at:
            sets.append('updated_at = NOW()')
        table_sql = _quote_table_fqn(table_fqn)
        stmt = text(f'UPDATE {table_sql} SET {", ".join(sets)} WHERE {where}')
        with self.engine.begin() as conn:
            conn.execute(stmt, {**update_params, **params})

    @require_allowed_table
    def select_rows_where(
        self,
        table_fqn: str,
        columns: Sequence[str] | None,
        criteria: Iterable[tuple[str, Any]] | None = None,
        order_by: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[tuple[Any, ...]]:
        """Select multiple rows from a specified table using simple equality
        criteria.

        This is a convenience wrapper around :meth:`select_rows` that builds a
        parameterised WHERE clause from a list of ``(column, value)`` pairs.
        All criteria are joined with ``AND`` and use equality comparisons.

        Args:
            table_fqn (str): Fully qualified table name (e.g. "public.users" or "users").
            columns (Sequence[str] | None): Column names to select. If None, selects all columns ("*").
            criteria (Iterable[tuple[str, Any]] | None): Optional list of criteria as ``(column, value)``
                pairs. Each criterion becomes ``"<column>" = :pN`` with bound parameters. Criteria are
                combined using ``AND``.
            order_by (Sequence[str] | None): Optional list of columns to order by (ASC). Column names are
                quoted.
            limit (int | None): Optional maximum number of rows to return. Must be >= 0 if provided.

        Returns:
            list[tuple[Any, ...]]: Rows returned by the query as tuples, in the order of ``columns``.

        Raises:
            ValueError: If ``table_fqn`` is not in ``allowed_tables``.
            ValueError: If ``limit`` is provided and is negative.
            sqlalchemy.exc.SQLAlchemyError: If query execution fails.

        Examples:
            >>> dbm.select_rows_where(
            ...     "glossary_acronyms",
            ...     columns=["acronym", "normalized"],
            ...     criteria=[("acronym", "QWE")],
            ... )
            [('QWE', 'qwe')]

            >>> dbm.select_rows_where(
            ...     "glossary_acronyms",
            ...     columns=["acronym"],
            ...     criteria=[("tenant_id", None)],
            ...     order_by=["acronym"],
            ...     limit=10,
            ... )
        """
        where = None
        params: dict[str, Any] = {}

        crit = list(criteria) if criteria else []
        if crit:
            parts: list[str] = []
            for i, (col, val) in enumerate(crit):
                key = f"p{i}"
                parts.append(f'{_quote_ident(col)} = :{key}')
                params[key] = val
            where = " AND ".join(parts)

        return self.select_rows(
            table_fqn=table_fqn,
            columns=columns,
            where=where,
            params=params,
            order_by=order_by,
            limit=limit,
        )
