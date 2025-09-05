from behave.runner import Context
from typing import Callable

from plainera_core.db_manager.connection import DBManager
from test_kit.helpers.db import create_temp_database, drop_database


def setup_temp_db_impl(context: Context, db_name: str, *, prefix: str = "temp") -> None:
    name, dbm = create_temp_database(prefix=f"{prefix}_{db_name}")
    context.temp_db_name = name
    context.db_manager = dbm  # or set different attr via wrapper if needed

def teardown_temp_db_impl(context: Context) -> None:
    if getattr(context, "temp_db_name", None):
        drop_database(context.temp_db_name)
        context.temp_db_name = None

def setup_table_impl(context: Context, schema_table: str, *, resolve_sql_path: Callable[[str, str], str]) -> None:
    dbm: DBManager = context.db_manager
    if "." not in schema_table:
        raise ValueError("Expect SCHEMA.TABLE")
    schema, table = schema_table.split(".", 1)
    dbm.create_schema(schema)
    sql_path = resolve_sql_path(schema, table)
    dbm.execute_sql_file(sql_path)

def load_file_into_db_impl(context: Context, file_type_loc: str, database: str, schema_table: str, *,
                           resolve_data_path: Callable[[str, str, str, str], str]) -> None:
    test_number, file_type = file_type_loc.split(".")
    file_path = resolve_data_path(file_type, database, test_number, schema_table)

    import csv
    with open(file_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    cols = list(rows[0].keys())
    dbm: DBManager = context.db_manager
    for r in rows:
        dbm.insert_row(schema_table, cols, [r[c] for c in cols])

def load_raw_impl(context: Context, scenario: str, file_ext: str, file_name: str, *,
                  resolve_raw_path: Callable[[str, str, str], str],
                  loader: Callable[[str, str], str]) -> None:
    path = resolve_raw_path(scenario, file_ext, file_name)
    content = loader(path, file_ext)
    context.file_content = content
    context.filename = f"{file_name}.{file_ext}"
