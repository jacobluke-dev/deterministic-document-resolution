from behave import given, step
from behave.runner import Context
from test_kit.helpers.data import load_file, project_path
from test_kit.step_implmentations.db_impl import (
    load_file_into_db_impl,
    load_raw_impl,
    setup_table_impl,
    setup_temp_db_impl,
    teardown_temp_db_impl,
)


def _resolve_sql_path(schema: str, table: str) -> str:
    return project_path(f"testdata/sql/{schema}/{table}.sql")

def _resolve_data_path(file_type: str, database: str, test_number: str, schema_table: str) -> str:
    return project_path(f"testdata/load_data/{file_type}/{database}/{test_number}/{schema_table}.{file_type}")

def _resolve_raw_path(scenario: str, file_ext: str, file_name: str) -> str:
    return project_path(f"testdata/load_data/{file_ext}/{scenario}/{file_name}.{file_ext}")

@given("I setup a temporary database called {db_name}")
def setup_temp_db(context: Context, db_name: str):
    setup_temp_db_impl(context, db_name, prefix="temp_document_resolution")

@step("I teardown the temporary database")
def teardown_temp_db(context: Context):
    teardown_temp_db_impl(context)

@step("I setup a table called {schema_table}")
def setup_table(context: Context, schema_table: str):
    setup_table_impl(context, schema_table, resolve_sql_path=_resolve_sql_path)

@step("I load the {file_type_loc} data into that {database} database into {schema_table} table")
def load_file_into_db(context: Context, file_type_loc: str, database: str, schema_table: str):
    load_file_into_db_impl(context, file_type_loc, database, schema_table, resolve_data_path=_resolve_data_path)

@given("I load the raw {scenario} {file_ext} data, {file_name}")
def load_raw(context: Context, scenario: str, file_ext: str, file_name: str):
    load_raw_impl(context, scenario, file_ext, file_name,
                  resolve_raw_path=_resolve_raw_path, loader=load_file)
