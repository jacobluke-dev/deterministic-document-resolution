"""
This contains common functions used throughout the feature tests.
"""

import json
import re
from pprint import pprint
from typing import Any

from plainera_core.db_manager.connection import DBManager
from plainera_core.utils.utils import get_project_path


def table_exists_check(dbm: DBManager, schema: str, tbl_name: str) -> tuple[Any, bool]:
    """Checks if a table exists in the specified schema.

    Args:
        dbm (DBManager): The database connection manager.
        schema (str): The schema name.
        tbl_name (str): The table name.

    Returns:
        tuple[Any, bool]: A tuple containing the s and a boolean indicating existence.
    """
    table_exists_query = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
    );
    """
    try:
        with dbm.session() as s:
            s.execute(table_exists_query, (schema, tbl_name))
            val = s.fetchone()[0]
        return s, bool(val)
    except Exception as e:
        raise AssertionError(f"Database error occurred: {e}")


def clear_tables_in_schema(dbm: DBManager, schema_name: str):
    """
    Deletes all entries in all tables within the specified schema.

    Args:
        dbm: The database manager instance to execute SQL commands.
        schema_name (str): The schema name where tables should be cleared.
    """

    get_tables_sql = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{schema_name}';
    """

    try:
        with dbm.session() as s:
            s.execute(get_tables_sql)
            tables = s.fetchall()

            for table in tables:
                table_name = table[0]
                delete_sql = f"DELETE FROM {schema_name}.{table_name};"
                s.execute(delete_sql)
                print(f"Cleared all entries in table: {schema_name}.{table_name}")
    except Exception as e:
        print(f"An error occurred while clearing tables: {e}")
        raise


def print_tables(dbm: DBManager, schema: str):
    """
    This prints out the tables for the given schema passed through.
    Args:
        dbm (DBManager): The Manager
        schema (str): The schema to be printed
    """
    try:
        with dbm.session() as s:
            s.execute(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s;
            """, (schema,))
            tables = s.fetchall()
            print(f"Tables in schema '{schema}':")
            for table in tables:
                print(table[0])
    except Exception as e:
        print(f"Error retrieving tables in schema {schema}: {e}")


def extract_table_script(sql_file: str, table_name: str) -> str:
    """
    Extract the CREATE TABLE script for a specific table from a SQL file.

    Args:
        sql_file (str): The path to the SQL file containing the CREATE TABLE statements.
        table_name (str): The name of the table for which to extract the CREATE TABLE script.

    Returns:
        str: The extracted CREATE TABLE script for the specified table.

    Raises:
        ValueError: If the specified table is not found in the provided SQL file.
    """
    with open(sql_file, 'r') as file:
        sql_content = file.read()

    pattern = re.compile(
        r'CREATE TABLE IF NOT EXISTS ' + re.escape(table_name) + r'\s*\((.*?)\);',
        re.DOTALL
    )
    match = pattern.search(sql_content)

    if match:
        return "CREATE TABLE IF NOT EXISTS " + table_name + " (" + match.group(1) + ");"
    else:
        raise ValueError(f"Table {table_name} not found in the provided SQL file.")


def get_mock_gpt_response(term: str, file_type: str, file_name: str = "gpt_response") -> dict[str, Any]:
    """
    Retrieve the mock GPT response for a given word from a JSON file.

    Args:
        term (str): The word or phrase for which to retrieve the mock GPT response.
        This should be asserted in the calling step.
        file_type (str): The type of the file. Currently, only 'json' is supported.
        file_name (str): The name of the file containing the mock response.

    Returns:
        dict[str, Any]: A dictionary containing the mock GPT response.

    Raises:
        TypeError: If the file type is not supported.
    """
    if file_type == "json":
        test_name = term.lower().replace(" ", "_")
    else:
        raise TypeError(f"filetype, {file_type} not supported, please implement")
    file_path = f"testdata/load_data/{file_type}/gpt_responses/{test_name}/{file_name}.{file_type}"
    file_path = get_project_path(file_path)
    responses = {}
    if file_type == "json":
        with open(file_path) as f:
            response = json.load(f)
            responses[term] = response
        return responses
    else:
        raise TypeError(f"filetype, {file_type} not supported, please implement")


def compare_json(json1: str | dict[str, Any], json2: str | dict[str, Any]) -> bool:
    """
    Compare two JSON objects (either as strings or dictionaries) and return a list of differences.

    Args:
        json1 (str | dict[str, Any]): The first JSON object, as a string or dictionary.
        json2 (str | dict[str, Any]): The second JSON object, as a string or dictionary.

    Returns:
        bool: False if assertion False, True if not
    """
    # If the input is a string, convert it to a dictionary
    if isinstance(json1, str):
        json1 = json.loads(json1)
    if isinstance(json2, str):
        json2 = json.loads(json2)

    differences = []

    def compare_values(key: str, value1: Any, value2: Any, path: str) -> None:
        """
        Compare two values and add Any differences to the differences list.

        Args:
            key (str): The key being compared.
            value1 (Any): The first value.
            value2 (Any): The second value.
            path (str): The path to the current key.
        """
        if type(value1) != type(value2):
            differences.append(f"Type mismatch at {path}: {type(value1).__name__} != {type(value2).__name__}")
        elif isinstance(value1, dict):
            compare_dicts(value1, value2, path)
        elif isinstance(value1, list):
            compare_lists(key, value1, value2, path)
        else:
            if value1 != value2:
                differences.append(f"Value mismatch at {path}: {value1} != {value2}")

    def compare_dicts(dict1: dict[str, Any], dict2: dict[str, Any], path: str) -> None:
        """
        Compare two dictionaries and add Any differences to the differences list.

        Args:
            dict1 (dict[str, Any]): The first dictionary.
            dict2 (dict[str, Any]): The second dictionary.
            path (str): The path to the current dictionary.
        """
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        for key in keys1 - keys2:
            differences.append(f"Key {path}.{key} missing in second JSON")
        for key in keys2 - keys1:
            differences.append(f"Key {path}.{key} missing in first JSON")

        for key in keys1 & keys2:
            compare_values(key, dict1[key], dict2[key], f"{path}.{key}")

    def compare_lists(key: str, list1: list[Any], list2: list[Any], path: str) -> None:
        """
        Compare two lists and add Any differences to the differences list.

        Args:
            key (str): The key being compared.
            list1 (list[Any]): The first list.
            list2 (list[Any]): The second list.
            path (str): The path to the current key.
        """
        if len(list1) != len(list2):
            differences.append(f"list length mismatch at {path}: {len(list1)} != {len(list2)}")
            return

        for index, (item1, item2) in enumerate(zip(list1, list2)):
            compare_values(f"{key}[{index}]", item1, item2, f"{path}[{index}]")

    compare_dicts(json1, json2, "")
    if differences:
        pprint(differences)
        return False
    return True
