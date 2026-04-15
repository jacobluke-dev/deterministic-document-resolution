from typing import Any

from document_resolution_core.db_manager.connection import DBManager


def save_to_db(dbm: DBManager, *,
               table: str,
               level: str,
               function: str,
               message: str,
               details: dict[str, Any] | None = None) -> None:
    dbm.insert_row(
        table,
        ["level_name", "function_name", "message", "details"],
        [level.upper(), function, message, (details or {})],
    )

def log_exists(dbm: DBManager, *,
               level: str | None = None,
               function: str | None = None,
               substr: str | None = None) -> bool:
    where, params = [], {}
    if level is not None:
        where.append("LOWER(level_name) = :lvl")
        params["lvl"] = level.lower()
    if function is not None:
        where.append("function_name = :fn")
        params["fn"] = function
    if substr is not None:
        where.append("message ILIKE :msg")
        params["msg"] = f"%{substr}%"
    rows = dbm.select_rows("logging.logger", ["id"], where=" AND ".join(where) if where else None, params=params)
    return bool(rows)
