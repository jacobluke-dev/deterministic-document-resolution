from pathlib import Path
from typing import cast

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.engine import Engine


def is_at_head(engine: Engine, *, schema: str | None) -> bool:
    if not schema:
        raise ValueError("schema is required for alembic head check")

    alembic_ini_path = Path.cwd() / "alembic.ini"
    if not alembic_ini_path.exists():
        raise FileNotFoundError(f"File '{alembic_ini_path}' does not exist.")

    cfg = Config(str(alembic_ini_path))
    script = ScriptDirectory.from_config(cfg)
    head = cast(str, script.get_current_head())

    with engine.connect() as conn:
        current = cast(
            str,
            conn.execute(
                text(f'SELECT version_num FROM "{schema}".alembic_version')
            ).scalar_one(),
        )

    return current == head
