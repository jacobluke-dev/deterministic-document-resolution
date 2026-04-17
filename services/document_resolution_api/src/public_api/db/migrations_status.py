from __future__ import annotations

from typing import cast

from alembic.config import Config
from alembic.script import ScriptDirectory
from document_resolution_core.utils.utils import get_project_path
from sqlalchemy import text
from sqlalchemy.engine import Engine

DEFAULT_INI_REL = "alembic.ini"
DEFAULT_SCRIPTS_REL = "public_api/migrations"

def _cfg(alembic_ini_path: str | None) -> Config:
    ini_abs = alembic_ini_path or get_project_path(DEFAULT_INI_REL, raise_error=True)
    cfg = Config(ini_abs)
    # ensure script_location is set even if the ini lacks it / wrong section
    if not cfg.get_main_option("script_location"):
        cfg.set_main_option("script_location", get_project_path(DEFAULT_SCRIPTS_REL, raise_error=True))
    return cfg


def is_at_head(engine: Engine, *, schema: str | None) -> bool:
    if not schema:
        raise ValueError("schema is required for alembic head check")

    cfg = Config(
        get_project_path(
            "services/document_resolution_api/alembic.ini",
            raise_error=True,
        )
    )
    script = ScriptDirectory.from_config(cfg)
    head = cast(str, script.get_current_head())

    with engine.connect() as conn:
        current = cast(
            str,
            conn.execute(
                text(f'SELECT version_num FROM "{schema}".alembic_version')
            ).scalar_one(),
        )

    return bool(current == head)
