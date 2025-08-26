from __future__ import annotations

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine

from public_api.utils.utils import get_project_path

DEFAULT_INI_REL = "alembic.ini"
DEFAULT_SCRIPTS_REL = "public_api/migrations"

def _cfg(alembic_ini_path: str | None) -> Config:
    ini_abs = alembic_ini_path or get_project_path(DEFAULT_INI_REL, raise_error=True)
    cfg = Config(ini_abs)
    # ensure script_location is set even if the ini lacks it / wrong section
    if not cfg.get_main_option("script_location"):
        cfg.set_main_option("script_location", get_project_path(DEFAULT_SCRIPTS_REL, raise_error=True))
    return cfg

def is_at_head(engine: Engine, alembic_ini_path: str | None = None) -> bool:
    cfg = _cfg(alembic_ini_path)
    script = ScriptDirectory.from_config(cfg)
    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    heads = script.get_heads()
    return current in heads
