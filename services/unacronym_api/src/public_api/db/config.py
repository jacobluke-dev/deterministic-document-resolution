from dataclasses import dataclass

@dataclass(frozen=True)
class DBConfig:
    DB_SCHEMA: str = "unacronym"   # single schema for app + logs
    CORE_TABLES: tuple[str, ...] = (
        "unacronym.glossary_entries",
        "unacronym.acronym_aliases",
    )
    LOG_TABLE: str = "unacronym.logger"

    @property
    def allowed_tables(self) -> set[str]:
        return set(self.CORE_TABLES) | {self.LOG_TABLE}

config = DBConfig()
ALLOWED_TABLES = config.allowed_tables
