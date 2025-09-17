# tests/test_logger_mixin_shape.py
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import Integer, inspect as sai

from observability.db.mixins import LoggerCommonMixin

Base = declarative_base()

class _TestLog(Base, LoggerCommonMixin):
    __tablename__ = "tmp_log_shape"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

def test_logger_mixin_columns_shape():
    m = sai(_TestLog)
    cols = {c.key: c for c in m.columns}

    # required basics
    assert cols["level_code"].nullable is False
    assert cols["level_name"].type.length == 16
    assert cols["event"].type.length == 128
    assert cols["logger_type"].type.length == 32

    # optional fields are nullable
    for name in ("function_name", "request_id", "duration_ms", "info", "arguments", "keyword_arguments"):
        assert cols[name].nullable is True

    # timestamp present with server_default
    assert "date_time" in cols
    assert cols["date_time"].server_default is not None
