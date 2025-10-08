from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Optional

T = TypeVar("T"); U = TypeVar("U")

@dataclass(frozen=True)
class StageResult(Generic[T]):
    value: T
    note: str = ""

@dataclass(frozen=True)
class StageReport:
    name: str
    ok: bool
    info: str
    preview: Optional[str] = None

class Stage(Generic[T, U]):
    def __init__(self, name: str, fn: Callable[[T], StageResult[U]], preview: Callable[[U], str] | None = None):
        self.name, self.fn, self.preview = name, fn, preview
    def run(self, x: T) -> tuple[U, StageReport]:
        try:
            r = self.fn(x)
            pv = self.preview(r.value) if self.preview else None
            return r.value, StageReport(self.name, True, r.note, pv)
        except Exception as e:
            raise  # prefer surfacing the exception in tests; add try/except only if you want soft-fail logs

class Chain(Generic[T, U]):
    def __init__(self, stages: list[Stage]):
        self.stages = stages
    def run(self, x: T) -> tuple[U, list[StageReport]]:
        reports: list[StageReport] = []
        cur = x
        for st in self.stages:
            cur, rep = st.run(cur)
            reports.append(rep)
        return cur, reports
