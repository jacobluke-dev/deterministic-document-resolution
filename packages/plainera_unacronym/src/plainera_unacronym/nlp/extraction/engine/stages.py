from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, Tuple, TypeVar, List
import re

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class StageResult(Generic[U]):
    value: U
    note: str = ""


@dataclass(frozen=True)
class StageReport:
    name: str
    ok: bool
    info: str
    preview: Optional[str] = None

@dataclass
class TraceEvent:
    stage: str
    before: dict
    after: dict


class Tracer:
    def __init__(self, filter_regex: Optional[str] = None):
        self._re = re.compile(filter_regex) if filter_regex else None
        self.events: List[TraceEvent] = []

    def _snap_value(self, v):
        # Picks: dict[str, InTextPick|None]  -> list of rows
        if isinstance(v, dict):
            rows = []
            for k, p in v.items():
                if p is None: continue
                if self._re and not self._re.search(k): continue
                rows.append({
                    "acr": k,
                    "definition": getattr(p, "definition", None),
                    "orig": getattr(p, "original_definition", None),
                    "acr_span": getattr(p, "acr_span", None),
                    "def_span": getattr(p, "def_span", None),
                    "conf": getattr(p, "confidence", None),
                })
            return rows
        # Defs: list[ExtractedDefinition] -> list of rows
        if isinstance(v, list):
            rows = []
            for d in v:
                acr = getattr(d, "acronym", None)
                if self._re and (acr is None or not self._re.search(acr)): continue
                rows.append({
                    "acr": acr,
                    "definition": getattr(d, "definition", None),
                    "orig": getattr(d, "original_definition", None),
                    "spans": (
                        getattr(d, "acr_start", None), getattr(d, "acr_end", None),
                        getattr(d, "def_start", None), getattr(d, "def_end", None),
                    ),
                    "conf": getattr(d, "confidence", None),
                    "src": getattr(d, "source", None),
                })
            return rows
        return None

    def snapshot(self, state, fields: Sequence[str]) -> dict:
        snap = {}
        for f in fields:
            if hasattr(state, f):
                snap[f] = self._snap_value(getattr(state, f))
        return snap

    def record(self, stage: str, before: dict, after: dict):
        if before == after:
            return
        self.events.append(TraceEvent(stage=stage, before=before, after=after))


class Stage(Generic[T, U]):
    def __init__(
        self,
        name: str,
        fn: Callable[[T], StageResult[U]],
        preview: Optional[Callable[[U], str]] = None,
        trace_fields: Tuple[str, ...] = (),
    ):
        self.name, self.fn, self.preview, self.trace_fields = name, fn, preview, trace_fields

    def run(self, x: T, *, tracer: Optional[Tracer] = None) -> Tuple[U, StageReport]:
        before = tracer.snapshot(x, self.trace_fields) if tracer else None
        r = self.fn(x)
        pv = self.preview(r.value) if self.preview else None
        if tracer:
            after = tracer.snapshot(r.value, self.trace_fields)
            tracer.record(self.name, before, after)
        return r.value, StageReport(self.name, True, r.note, pv)


class Chain(Generic[T, U]):
    def __init__(self, stages: List[Stage[T, U]]):
        self.stages = stages

    def run(self, x: T, *, tracer: Optional[Tracer] = None) -> Tuple[U, List[StageReport]]:
        reports: List[StageReport] = []
        cur = x
        for st in self.stages:
            cur, rep = st.run(cur, tracer=tracer)
            reports.append(rep)
        return cur, reports
