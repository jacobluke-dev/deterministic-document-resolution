from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    List,
)
import re

S = TypeVar("S")


@dataclass(frozen=True)
class StageResult(Generic[S]):
    """Result container returned by a `Stage` function.

    A stage function mutates or transforms a state object and returns it along with
    a short human-readable note that is surfaced in stage reports.

    Attributes:
        value (S): The resulting state after the stage has executed.
        note (str): A short descriptive note (e.g., counts, coverage).
    """
    value: S
    note: str = ""


@dataclass(frozen=True)
class StageReport:
    """Lightweight report emitted for each stage execution.

    Attributes:
        name (str): Stage name.
        ok (bool): Whether the stage completed successfully.
        info (str): Free-form informational message (typically the stage note).
        preview (str | None): Optional preview string summarising the stage output.
    """
    name: str
    ok: bool
    info: str
    preview: Optional[str] = None


@dataclass(frozen=True)
class TraceEvent:
    """Structured trace event describing how selected state fields changed.

    Attributes:
        stage (str): Stage name.
        before (dict): Snapshot of selected fields before the stage ran.
        after (dict): Snapshot of selected fields after the stage ran.
    """
    stage: str
    before: dict
    after: dict


class Tracer:
    """Captures stage-by-stage diffs of selected state fields.

    The tracer snapshots selected attributes (configured per stage by the caller)
    before and after each stage, then records a `TraceEvent` only when the
    snapshot changes.

    Filtering:
        If `filter_regex` is provided, snapshots are filtered to reduce noise.
        - For dict snapshots (e.g. picks), the regex is applied to the dict key.
        - For list snapshots (e.g. definitions), the regex is applied to the item's
          `acronym` attribute (if present).

    Typical usage is inside a stage runner:
        before = tracer.snapshot(state, fields)
        state = stage_fn(state)
        after = tracer.snapshot(state, fields)
        tracer.record(stage_name, before, after)

    Args:
        filter_regex (str | None): Optional regex pattern used to filter snapshot
            rows. Intended for narrowing traces while debugging a specific acronym
            or key. If None, no filtering is applied.
    """

    def __init__(self, filter_regex: Optional[str] = None):
        self._re = re.compile(filter_regex) if filter_regex else None
        self.events: List[TraceEvent] = []

    def snapshot(self, state: Any, fields: Sequence[str]) -> dict:
        """Build a snapshot of selected attributes from `state`.

        Only attributes that exist on `state` are included. Each value is
        normalised via `_snap_value()` into a trace-friendly structure.

        Args:
            state (Any): Pipeline state object (e.g., `FlowState`).
            fields (Sequence[str]): Attribute names to snapshot.

        Returns:
            dict: Mapping of field name to a normalised, serialisable snapshot
            value. Fields not found on `state` are omitted.
        """
        snap: dict = {}
        for f in fields:
            if hasattr(state, f):
                snap[f] = self._snap_value(getattr(state, f))
        return snap

    def record(self, stage: str, before: Optional[dict], after: Optional[dict]) -> None:
        """Record a trace event if the snapshot changed.

        No event is recorded when:
        - `before` or `after` is None
        - the snapshots are equal

        Args:
            stage (str): Stage name.
            before (dict | None): Snapshot taken before stage execution.
            after (dict | None): Snapshot taken after stage execution.

        Returns:
            None
        """
        if before is None or after is None:
            return
        if before == after:
            return
        self.events.append(TraceEvent(stage=stage, before=before, after=after))

    def _snap_value(self, v: Any) -> Any:
        """Normalise common extraction structures into trace-friendly rows.

        Supported inputs:
            - dict[str, InTextPick | None]:
                Converted to a list of row dicts. Entries with None values are
                omitted. If filtering is enabled, the regex is applied to the dict key.
            - list[ExtractedDefinition]:
                Converted to a list of row dicts. If filtering is enabled, the regex
                is applied to each item's `acronym` attribute (when present).

        For unsupported inputs, returns None (so the snapshot contains a stable,
        low-noise representation).

        Args:
            v (Any): Value to normalise.

        Returns:
            Any: A serialisable structure (typically a list of dict rows) or None
            if `v` is not a recognised/handled structure.
        """
        # Picks: dict[str, InTextPick|None] -> list of row dicts
        if isinstance(v, dict):
            rows: list[dict] = []
            for k, p in v.items():
                if p is None:
                    continue
                if self._re and not self._re.search(k):
                    continue
                rows.append(
                    {
                        "acr": k,
                        "definition": getattr(p, "definition", None),
                        "orig": getattr(p, "original_definition", None),
                        "acr_span": getattr(p, "acr_span", None),
                        "def_span": getattr(p, "def_span", None),
                        "conf": getattr(p, "confidence", None),
                    }
                )
            return rows

        # Defs: list[ExtractedDefinition] -> list of row dicts
        if isinstance(v, list):
            rows: list[dict] = []
            for d in v:
                acr = getattr(d, "acronym", None)
                if self._re and (acr is None or not self._re.search(acr)):
                    continue
                rows.append(
                    {
                        "acr": acr,
                        "definition": getattr(d, "definition", None),
                        "orig": getattr(d, "original_definition", None),
                        "spans": (
                            getattr(d, "acr_start", None),
                            getattr(d, "acr_end", None),
                            getattr(d, "def_start", None),
                            getattr(d, "def_end", None),
                        ),
                        "conf": getattr(d, "confidence", None),
                        "src": getattr(d, "source", None),
                    }
                )
            return rows

        return None


class Stage(Generic[S]):
    """Single pipeline stage.

    A stage wraps a pure-ish function that accepts a state object and returns a
    `StageResult[state]`. Stages optionally produce a preview string and may be
    traced via a `Tracer`.

    Args:
        name (str): Stage name (stable identifier used in reports/trace).
        fn (Callable[[S], StageResult[S]]): Stage function. Usually mutates `S` in-place
            and returns it as the result value.
        preview (Callable[[S], str] | None): Optional function to produce a preview string
            from the post-stage state.
        trace_fields (tuple[str, ...]): Attribute names to snapshot for tracing diffs.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[S], StageResult[S]],
        preview: Optional[Callable[[S], str]] = None,
        trace_fields: Tuple[str, ...] = (),
    ):
        self.name = name
        self.fn = fn
        self.preview = preview
        self.trace_fields = trace_fields

    def run(self, x: S, *, tracer: Optional[Tracer] = None) -> Tuple[S, StageReport]:
        """Execute the stage.

        Args:
            x (S): Current pipeline state.
            tracer (Tracer | None): Optional tracer used to capture diffs of `trace_fields`.

        Returns:
            tuple[S, StageReport]: The updated state and a per-stage report.
        """
        before = tracer.snapshot(x, self.trace_fields) if tracer else None
        r = self.fn(x)
        pv = self.preview(r.value) if self.preview else None
        if tracer:
            after = tracer.snapshot(r.value, self.trace_fields)
            tracer.record(self.name, before, after)
        return r.value, StageReport(self.name, True, r.note, pv)


class Chain(Generic[S]):
    """Run a sequence of stages over a shared state object.

    Args:
        stages (list[Stage[S]]): Ordered list of stages to execute.
    """

    def __init__(self, stages: List[Stage[S]]):
        self.stages = stages

    def run(self, x: S, *, tracer: Optional[Tracer] = None) -> Tuple[S, List[StageReport]]:
        """Execute all stages in order.

        Args:
            x (S): Initial pipeline state.
            tracer (Tracer | None): Optional tracer passed to each stage.

        Returns:
            tuple[S, list[StageReport]]: Final state and per-stage reports.
        """
        reports: List[StageReport] = []
        cur: S = x
        for st in self.stages:
            cur, rep = st.run(cur, tracer=tracer)
            reports.append(rep)
        return cur, reports
