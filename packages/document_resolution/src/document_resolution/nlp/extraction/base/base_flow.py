from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

from document_resolution.nlp.extraction.base.stages import Chain, StageReport, TraceEvent, Tracer

TState = TypeVar("TState")
TDetRes = TypeVar("TDetRes")
TExtrRes = TypeVar("TExtrRes")
TDetCfg = TypeVar("TDetCfg")
TExtCfg = TypeVar("TExtCfg")

StateFactory: TypeAlias = Callable[[str, TDetCfg, TExtCfg], TState]


class BaseResolutionFlow(ABC, Generic[TState, TDetRes, TExtrRes, TDetCfg, TExtCfg]):
    """Base class for staged detection/extraction resolution flows.

    A flow is responsible for:
        1. constructing the initial typed state,
        2. building the execution chain,
        3. running the chain over the input text, and
        4. finalising the terminal state into typed outputs.

    Type Parameters:
        TState: Concrete state type threaded through the stage chain.
        TDetRes: Final detector result type returned by the flow.
        TExtrRes: Final extraction / resolution result type returned by the flow.
        TDetCfg: Detector configuration type.
        TExtCfg: Extraction / resolution configuration type.
    """

    def __init__(
        self,
        state_factory: StateFactory[TDetCfg, TExtCfg, TState],
        det_cfg: TDetCfg,
        ext_cfg: TExtCfg,
        trace: bool = False,
        trace_filter: str | None = None,
    ):
        """Initialise the flow.

        Args:
            state_factory: Callable that builds the initial flow state from
                the source text and typed configs.
            det_cfg: Detector configuration attached to the flow.
            ext_cfg: Extraction / resolution configuration attached to the flow.
            trace: Whether to capture structured trace events.
            trace_filter: Optional regex filter applied to traced keys.
        """
        self.state_factory = state_factory
        self.det_cfg = det_cfg
        self.ext_cfg = ext_cfg
        self._tracer = Tracer(trace_filter) if trace else None
        self.trace_events: list[TraceEvent] | None = None

    @abstractmethod
    def build_chain(self) -> Chain[TState]:
        """Return the ordered stage chain for this flow."""
        raise NotImplementedError

    def make_state(self, text: str) -> TState:
        """Construct the initial state for the provided source text.

        Args:
            text: Source document text to process.

        Returns:
            Newly constructed initial state.
        """
        return self.state_factory(text, self.det_cfg, self.ext_cfg)

    def run(self, text: str) -> tuple[TDetRes, TExtrRes, list[StageReport]]:
        """Execute the full flow for a single input text.

        Args:
            text: Source document text to process.

        Returns:
            Final detector result, final extraction/resolution result,
            and per-stage reports.
        """
        state = self.make_state(text)
        tracer = getattr(self, "_tracer", None)
        state, reports = self.build_chain().run(state, tracer=tracer)
        return self._finalize(state, reports)

    @abstractmethod
    def _finalize(
        self,
        state: TState,
        reports: list[StageReport],
    ) -> tuple[TDetRes, TExtrRes, list[StageReport]]:
        """Validate terminal state and return typed flow outputs."""
        raise NotImplementedError
