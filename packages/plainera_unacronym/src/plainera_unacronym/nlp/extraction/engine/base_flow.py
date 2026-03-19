from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from plainera_unacronym.nlp.extraction.engine.stages import Chain, StageReport

TState = TypeVar("TState")
TDetRes = TypeVar("TDetRes")
TExtrRes = TypeVar("TExtrRes")
TDetCfg = TypeVar("TDetCfg")
TExtCfg = TypeVar("TExtCfg")


class BaseResolutionFlow(ABC, Generic[TState, TDetRes, TExtrRes, TDetCfg, TExtCfg]):
    """Base class for staged extraction flows."""

    def __init__(self, state_cls: type[TState], det_cfg: TDetCfg, ext_cfg: TExtCfg):
        self.state_cls = state_cls
        self.det_cfg = det_cfg
        self.ext_cfg = ext_cfg

    @abstractmethod
    def build_chain(self) -> Chain[TState]:
        """Build the staged execution chain for the flow."""
        raise NotImplementedError

    def make_state(self, text: str) -> TState:
        """Construct the initial flow state for the provided source text."""
        return self.state_cls(
            text=text,
            det_cfg=self.det_cfg,
            ext_cfg=self.ext_cfg,
        )

    def run(self, text: str) -> tuple[TDetRes, TExtrRes, list[StageReport]]:
        """Run the flow over the provided text."""
        state = self.make_state(text)
        state, reports = self.build_chain().run(state)
        return self._finalize(state, reports)

    @abstractmethod
    def _finalize(
        self,
        state: TState,
        reports: list[StageReport],
    ) -> tuple[TDetRes, TExtrRes, list[StageReport]]:
        """Validate final state and return typed flow outputs."""
        raise NotImplementedError
