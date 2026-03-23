from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from plainera_unacronym.orchestration.interface import PipelineKey, PipelineRunner


class PipelineRegistryError(Exception):
    """Base registry error."""


class DuplicatePipelineKeyError(PipelineRegistryError):
    """Raised when a pipeline key is registered more than once."""


class UnknownPipelineKeyError(PipelineRegistryError):
    """Raised when a requested pipeline key is not registered."""


@dataclass(slots=True)
class PipelineRegistry:
    """Registry of top-level pipeline runners.

    Resolution order is deterministic and follows registration order.
    """

    _pipelines: dict[PipelineKey, PipelineRunner] = field(default_factory=dict)
    _order: list[PipelineKey] = field(default_factory=list)

    def register(self, runner: PipelineRunner) -> None:
        if runner.key in self._pipelines:
            raise DuplicatePipelineKeyError(
                f"Pipeline already registered for key {runner.key!r}."
            )

        self._pipelines[runner.key] = runner
        self._order.append(runner.key)

    def get(self, key: PipelineKey) -> PipelineRunner:
        try:
            return self._pipelines[key]
        except KeyError as exc:
            raise UnknownPipelineKeyError(
                f"Unknown pipeline key {key!r}."
            ) from exc

    def resolve(
        self,
        requested: Iterable[PipelineKey] | None = None,
    ) -> tuple[PipelineRunner, ...]:
        if requested is None:
            requested_keys = set(self._order)
        else:
            requested_keys = set(requested)
            missing = sorted(
                key for key in requested_keys if key not in self._pipelines
            )
            if missing:
                formatted = ", ".join(repr(key) for key in missing)
                raise UnknownPipelineKeyError(
                    f"Unknown pipeline key(s): {formatted}."
                )

        return tuple(
            self._pipelines[key]
            for key in self._order
            if key in requested_keys
        )

    def keys(self) -> tuple[PipelineKey, ...]:
        return tuple(self._order)
