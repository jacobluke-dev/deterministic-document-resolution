from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from document_resolution.orchestration.interface import PipelineKey, PipelineRunner


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
        """Register a pipeline runner under its stable key.

        Args:
            runner: Top-level pipeline runner to register.

        Raises:
            DuplicatePipelineKeyError: Raised when the runner key is already
                registered.
        """
        if runner.key in self._pipelines:
            raise DuplicatePipelineKeyError(f"Pipeline already registered for key {runner.key!r}.")

        self._pipelines[runner.key] = runner
        self._order.append(runner.key)

    def get(self, key: PipelineKey) -> PipelineRunner:
        """Return the registered runner for a pipeline key.

        Args:
            key: Stable pipeline key.

        Returns:
            The registered pipeline runner.

        Raises:
            UnknownPipelineKeyError: Raised when the key is not registered.
        """
        try:
            return self._pipelines[key]
        except KeyError as exc:
            raise UnknownPipelineKeyError(f"Unknown pipeline key {key!r}.") from exc

    def resolve(self, targets: Sequence[PipelineKey]) -> tuple[PipelineRunner, ...]:
        """Resolve requested targets in deterministic registry order.

        Args:
            targets: Requested pipeline keys.

        Returns:
            Registered runners for the requested keys, ordered by registry
            registration order.

        Raises:
            UnknownPipelineKeyError: Raised when one or more requested keys are
                not registered.
        """
        requested_keys = set(targets)
        missing = sorted(key for key in requested_keys if key not in self._pipelines)
        if missing:
            formatted = ", ".join(repr(key) for key in missing)
            raise UnknownPipelineKeyError(f"Unknown pipeline key(s): {formatted}.")

        return tuple(self._pipelines[key] for key in self._order if key in requested_keys)

    def keys(self) -> tuple[PipelineKey, ...]:
        """Return registered pipeline keys in deterministic order."""
        return tuple(self._order)
