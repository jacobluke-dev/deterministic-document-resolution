from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Tier2Config:
    mode: Literal["off", "auto", "on"] = "auto"
    model_name: str = "all-MiniLM-L6-v2"
    weight: float = 0.35  # here for testing only
    # If True, Tier-2 only runs when Tier-1 chose None.
    # If False, Tier-2 may rerank even when Tier-1 chose, but you still gate via ceilings below.
    only_when_undecided: bool = False

    auto_margin_ceiling: float = 0.75


@dataclass(frozen=True, slots=True)
class ResolutionConfig:
    """
    Configuration for final selection across multiple ranking tiers.

    This controls acceptance/arbitration rules applied after Tier-1, Tier-2,
    and any future tiers have produced candidate rankings.
    """

    # Minimum normalized separation required to accept a final chosen sense.
    select_margin_threshold: float = 0.10
