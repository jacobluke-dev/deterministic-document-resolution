from __future__ import annotations

from typing import Literal

from pydantic import Field, confloat

from public_api.schemas.base import BaseSchema
from public_api.schemas.shared import TextSpan


class StructuralReferenceBlock(BaseSchema):
    kind: str = Field(..., description="Structural reference kind, for example Section or Schedule.")
    label: str = Field(..., description="Source-close detected label text.")
    canonical_label: str = Field(..., description="Canonicalized label used for deterministic matching.")
    normalized_key: str = Field(..., description="Normalized structural key.")
    canonical_key: str = Field(..., description="Canonical lookup key used for linking.")
    reference_span: TextSpan = Field(..., description="Occurrence span of the structural reference.")
    target_span: TextSpan | None = Field(
        None,
        description="Resolved target heading span, or null if unresolved.",
    )
    match_strategy: Literal["forward", "backward", "overlap", "unresolved"] = Field(
        ...,
        description="Deterministic positional strategy used to resolve the link.",
    )
    strength: confloat(ge=0.0) = Field(  # type: ignore[valid-type]
        ...,
        description="Deterministic link-strength score. Ordinal heuristic, not calibrated confidence.",
    )
    provenance: str = Field(..., description="Provenance tag for the structural reference.")
    resolved: bool = Field(..., description="True when the occurrence resolved to a target span.")
