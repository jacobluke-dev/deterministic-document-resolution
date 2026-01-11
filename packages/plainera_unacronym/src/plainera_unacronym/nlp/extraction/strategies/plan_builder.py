from dataclasses import dataclass
from typing import Callable, Iterable

from plainera_unacronym.nlp.extraction.config import ExtractionConfig


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    inline_cues: tuple[str, ...]
    parenthetical_allows: tuple[Callable[[str, str], bool], ...] = ()


# ---------- Plugin hook (optional) ----------
class ExtractorBuilder:
    def __init__(self) -> None:
        self._extra_inline: list[str] = []
        self._parenthetical_allows: list[Callable[[str, str], bool]] = []

    def add_inline_cues(self, cues: Iterable[str]) -> None:
        self._extra_inline.extend(cues)

    def add_parenthetical_allow(self, fn: Callable[[str, str], bool]) -> None:
        self._parenthetical_allows.append(fn)

    @property
    def parenthetical_allows(self):
        return self._parenthetical_allows

    @property
    def extra_inline(self):
        return self._extra_inline


def build_plan(cfg: ExtractionConfig) -> ExtractionPlan:
    b = ExtractorBuilder()
    if cfg.plugins:
        try:
            from plainera_unacronym.nlp.plugins.registry import get as get_plugins  # type: ignore

            for p in get_plugins(cfg.plugins):
                if hasattr(p, "extend_extraction"):
                    p.extend_extraction(b)  # type: ignore[attr-defined]
        except Exception:
            pass
    merged_cues = tuple(cfg.inline_cues) + tuple(b.extra_inline)
    return ExtractionPlan(
        inline_cues=merged_cues,
        parenthetical_allows=tuple(b.parenthetical_allows),
    )
