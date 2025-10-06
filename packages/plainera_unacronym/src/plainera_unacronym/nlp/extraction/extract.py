import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Pattern

from ..common.shared import normalize_definition, tighten_definition_span
from ..common.types import ExtractedDefinition
from .config import ExtractionConfig
from .tighten import tighten_label_by_acronym

__all__ = ["ExtractionConfig", "extract_iter", "ExtractedDefinition", "extract_in_text_definitions"]

# ---------- Pattern builders ----------


def _acr_pat(cfg: ExtractionConfig) -> str:
    return rf"(?P<acr>[{cfg.acr_allowed}]{{{cfg.min_acr_len},{cfg.max_acr_len}}})"


def _def_pat(cfg: ExtractionConfig) -> str:
    # Up to N chars; conservative: forbid ')'
    return rf"(?P<def>[^){{}}]{{1,{cfg.max_phrase_chars}}}?)"


def _compile_parenthetical(cfg: ExtractionConfig) -> tuple[Pattern[str], Pattern[str]]:
    acr, phr = _acr_pat(cfg), _def_pat(cfg)
    fwd = re.compile(rf"\b{phr}\s*\(\s*{acr}\s*\)", re.IGNORECASE | re.MULTILINE)
    rev = re.compile(rf"\b{acr}\s*\(\s*{phr}\s*\)", re.IGNORECASE | re.MULTILINE)
    return fwd, rev


def _compile_inline(cfg: ExtractionConfig, cues: tuple[str, ...]) -> list[Pattern[str]]:
    acr = _acr_pat(cfg)

    # Greedy DEF up to a boundary: end of string OR punctuation (, ; : .) possibly with spaces.
    # This grabs the full "Cost per Acquisition" instead of just "C".
    def_frag = rf"(?P<def>[^){{}}]{{1,{cfg.max_phrase_chars}}})(?=\s*(?:$|[.,;:]))"

    return [
        re.compile(
            rf"\b{acr}\b\s*,?\s*{cue}\s+{def_frag}",
            re.IGNORECASE | re.MULTILINE,
        )
        for cue in cues
    ]


# ---------- Cheap validators ----------
def _has_letters(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))


def _two_words(s: str) -> bool:
    return len([w for w in s.split() if re.search(r"[A-Za-z]", w)]) >= 2


def _initials_match(acr: str, phrase: str) -> bool:
    # Build uppercase initials from the phrase
    initials = "".join(
        w[0].upper()
        for w in phrase.split()
        if w and w[0].isalpha()
    )

    # Compare using ONLY alpha chars from the acronym, uppercased
    acr_letters = [c.upper() for c in acr if c.isalpha()]
    j = 0
    for ch in acr_letters:
        j = initials.find(ch, j) + 1
        if j == 0:
            return False
    return True


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


def _build_plan(cfg: ExtractionConfig) -> ExtractionPlan:
    b = ExtractorBuilder()
    if cfg.plugins:
        try:
            from ..plugins.registry import get as get_plugins  # type: ignore

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


def _parenthetical_allowed(cfg: ExtractionConfig, definition: str, acronym: str) -> bool:
    allows: list[Callable[[str, str], bool]] = getattr(cfg, "_parenthetical_allows", [])  # type: ignore[attr-defined]
    return all(fn(definition, acronym) for fn in allows) if allows else True


def _collect_matches(
    text: str,
    pat: re.Pattern[str],
    *,
    cfg: ExtractionConfig,
    plan: "ExtractionPlan",
    base_conf: float,
    is_parenthetical: bool,
    seen: set[tuple[int, int, int, int]],
    start: int,
    end: int,
) -> Iterator[ExtractedDefinition]:
    for m in pat.finditer(text, start, end):
        acr_raw, def_raw = m.group("acr"), m.group("def")
        if not acr_raw or not def_raw:
            continue

        acronym = acr_raw.strip().upper()
        if not (cfg.min_acr_len <= len(acronym) <= cfg.max_acr_len):
            continue

        original_def = def_raw
        definition = normalize_definition(tighten_definition_span(def_raw))
        if not definition or len(definition) > cfg.max_phrase_chars:
            continue
        if not _has_letters(definition):
            continue
        if cfg.require_two_words and not _two_words(definition):
            continue
        if is_parenthetical:
            # Config-driven allows
            if not _parenthetical_allowed(cfg, definition, acronym):
                continue
            # Plan-driven allows (plugins)
            if plan.parenthetical_allows and not all(fn(definition, acronym) for fn in plan.parenthetical_allows):
                continue

        a0, a1 = m.span("acr")
        d0, d1 = m.span("def")
        key = (a0, a1, d0, d1)
        if key in seen:
            continue
        seen.add(key)

        conf = min(base_conf + (0.03 if _initials_match(acronym, definition) else 0.0), 0.99)
        yield ExtractedDefinition(
            acronym=acronym,
            definition=tighten_label_by_acronym(definition, acronym.upper()),
            source="in_text",
            confidence=conf,
            acr_start=a0,
            acr_end=a1,
            def_start=d0,
            def_end=d1,
            original_definition=original_def,
        )


# ---------- Core API ----------

def extract_iter(
    text: str,
    cfg: ExtractionConfig | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[ExtractedDefinition]:
    cfg = cfg or ExtractionConfig()
    plan = _build_plan(cfg)
    seen: set[tuple[int, int, int, int]] = set()
    s = 0 if start is None else max(0, start)
    e = len(text) if end is None else min(len(text), end)

    if cfg.enabled_parenthetical:
        fwd, rev = _compile_parenthetical(cfg)
        yield from _collect_matches(
            text, fwd, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
            is_parenthetical=True, seen=seen, start=s, end=e
        )
        yield from _collect_matches(
            text, rev, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
            is_parenthetical=True, seen=seen, start=s, end=e
        )

    if cfg.enabled_inline:
        for pat in _compile_inline(cfg, plan.inline_cues):
            yield from _collect_matches(
                text, pat, cfg=cfg, plan=plan, base_conf=cfg.conf_inline,
                is_parenthetical=False, seen=seen, start=s, end=e
            )


def extract_in_text_definitions(text: str, cfg: ExtractionConfig | None = None) -> list[ExtractedDefinition]:
    return list(extract_iter(text, cfg))
