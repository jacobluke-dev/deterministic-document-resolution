import re
from re import Pattern
from typing import Iterator

from plainera_unacronym.nlp.common.types import ExtractedDefinition
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.core.collect import collect_matches
from plainera_unacronym.nlp.extraction.strategies.plan_builder import build_plan


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
    # IMPORTANT:
    # - capture well beyond max_phrase_chars (so max is a gate, not a truncator)
    # - do NOT treat commas as terminators (except for a small copula-clause case)
    # - avoid \b boundaries because acronyms like C/A, R&D don't behave well with \b

    acr = _acr_pat(cfg)

    # Gate-not-truncate: allow a much longer capture, then enforce cfg.max_phrase_chars in collect_matches.
    search_cap = max(cfg.max_phrase_chars * 4, 400)

    # Def fragment:
    # - forbid newline and parentheses/braces
    # - LAZY to stop at the FIRST boundary, not the last
    body = rf"(?P<def>[^\n\(\){{}}]{{1,{search_cap}}}?)"

    # Boundary:
    # - sentence end punctuation, EOS
    # - ALSO allow a comma boundary only when it introduces a copula clause
    #   e.g. "..., is a legacy technique."
    boundary = r"(?=\s*(?:$|[!?;:]|\.(?=\s|$)|,(?=\s+(?:is|are|was|were|be|being|been)\b)))"

    def_frag = body + boundary

    # Better boundaries than \b for punctuation-heavy acronyms:
    left_bd = r"(?<![A-Za-z0-9])"
    right_bd = r"(?![A-Za-z0-9])"

    return [
        re.compile(
            rf"{left_bd}{acr}{right_bd}\s*,?\s*{cue}\s+{def_frag}",
            re.IGNORECASE | re.MULTILINE,
        )
        for cue in cues
    ]

def extract_iter(
    text: str,
    cfg: ExtractionConfig | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[ExtractedDefinition]:
    cfg = cfg or ExtractionConfig()
    plan = build_plan(cfg)
    seen: set[tuple[int, int, int, int]] = set()
    s = 0 if start is None else max(0, start)
    e = len(text) if end is None else min(len(text), end)

    if cfg.enabled_parenthetical:
        fwd, rev = _compile_parenthetical(cfg)
        yield from collect_matches(
            text, fwd, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
            is_parenthetical=True, seen=seen, start=s, end=e
        )
        yield from collect_matches(
            text, rev, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
            is_parenthetical=True, seen=seen, start=s, end=e
        )

    if not getattr(cfg, "enabled_inline", True):
        return


    if cfg.enabled_inline:
        for pat in _compile_inline(cfg, plan.inline_cues):
            yield from collect_matches(
                text, pat, cfg=cfg, plan=plan, base_conf=cfg.conf_inline,
                is_parenthetical=False, seen=seen, start=s, end=e
            )


def extract_in_text_definitions(text: str, cfg: ExtractionConfig | None = None) -> list[ExtractedDefinition]:
    return list(extract_iter(text, cfg))
