from typing import Callable

import numpy as np
import plainera_unacronym.nlp.detection.detector as det
import pytest
from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence, Span
from plainera_unacronym.nlp.extraction import ExtractionConfig


@pytest.fixture
def span() -> Callable[[str, str], Span]:
    def _span(text: str, token: str) -> Span:
        s = text.index(token)
        return s, s + len(token)

    return _span


class NullSink:
    def __call__(self, *a, **k):
        pass

    def __getattr__(self, _):
        return lambda *a, **k: None


@pytest.fixture(autouse=True)
def patch_sink(monkeypatch):
    dummy = NullSink()
    monkeypatch.setattr(det, "sink", dummy, raising=True)
    yield dummy


@pytest.fixture
def _patch(monkeypatch):
    def _apply(func, **replacements):
        g = func.__globals__
        for name, impl in replacements.items():
            monkeypatch.setitem(g, name, impl)
        return func

    return _apply


class DummyCfgCls:
    def __init__(self, max_phrase_chars=80):
        self.max_phrase_chars = max_phrase_chars
        self.require_initials_match = False


@pytest.fixture
def dummy_cfg():
    return DummyCfgCls


class HitCls:
    def __init__(self, tok_left, tok_right=0, hit_tokens=None):
        if hit_tokens is None:
            hit_tokens = {0}
        self.tok_left = tok_left
        self.tok_right = tok_right
        self.hit_tokens = set(hit_tokens)


@pytest.fixture
def hit_cfg():
    return HitCls


@pytest.fixture
def build_stream_seen():
    seen = {}

    def _impl(tokens, **kwargs):
        seen.update(kwargs)
        return "STREAM"

    return _impl, seen


@pytest.fixture
def picked_def():
    def _picked_def(extr, key: str):
        """Return extracted definition for acronym key if present, else None."""
        pick = extr.picks.get(key)
        if pick is None:
            return None
        return pick.definition

    return _picked_def


@pytest.fixture
def cfg() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def fo():
    def _fo(acr: str, s: int, e: int, conf: float = 0.9, cfg: DetectorConfig = None) -> FirstOccurrence:
        if cfg is None:
            cfg = DetectorConfig()
        k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        assert k
        return FirstOccurrence(acronym=acr, start_offset=s, end_offset=e, occurrence_confidence=conf, normalized_key=k)

    return _fo


@pytest.fixture
def occ():
    def _occ(cfg: DetectorConfig, acr: str, s: int, e: int, conf: float = 0.9) -> Occurrence:
        k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        return Occurrence(
            acronym=acr,
            start_offset=s,
            end_offset=e,
            occurrence_confidence=conf,
            segement_window=(max(0, s - 20), e + 20),
            normalized_key=k,
            reasons=None,
        )

    return _occ


@pytest.fixture
def cfg_integrated():
    def _cfg_integrated(require_two_words=True, max_chars=200):
        return (
            DetectorConfig(),
            ExtractionConfig(
                inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
                max_phrase_chars=max_chars,
                require_two_words=require_two_words,
            ),
        )

    return _cfg_integrated


@pytest.fixture(autouse=True)
def _mock_tier2_embeddings(monkeypatch):
    import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2

    def _fast_embed_texts(texts, *, model=None, model_name=None, **_kw):
        xs = list(texts)
        # shape doesn't matter much as long as consistent + non-empty
        return np.zeros((len(xs), 8), dtype=np.float32)

    # This is the real seam Tier-2 uses now
    monkeypatch.setattr(t2, "embed_texts", _fast_embed_texts, raising=True)
