from dataclasses import replace
from typing import Callable

import numpy as np
import plainera_unacronym.nlp.detection.acronym.detector as det
import plainera_unacronym.nlp.detection.base as bs
import pytest
from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import (
    AcronymDetectorConfig,
    DefinedTermDetectorConfig,
    FirstOccurrence,
    Occurrence,
    Span,
)
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector
from plainera_unacronym.nlp.extraction.acronyms.config import ExtractionConfig


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
    monkeypatch.setattr(bs, "sink", dummy, raising=True)
    yield dummy


@pytest.fixture(autouse=True)
def patch_sink_and_logger(monkeypatch):
    class NullSink:
        def __call__(self, *a, **k):
            pass

        def __getattr__(self, _):
            return lambda *a, **k: None

    dummy_sink = NullSink()
    monkeypatch.setattr(bs, "sink", dummy_sink, raising=True)

    logs = []

    def spy_logger(message, *a, **kw):
        logs.append({"message": message, **kw})

    # Base-level logs
    monkeypatch.setattr(bs, "message_logger", spy_logger, raising=True)
    # Acronym detector logs
    monkeypatch.setattr(det, "message_logger", spy_logger, raising=True)

    return logs


@pytest.fixture
def _patch(monkeypatch):
    """Return a helper that patches names in a function's global namespace.

    The returned helper replaces entries in ``func.__globals__`` using pytest's
    ``monkeypatch.setitem``. This is useful when code under test resolves
    imported symbols from the module where the function is defined, rather than
    from the original import source. It is particularly handy for patching
    module-level collaborators such as loggers, sinks, helper functions, or
    imported dependencies exactly where they are used.

    Args:
        monkeypatch: Built-in pytest fixture used to apply reversible test-time
            patches.

    Returns:
        A callable that accepts a target function plus keyword replacements, where
        each keyword is the global name to replace and each value is the test
        implementation to inject.

    Example:
        Patch ``message_logger`` in the globals of ``detect`` so the wrapped
        method does not write to a real sink during tests::

            _patch(
                detector.detect.__func__,
                message_logger=lambda *args, **kwargs: None,
            )
    """
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
def cfg() -> AcronymDetectorConfig:
    return AcronymDetectorConfig()


@pytest.fixture
def fo():
    def _fo(acr: str, s: int, e: int, conf: float = 0.9, cfg: AcronymDetectorConfig = None) -> FirstOccurrence:
        if cfg is None:
            cfg = AcronymDetectorConfig()
        k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        assert k
        return FirstOccurrence(acronym=acr, start_offset=s, end_offset=e, occurrence_confidence=conf, normalized_key=k)

    return _fo


@pytest.fixture
def occ():
    def _occ(cfg: AcronymDetectorConfig, acr: str, s: int, e: int, conf: float = 0.9) -> Occurrence:
        k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        return Occurrence(
            acronym=acr,
            start_offset=s,
            end_offset=e,
            occurrence_confidence=conf,
            segment_window=(max(0, s - 20), e + 20),
            normalized_key=k,
            reasons=None,
        )

    return _occ


@pytest.fixture
def cfg_integrated():
    def _cfg_integrated(require_two_words=True, max_chars=200):
        return (
            AcronymDetectorConfig(),
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

@pytest.fixture
def test_cfg():
    def _make_cfg(**overrides):
        defaults = {
            "allow_chars": "&/-",
            "window_chars": 80,
            "dotted_display": "strip",
            "debug_reasons": False,
            "debug_anomalies": False,
        }
        defaults.update(overrides)
        return AcronymDetectorConfig(**defaults)

    return _make_cfg


@pytest.fixture
def cfg_terms_det_factory():
    def make(**overrides) -> DefinedTermDetectorConfig:
        return replace(DefinedTermDetectorConfig(), **overrides)
    return make


@pytest.fixture
def defined_term_detector_factory(cfg_terms_det_factory):
    def make(**overrides) -> DefinedTermDetector:
        cfg = cfg_terms_det_factory(**overrides)
        return DefinedTermDetector(cfg)

    return make
