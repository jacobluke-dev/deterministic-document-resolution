from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence, Optional

import numpy as np
from numpy._typing import NDArray

from plainera_unacronym.nlp.extraction.tiers.types import FloatMat, FloatVec


@lru_cache(maxsize=4)
def _load_st_model(model_name: str):
    # Lazy import so missing deps don’t break Tier-1.
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(model_name)


def _as_list(xs: Iterable[str]) -> list[str]:
    return list(xs)


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    # m: [N, D]
    denom = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return m / denom


def embed_texts(model_name: str, texts: Sequence[str]) -> Optional[FloatMat]:
    """Return float32 embeddings [N, D] or None if model not available."""
    try:
        model = _load_st_model(model_name)
        # sentence-transformers returns numpy by default on CPU
        embs = model.encode(texts, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        return _normalize_rows(embs)
    except Exception:
        return None


def cosine_sim01(ctx_vec: FloatVec, cand_mat: FloatMat) -> NDArray[np.floating]:
    """Cosine in [-1,1] -> [0,1]. ctx_vec is [D], cand_mat is [K,D]."""
    cos = cand_mat @ ctx_vec  # dot since both are unit-normalised
    sim01 = 0.5 * (cos + 1.0)
    return np.clip(sim01, 0.0, 1.0)
