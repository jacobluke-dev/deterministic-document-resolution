from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plainera_unacronym.nlp.extraction.tiers.types import FloatMat, FloatVec


@lru_cache(maxsize=4)
def _load_st_model(model_name: str, *, cache_folder: str | None = None) -> Any:
    """Load (and memoise) a Sentence-Transformers model by name.

    Uses an LRU cache to avoid repeatedly initialising the same embedding model.
    The cache is intentionally small because models are large in memory.

    Args:
        model_name: Sentence-Transformers model identifier (e.g. "all-MiniLM-L6-v2").
        cache_folder: Optional filesystem path for the HF/SBERT cache. If None,
            the library default is used.

    Returns:
        A `SentenceTransformer` instance ready to encode text.

    Notes:
        The import of `SentenceTransformer` is intentionally lazy to keep the
        module import-light when Tier-2 is disabled.
    """
    st = importlib.import_module("sentence_transformers")
    SentenceTransformer = st.SentenceTransformer
    return SentenceTransformer(
        model_name,
        cache_folder=cache_folder,
    )


def _as_list(xs: Iterable[str]) -> list[str]:
    """Materialise an iterable of strings into a list.

    Args:
        xs: Iterable of strings (may be a generator).

    Returns:
        A list containing the same strings in iteration order.
    """
    return list(xs)


def embed_texts(
    texts: Sequence[str],
    *,
    model: Any | None = None,
    model_name: str | None = None,
) -> FloatMat | None:
    """Embed a batch of texts using Sentence-Transformers.

    Encodes `texts` into a dense float32 embedding matrix and row-normalises the
    result so that dot products equal cosine similarity.

    Args:
        texts: Sequence of input strings to embed.
        model: Sentence-Transformers model to load.
        model_name: Sentence-Transformers model identifier to load.

    Returns:
        A float32 matrix of shape [N, D] (where N == len(texts)) with each row
        L2-normalised, or None if the model could not be loaded or embedding
        failed for any reason.

    Notes:
        - Normalisation is applied here so downstream scoring can use fast dot
          products for cosine similarity.
        - Exceptions are swallowed and represented as `None` to keep Tier-2
          failure non-fatal to the overall pipeline.
    """
    try:
        if model is None:
            if not model_name:
                return None
            model = _load_st_model(model_name)
        sentences = list(texts)
        embs = model.encode(sentences=sentences, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)
    except Exception:
        return None


def cosine_sim01(ctx_vec: FloatVec, cand_mat: FloatMat) -> NDArray[np.floating]:
    """Compute cosine similarity mapped from [-1, 1] into [0, 1].

    This function assumes a single context vector and a matrix of candidate
    vectors. It normalises both inputs defensively to ensure the dot product
    equals cosine similarity, then maps cosine similarity to the [0,1] range.

    Args:
        ctx_vec: Context embedding vector of shape [D].
        cand_mat: Candidate embedding matrix of shape [K, D].

    Returns:
        A vector of shape [K] with similarities in [0, 1], where 1.0 indicates
        identical direction and 0.0 indicates opposite direction.

    Notes:
        Mapping to [0,1] is a monotonic linear transform:
            sim01 = 0.5 * (cos + 1)
        which preserves ranking while being easier to interpret in reports.
    """
    cos = np.asarray(cand_mat) @ np.asarray(ctx_vec)  # [K]
    sim01 = 0.5 * (cos + 1.0)
    return np.clip(sim01, 0.0, 1.0)
