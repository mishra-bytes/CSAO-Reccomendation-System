"""Semantic item embeddings powered by sentence-transformers.

Uses a lightweight transformer model (all-MiniLM-L6-v2, ~80 MB) to generate
384-dim dense embeddings from item names + category descriptions, then reduces
to 8 dims via PCA for efficient use as LightGBM features.

This replaces the synthetic random embeddings with **semantically meaningful**
representations — items with similar names/descriptions cluster together in
embedding space, enabling the ranker to learn cross-item compatibility from
natural-language signal rather than co-occurrence alone.

Design:
- GPU used if available (automatic); CPU fallback (~10 s for 1 K items).
- Embeddings cached to disk (parquet) to avoid re-computation.
- Cosine-similarity matrix exposed for candidate-generation experiments.

Usage:
    embeddings_df = generate_item_embeddings(items_df)
    # → DataFrame with columns [item_id, emb_0, emb_1, ..., emb_7]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target dimensionality after PCA (matches existing pipeline expectation)
DEFAULT_EMBED_DIM = 8
# Hugging Face model — 384-dim all-MiniLM-L6-v2 (22 M params, ~80 MB)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _build_item_texts(items: pd.DataFrame) -> list[str]:
    """Construct a natural-language description for each item.

    Format: "<item_name> — <category> food item priced at ₹<price>"
    This gives the sentence-transformer enough semantic signal to cluster
    similar foods together (e.g., "Paneer Butter Masala — main_course"
    should be near "Dal Makhani — main_course").
    """
    texts: list[str] = []
    for _, row in items.iterrows():
        name = str(row.get("item_name", row.get("item_id", "unknown")))
        cat = str(row.get("item_category", "food"))
        price = row.get("item_price", 0)
        text = f"{name} — {cat} food item priced at ₹{price:.0f}"
        texts.append(text)
    return texts


def generate_item_embeddings(
    items: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    n_components: int = DEFAULT_EMBED_DIM,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Generate PCA-reduced sentence-transformer embeddings for the item catalog.

    Parameters
    ----------
    items : pd.DataFrame
        Must contain ``item_id``; optionally ``item_name``, ``item_category``, ``item_price``.
    model_name : str
        Sentence-transformer model name (from Hugging Face hub).
    n_components : int
        Output embedding dimensionality (default 8 for LightGBM features).
    cache_path : Path | None
        If given, load from / save to this parquet file.
    force_recompute : bool
        Ignore cache and regenerate.

    Returns
    -------
    pd.DataFrame
        Columns: [item_id, emb_0, emb_1, ..., emb_{n_components-1}]
    """
    # --- Check cache ---
    if cache_path and cache_path.exists() and not force_recompute:
        logger.info("Loading cached embeddings from %s", cache_path)
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(items) and set(cached.columns) >= {"item_id", "emb_0"}:
            return cached
        logger.info("Cache mismatch (rows/cols), regenerating embeddings")

    items = items.drop_duplicates("item_id").reset_index(drop=True)
    texts = _build_item_texts(items)
    item_ids = items["item_id"].astype(str).tolist()

    # --- Encode with sentence-transformer ---
    logger.info("Loading sentence-transformer model '%s' ...", model_name)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        logger.info("Encoding %d item texts ...", len(texts))
        raw_embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=128,
            normalize_embeddings=True,
        )
        raw_embeddings = np.array(raw_embeddings, dtype=np.float32)
        logger.info("Raw embedding shape: %s", raw_embeddings.shape)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — falling back to TF-IDF + SVD."
        )
        raw_embeddings = _tfidf_fallback(texts, n_components=min(n_components, 50))

    # --- PCA reduction ---
    reduced = _pca_reduce(raw_embeddings, n_components)

    # --- Build output DataFrame ---
    col_names = [f"emb_{i}" for i in range(reduced.shape[1])]
    df = pd.DataFrame(reduced, columns=col_names)
    df.insert(0, "item_id", item_ids)

    # --- Cache ---
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached embeddings to %s", cache_path)

    return df


def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce dimensionality with PCA (scikit-learn)."""
    from sklearn.decomposition import PCA

    actual_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=actual_components)
    reduced = pca.fit_transform(embeddings)
    explained = sum(pca.explained_variance_ratio_)
    logger.info(
        "PCA: %d → %d dims, %.1f%% variance explained",
        embeddings.shape[1], actual_components, explained * 100,
    )
    return reduced.astype(np.float32)


def _tfidf_fallback(texts: list[str], n_components: int = 8) -> np.ndarray:
    """Fallback: TF-IDF + truncated SVD when sentence-transformers unavailable."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info("TF-IDF fallback: encoding %d texts ...", len(texts))
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(n_components, X.shape[1]))
    reduced = svd.fit_transform(X)
    return reduced.astype(np.float32)


def compute_cosine_similarity_matrix(
    embeddings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a full item×item cosine similarity matrix from embeddings.

    Useful for content-based candidate retrieval or diversity calculations.
    """
    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    mat = embeddings_df[emb_cols].values.astype(np.float32)
    # Normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    mat_normed = mat / norms
    sim = mat_normed @ mat_normed.T
    ids = embeddings_df["item_id"].astype(str).tolist()
    return pd.DataFrame(sim, index=ids, columns=ids)


def get_most_similar_items(
    item_id: str,
    similarity_matrix: pd.DataFrame,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k most similar items (excluding self)."""
    if item_id not in similarity_matrix.index:
        return []
    sims = similarity_matrix.loc[item_id].drop(item_id, errors="ignore")
    top = sims.nlargest(top_k)
    return list(zip(top.index.tolist(), top.values.tolist()))
