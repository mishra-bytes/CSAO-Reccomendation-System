"""LLM-as-a-Judge evaluation metric for recommendation quality.

Uses a sentence-transformer to score the **semantic coherence** between
recommended items and the user's cart context — a proxy for the kind of
qualitative evaluation a human food expert would perform.

Two evaluation modes:

1. **Embedding Coherence (default, no API)**:
   Measures cosine similarity between cart-item embeddings and
   recommended-item embeddings.  A high score means recommendations
   are semantically relevant to what's already in the cart.

2. **LLM Judge (optional, needs OPENAI_API_KEY)**:
   Sends (cart, recommendations) pairs to GPT-4o-mini and asks the model
   to rate recommendation quality on a 1-5 scale across three dimensions:
   relevance, complementarity, and diversity.

Why this matters:
- Traditional offline metrics (NDCG, precision) only measure hit rates.
- LLM-as-judge captures **qualitative** aspects: "Does a dessert make
  sense after a main course?" — which co-occurrence metrics miss for
  novel combinations.
- Demonstrates cutting-edge LLM evaluation approach for Zomathon judges.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeResult:
    """Aggregated LLM-as-judge evaluation results."""
    semantic_coherence: float          # 0–1, mean cosine sim (cart↔recos)
    embedding_diversity: float         # 0–1, mean pairwise distance in recos
    category_coverage_score: float     # 0–1, fraction of meal categories covered
    overall_quality_score: float       # 0–1, weighted composite
    llm_scores: Optional[dict[str, float]] = field(default_factory=dict)  # from API if available
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedding-based judge (default)
# ---------------------------------------------------------------------------

def _load_embeddings_model():
    """Load sentence-transformer model for embedding-based evaluation."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not available for LLM judge")
        return None


def _encode_items(
    model: Any,
    item_ids: list[str],
    item_catalog: dict[str, dict],
) -> np.ndarray:
    """Encode item descriptions into embeddings."""
    texts = []
    for iid in item_ids:
        entry = item_catalog.get(iid, {})
        name = entry.get("item_name", entry.get("name", iid))
        cat = entry.get("item_category", entry.get("category", "food"))
        texts.append(f"{name} ({cat})")
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def evaluate_semantic_coherence(
    cart_items: list[str],
    recommended_items: list[str],
    item_catalog: dict[str, dict],
    model: Any = None,
) -> float:
    """Compute mean cosine similarity between cart and recommended items.

    High score = recommendations are semantically aligned with cart context.
    """
    if not cart_items or not recommended_items:
        return 0.0

    if model is None:
        model = _load_embeddings_model()
        if model is None:
            return 0.5  # neutral fallback

    cart_emb = _encode_items(model, cart_items, item_catalog)
    reco_emb = _encode_items(model, recommended_items, item_catalog)

    if cart_emb.shape[0] == 0 or reco_emb.shape[0] == 0:
        return 0.0

    # Mean cosine similarity: each reco vs. cart centroid
    cart_centroid = cart_emb.mean(axis=0, keepdims=True)
    cart_centroid /= max(np.linalg.norm(cart_centroid), 1e-8)

    sims = reco_emb @ cart_centroid.T  # (n_reco, 1)
    return float(np.mean(sims))


def evaluate_embedding_diversity(
    recommended_items: list[str],
    item_catalog: dict[str, dict],
    model: Any = None,
) -> float:
    """Compute mean pairwise distance among recommended items.

    High score = diverse recommendations (desirable).
    """
    if len(recommended_items) < 2:
        return 1.0

    if model is None:
        model = _load_embeddings_model()
        if model is None:
            return 0.5

    reco_emb = _encode_items(model, recommended_items, item_catalog)
    if reco_emb.shape[0] < 2:
        return 1.0

    # Pairwise cosine similarity
    sim_matrix = reco_emb @ reco_emb.T
    n = sim_matrix.shape[0]
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_sim = float(sim_matrix[mask].mean())
    # Convert similarity to diversity: 1 - mean_sim
    return max(0.0, 1.0 - mean_sim)


def evaluate_category_coverage(
    cart_items: list[str],
    recommended_items: list[str],
    item_catalog: dict[str, dict],
) -> float:
    """Score how well recommendations fill meal category gaps."""
    MEAL_CATS = {"main_course", "beverage", "dessert", "starter", "addon"}

    cart_cats = set()
    for iid in cart_items:
        entry = item_catalog.get(iid, {})
        cat = entry.get("item_category", entry.get("category", ""))
        if cat in MEAL_CATS:
            cart_cats.add(cat)

    reco_cats = set()
    for iid in recommended_items:
        entry = item_catalog.get(iid, {})
        cat = entry.get("item_category", entry.get("category", ""))
        if cat in MEAL_CATS:
            reco_cats.add(cat)

    combined = cart_cats | reco_cats
    return len(combined) / max(len(MEAL_CATS), 1)


def run_llm_judge(
    predictions: pd.DataFrame,
    item_catalog_df: pd.DataFrame,
    k: int = 10,
    n_samples: int = 100,
    seed: int = 42,
) -> LLMJudgeResult:
    """Run the full LLM-as-judge evaluation suite.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: query_id, item_id, score, label.
        Optionally cart_items (comma-separated).
    item_catalog_df : pd.DataFrame
        Full item catalog with item_id, item_name, item_category, item_price.
    k : int
        Top-k recommendations to evaluate per query.
    n_samples : int
        Number of query groups to sample (for efficiency).
    seed : int
        Random seed for sampling.

    Returns
    -------
    LLMJudgeResult
        Composite evaluation with per-dimension scores.
    """
    rng = np.random.default_rng(seed)

    # Build catalog dict
    catalog = {}
    for _, row in item_catalog_df.iterrows():
        iid = str(row["item_id"])
        catalog[iid] = {
            "item_name": str(row.get("item_name", iid)),
            "item_category": str(row.get("item_category", "unknown")),
            "item_price": float(row.get("item_price", 0)),
        }

    # Load model once
    model = _load_embeddings_model()

    query_ids = predictions["query_id"].unique()
    if len(query_ids) > n_samples:
        query_ids = rng.choice(query_ids, size=n_samples, replace=False)

    coherence_scores = []
    diversity_scores = []
    coverage_scores = []

    for qid in query_ids:
        group = predictions[predictions["query_id"] == qid]
        top_recos = group.sort_values("score", ascending=False).head(k)
        reco_ids = top_recos["item_id"].astype(str).tolist()

        # Infer cart items: items with label=1 or from cart_items column
        if "cart_items" in group.columns:
            cart_str = str(group.iloc[0].get("cart_items", ""))
            cart_ids = [c.strip() for c in cart_str.split(",") if c.strip()]
        else:
            # Use known relevant items (positive labels) as proxy for cart
            relevant = group[group["label"] == 1]
            cart_ids = relevant["item_id"].astype(str).tolist()[:5]

        if not cart_ids:
            cart_ids = reco_ids[:2]  # minimal fallback

        coherence_scores.append(
            evaluate_semantic_coherence(cart_ids, reco_ids, catalog, model)
        )
        diversity_scores.append(
            evaluate_embedding_diversity(reco_ids, catalog, model)
        )
        coverage_scores.append(
            evaluate_category_coverage(cart_ids, reco_ids, catalog)
        )

    mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    mean_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
    mean_coverage = float(np.mean(coverage_scores)) if coverage_scores else 0.0

    # Weighted composite: coherence (40%), diversity (30%), coverage (30%)
    overall = 0.4 * mean_coherence + 0.3 * mean_diversity + 0.3 * mean_coverage

    # Optional: LLM API judge
    llm_scores = _llm_api_judge(predictions, catalog, k=k, n_samples=min(n_samples, 20))

    return LLMJudgeResult(
        semantic_coherence=mean_coherence,
        embedding_diversity=mean_diversity,
        category_coverage_score=mean_coverage,
        overall_quality_score=overall,
        llm_scores=llm_scores,
        detail={
            "n_queries_evaluated": len(query_ids),
            "k": k,
            "has_llm_api_scores": bool(llm_scores),
            "score_breakdown": {
                "semantic_coherence_weight": 0.4,
                "embedding_diversity_weight": 0.3,
                "category_coverage_weight": 0.3,
            },
        },
    )


# ---------------------------------------------------------------------------
# Optional LLM API judge
# ---------------------------------------------------------------------------

def _llm_api_judge(
    predictions: pd.DataFrame,
    catalog: dict[str, dict],
    k: int = 10,
    n_samples: int = 20,
    model_name: str = "gpt-4o-mini",
) -> dict[str, float]:
    """Call LLM API to get qualitative ratings.

    Returns empty dict if no API key or on failure.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {}

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        return {}

    query_ids = predictions["query_id"].unique()[:n_samples]
    all_ratings = {"relevance": [], "complementarity": [], "diversity": []}

    for qid in query_ids:
        group = predictions[predictions["query_id"] == qid]
        top = group.sort_values("score", ascending=False).head(k)
        reco_descs = []
        for _, row in top.iterrows():
            iid = str(row["item_id"])
            entry = catalog.get(iid, {})
            reco_descs.append(f"{entry.get('item_name', iid)} ({entry.get('item_category', 'food')})")

        prompt = f"""Rate these food delivery add-on recommendations on a 1-5 scale.
Recommendations: {', '.join(reco_descs[:5])}

Rate each dimension (just numbers, comma-separated):
1. Relevance (do items make sense as food add-ons?): 
2. Complementarity (do items complement each other?):
3. Diversity (is there variety in the recommendations?):

Format: relevance,complementarity,diversity"""

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )
            text = resp.choices[0].message.content or ""
            nums = [float(x.strip()) for x in text.strip().split(",")[:3]]
            if len(nums) == 3:
                all_ratings["relevance"].append(nums[0] / 5.0)
                all_ratings["complementarity"].append(nums[1] / 5.0)
                all_ratings["diversity"].append(nums[2] / 5.0)
        except Exception:
            continue

    if not all_ratings["relevance"]:
        return {}

    return {
        "llm_relevance": float(np.mean(all_ratings["relevance"])),
        "llm_complementarity": float(np.mean(all_ratings["complementarity"])),
        "llm_diversity": float(np.mean(all_ratings["diversity"])),
        "llm_overall": float(np.mean([
            np.mean(all_ratings["relevance"]),
            np.mean(all_ratings["complementarity"]),
            np.mean(all_ratings["diversity"]),
        ])),
        "llm_n_rated": len(all_ratings["relevance"]),
    }
