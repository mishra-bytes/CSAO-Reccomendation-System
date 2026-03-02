"""Business impact evaluation metrics for CSAO recommendations.

Maps recommendation quality signals to proxy business KPIs:
- Attach rate: fraction of recommendation lists where at least one item is relevant
- Incremental AOV: estimated additional order value from successful recommendations
- Revenue uplift: simulated revenue impact at various take rates
- Recommendation fatigue: diversity safety-check against repetitive suggestions

These are *proxies* computed from offline evaluation data to demonstrate
business credibility. In production they'd be A/B tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BusinessMetrics:
    attach_rate: float
    incremental_aov: float
    revenue_uplift_per_1k_orders: float
    avg_reco_price: float
    fatigue_score: float
    detail: dict[str, Any]


def compute_attach_rate(predictions: pd.DataFrame, k: int = 10) -> float:
    """Fraction of query groups where at least one recommended item is relevant (label=1).

    This proxy maps directly to the real-world attach rate: "of all users
    who saw recommendations, what % added at least one to cart?"
    """
    # Convert to native types to avoid slow arrow-backed groupby
    df = predictions[["query_id", "score", "label"]].copy()
    df["query_id"] = df["query_id"].astype(str)
    df["score"] = df["score"].astype(float)
    df["label"] = df["label"].astype(int)

    # Rank within each group and keep top-k
    df["rank"] = df.groupby("query_id")["score"].rank(ascending=False, method="first")
    top_k = df[df["rank"] <= k]

    # Check if any relevant item exists in top-k per query
    attach_per_query = top_k.groupby("query_id")["label"].max()
    return float((attach_per_query > 0).mean())


def compute_incremental_aov(
    predictions: pd.DataFrame,
    item_prices: dict[str, float],
    k: int = 10,
    take_rate: float = 0.15,
) -> float:
    """Estimated additional order value from successful recommendations.

    Model: for each query where at least one relevant item is in top-k,
    the user adds the highest-ranked relevant item. Multiply its price
    by the take_rate (probability the user actually acts on the
    recommendation in production — typically 10-20% from industry
    benchmarks).
    """
    df = predictions[["query_id", "item_id", "score", "label"]].copy()
    df["query_id"] = df["query_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["score"] = df["score"].astype(float)
    df["label"] = df["label"].astype(int)
    df["rank"] = df.groupby("query_id")["score"].rank(ascending=False, method="first")
    top_k = df[df["rank"] <= k].copy()

    # Among relevant items in top-k, get the highest-scored one per query
    relevant = top_k[top_k["label"] == 1].copy()
    if relevant.empty:
        return 0.0
    best = relevant.sort_values("score", ascending=False).drop_duplicates("query_id")
    best["price"] = best["item_id"].map(item_prices).fillna(0.0)
    return float((best["price"] * take_rate).mean())


def compute_revenue_uplift(
    predictions: pd.DataFrame,
    item_prices: dict[str, float],
    k: int = 10,
    take_rate: float = 0.15,
    orders_per_1k: int = 1000,
) -> float:
    """Simulated revenue uplift per 1k orders.

    = attach_rate * avg_incremental_value * orders_per_1k
    """
    attach = compute_attach_rate(predictions, k=k)
    inc_aov = compute_incremental_aov(predictions, item_prices, k=k, take_rate=take_rate)
    return attach * inc_aov * orders_per_1k


def compute_fatigue_score(predictions: pd.DataFrame, k: int = 10) -> float:
    """Measures recommendation repetitiveness across queries.

    Lower is better. Computed as: 1 - (unique items in top-k / total
    recommended items). A score near 1.0 means the system recommends
    the same items to everyone (fatigue risk). Near 0.0 means high
    diversity.
    """
    df = predictions[["query_id", "item_id", "score"]].copy()
    df["query_id"] = df["query_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["score"] = df["score"].astype(float)
    df["rank"] = df.groupby("query_id")["score"].rank(ascending=False, method="first")
    top_k = df[df["rank"] <= k]

    if top_k.empty:
        return 0.0
    total = len(top_k)
    unique = top_k["item_id"].nunique()
    return 1.0 - (unique / max(total, 1))  # closer to 0 = less fatigue


def evaluate_business_impact(
    predictions: pd.DataFrame,
    item_catalog: pd.DataFrame,
    k: int = 10,
    take_rate: float = 0.15,
) -> BusinessMetrics:
    """Run the full business impact evaluation suite."""
    item_prices = (
        item_catalog.drop_duplicates("item_id")
        .set_index("item_id")["item_price"]
        .astype(float)
        .to_dict()
    )

    attach = compute_attach_rate(predictions, k=k)
    inc_aov = compute_incremental_aov(predictions, item_prices, k=k, take_rate=take_rate)
    rev_uplift = compute_revenue_uplift(predictions, item_prices, k=k, take_rate=take_rate)
    fatigue = compute_fatigue_score(predictions, k=k)

    # Average price of recommended items (vectorized)
    df = predictions[["query_id", "item_id", "score"]].copy()
    df["query_id"] = df["query_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["score"] = df["score"].astype(float)
    df["rank"] = df.groupby("query_id")["score"].rank(ascending=False, method="first")
    top_k = df[df["rank"] <= k]
    reco_prices = top_k["item_id"].map(item_prices).dropna()
    reco_prices = reco_prices[reco_prices > 0]
    avg_price = float(reco_prices.mean()) if len(reco_prices) > 0 else 0.0

    detail = {
        "take_rate_assumed": take_rate,
        "k": k,
        "n_queries": predictions["query_id"].nunique(),
        "n_unique_items_recommended": len(set(predictions["item_id"].astype(str))),
        "fatigue_interpretation": (
            "LOW risk" if fatigue < 0.5
            else "MODERATE risk" if fatigue < 0.8
            else "HIGH risk — consider diversification"
        ),
    }

    return BusinessMetrics(
        attach_rate=attach,
        incremental_aov=inc_aov,
        revenue_uplift_per_1k_orders=rev_uplift,
        avg_reco_price=avg_price,
        fatigue_score=fatigue,
        detail=detail,
    )
