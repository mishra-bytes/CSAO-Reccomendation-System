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
    attach = 0
    total = 0
    for _, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k)
        total += 1
        if top["label"].sum() > 0:
            attach += 1
    return attach / max(total, 1)


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
    incremental_values = []
    for _, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k)
        relevant = top[top["label"] == 1]
        if relevant.empty:
            continue
        # Assume user adds the top recommended relevant item
        best_item = str(relevant.iloc[0]["item_id"])
        price = item_prices.get(best_item, 0.0)
        incremental_values.append(price * take_rate)

    if not incremental_values:
        return 0.0
    return float(np.mean(incremental_values))


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
    all_recommended = []
    for _, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k)
        all_recommended.extend(top["item_id"].astype(str).tolist())

    if not all_recommended:
        return 0.0
    unique_ratio = len(set(all_recommended)) / max(len(all_recommended), 1)
    return 1.0 - unique_ratio  # closer to 0 = less fatigue


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

    # Average price of recommended items (for sanity check)
    all_reco_prices = []
    for _, group in predictions.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k)
        for iid in top["item_id"].astype(str):
            p = item_prices.get(iid, 0.0)
            if p > 0:
                all_reco_prices.append(p)
    avg_price = float(np.mean(all_reco_prices)) if all_reco_prices else 0.0

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
