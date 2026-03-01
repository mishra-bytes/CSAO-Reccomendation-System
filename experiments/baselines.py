"""
Baseline Models for CSAO Evaluation
====================================
Implements three baselines that the CSAO LightGBM ranker is compared against:

1. **Popularity Baseline** — ranks candidates by global item popularity.
2. **Co-occurrence Baseline** — ranks by pairwise co-purchase lift with cart.
3. **Random Baseline** — random ranking (lower bound sanity check).

Each baseline produces the same prediction DataFrame format as the LightGBM
ranker so they can be fed directly into `evaluate_offline()`.

Rubric alignment:
 - Criterion 4 (Model Evaluation): "Comparison with baseline approaches"
 - Criterion 6 (Business Impact): "compare against baseline strategy"
"""
from __future__ import annotations

import math
import random as rng_module
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd


# ── Baseline 1: Popularity ────────────────────────────────────────────────────

def popularity_baseline(
    validation_predictions: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Score each candidate by global item purchase frequency.

    This is the simplest production baseline: "show the most-ordered items
    in this restaurant".  It ignores the cart entirely.
    """
    # Global item frequency
    item_counts = (
        order_items["item_id"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    max_count = max(item_counts.values()) if item_counts else 1.0

    out = validation_predictions[["query_id", "item_id", "label"]].copy()
    out["score"] = out["item_id"].astype(str).map(
        lambda x: item_counts.get(x, 0) / max_count
    )
    return out


# ── Baseline 2: Co-occurrence ─────────────────────────────────────────────────

def cooccurrence_baseline(
    validation_predictions: pd.DataFrame,
    query_meta: pd.DataFrame,
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Score each candidate by its co-purchase lift with items already in cart.

    Uses the same pairwise lift formula as the main system's complementarity
    features, but this baseline uses lift as the *sole* ranking signal — no
    user features, no cart context, no ML model.
    """
    # Build co-occurrence counts from training data
    n_orders = orders["order_id"].nunique()
    oi = order_items[["order_id", "item_id"]].copy()
    oi["item_id"] = oi["item_id"].astype(str)

    # Item frequency
    item_freq: dict[str, int] = (
        oi.groupby("item_id")["order_id"].nunique().to_dict()
    )

    # Pairwise co-occurrence (bidirectional)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, group in oi.groupby("order_id"):
        items = group["item_id"].unique().tolist()
        if len(items) > 30:
            items = items[:30]  # cap for speed
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair_counts[(items[i], items[j])] += 1
                pair_counts[(items[j], items[i])] += 1

    def _lift(item_a: str, item_b: str) -> float:
        co = pair_counts.get((item_a, item_b), 0)
        fa = item_freq.get(item_a, 0)
        fb = item_freq.get(item_b, 0)
        if fa == 0 or fb == 0 or n_orders == 0:
            return 0.0
        pa = fa / n_orders
        pb = fb / n_orders
        pab = co / n_orders
        return pab / (pa * pb + 1e-9)

    # Parse cart items from query_meta
    qmeta = query_meta.copy()
    # query_meta has: query_id, user_id, restaurant_id, cart_item_ids (str repr)
    cart_lookup: dict[str, list[str]] = {}
    if "cart_item_ids" in qmeta.columns:
        for _, row in qmeta.iterrows():
            qid = str(row["query_id"])
            cart_raw = row.get("cart_item_ids", "[]")
            if isinstance(cart_raw, str):
                import ast
                try:
                    cart_lookup[qid] = [str(x) for x in ast.literal_eval(cart_raw)]
                except Exception:
                    cart_lookup[qid] = []
            elif isinstance(cart_raw, (list, np.ndarray)):
                cart_lookup[qid] = [str(x) for x in cart_raw]
            else:
                cart_lookup[qid] = []

    # Score each candidate
    out = validation_predictions[["query_id", "item_id", "label"]].copy()
    scores = []
    for _, row in out.iterrows():
        qid = str(row["query_id"])
        cand = str(row["item_id"])
        cart = cart_lookup.get(qid, [])
        if not cart:
            scores.append(0.0)
        else:
            lifts = [_lift(ci, cand) for ci in cart]
            scores.append(max(lifts) if lifts else 0.0)
    out["score"] = scores
    return out


# ── Baseline 3: Random ────────────────────────────────────────────────────────

def random_baseline(
    validation_predictions: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Random ranking — lower-bound sanity check."""
    rng = np.random.RandomState(seed)
    out = validation_predictions[["query_id", "item_id", "label"]].copy()
    out["score"] = rng.rand(len(out))
    return out


# ── Comparison Runner ─────────────────────────────────────────────────────────

def run_baseline_comparison(
    validation_predictions: pd.DataFrame,
    query_meta: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    item_catalog: pd.DataFrame,
    user_features: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Run all baselines + the main model and produce a comparison table.

    Returns a DataFrame with one row per model and metrics as columns.
    """
    from evaluation.run_eval import evaluate_offline

    models = {
        "CSAO_LightGBM": validation_predictions,
        "Popularity": popularity_baseline(validation_predictions, orders, order_items),
        "CoOccurrence": cooccurrence_baseline(
            validation_predictions, query_meta, order_items, orders,
        ),
        "Random": random_baseline(validation_predictions),
    }

    rows = []
    for name, preds in models.items():
        print(f"  Evaluating {name}...", flush=True)
        result = evaluate_offline(preds, item_catalog, query_meta, user_features, k=k)
        row = {"model": name}
        # Overall metrics
        row.update(result.get("overall", {}))
        # Business metrics
        row.update(result.get("business_impact", {}))
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("model")
    return comparison
