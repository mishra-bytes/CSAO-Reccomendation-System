from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from common.feature_names import normalize_feature_name


# ---------------------------------------------------------------------------
# Canonical meal archetypes — models typical Zomato/food-delivery meal patterns
# Used for cart completeness scoring and missing-category detection.
# ---------------------------------------------------------------------------
MEAL_ARCHETYPES: list[set[str]] = [
    {"main_course", "beverage"},
    {"main_course", "beverage", "dessert"},
    {"main_course", "starter", "beverage"},
    {"main_course", "starter", "dessert", "beverage"},
    {"main_course", "addon"},
    {"starter", "main_course"},
]


def _cart_completeness_score(category_set: set[str]) -> float:
    """How close the current cart is to completing a canonical meal.

    Returns the best (highest) Jaccard overlap with any meal archetype.
    Range: [0.0, 1.0] — 1.0 means the cart exactly matches a template.
    """
    if not category_set:
        return 0.0
    best = 0.0
    for archetype in MEAL_ARCHETYPES:
        intersection = len(category_set & archetype)
        union = len(category_set | archetype)
        if union > 0:
            best = max(best, intersection / union)
    return best


def _missing_categories(category_set: set[str]) -> tuple[int, float]:
    """Detect categories missing from the closest meal archetype.

    Returns:
        n_missing: number of categories still needed for the best-matching archetype
        missing_ratio: n_missing / archetype_size
    """
    if not category_set:
        return 0, 0.0
    best_archetype = None
    best_overlap = -1
    for arch in MEAL_ARCHETYPES:
        overlap = len(category_set & arch)
        if overlap > best_overlap:
            best_overlap = overlap
            best_archetype = arch
    if best_archetype is None:
        return 0, 0.0
    missing = best_archetype - category_set
    return len(missing), len(missing) / max(len(best_archetype), 1)


def build_cart_feature_vector(
    cart_item_ids: list[str],
    item_lookup: pd.DataFrame,
    max_categories: int = 12,
) -> dict[str, float]:
    if len(cart_item_ids) == 0:
        return {
            "cart_size": 0.0,
            "cart_value": 0.0,
            "session_position": 0.0,
            "last_added_item": "none",
            "cart_completeness": 0.0,
            "cart_missing_cats": 0.0,
            "cart_missing_cat_ratio": 0.0,
            "cart_unique_categories": 0.0,
            "cart_avg_price": 0.0,
            "cart_price_std": 0.0,
            "cart_has_main": 0.0,
            "cart_has_beverage": 0.0,
            "cart_has_dessert": 0.0,
            "cart_has_starter": 0.0,
        }

    lookup = item_lookup.set_index("item_id", drop=False)
    cart_items = lookup.reindex(cart_item_ids).fillna({"item_price": 0.0, "item_category": "unknown"})

    cart_size = float(len(cart_item_ids))
    prices = cart_items["item_price"].fillna(0.0)
    cart_value = float(prices.sum())
    cart_avg_price = float(prices.mean()) if cart_size > 0 else 0.0
    cart_price_std = float(prices.std()) if cart_size > 1 else 0.0
    last_added_item = str(cart_item_ids[-1])
    session_position = float(len(cart_item_ids))

    category_list = cart_items["item_category"].astype(str).tolist()
    category_counts = Counter(category_list)
    category_set = set(category_list) - {"unknown"}
    total = max(sum(category_counts.values()), 1)
    top_categories = [cat for cat, _ in category_counts.most_common(max_categories)]

    # CSAO-specific intelligence
    completeness = _cart_completeness_score(category_set)
    n_missing, missing_ratio = _missing_categories(category_set)

    features: dict[str, Any] = {
        "cart_size": cart_size,
        "cart_value": cart_value,
        "cart_avg_price": cart_avg_price,
        "cart_price_std": cart_price_std,
        "session_position": session_position,
        "last_added_item": last_added_item,
        # Meal-completion features
        "cart_completeness": completeness,
        "cart_missing_cats": float(n_missing),
        "cart_missing_cat_ratio": missing_ratio,
        "cart_unique_categories": float(len(category_set)),
        # Binary meal-component indicators
        "cart_has_main": float("main_course" in category_set),
        "cart_has_beverage": float("beverage" in category_set),
        "cart_has_dessert": float("dessert" in category_set),
        "cart_has_starter": float("starter" in category_set),
    }
    for cat in top_categories:
        safe_cat = normalize_feature_name(cat)
        features[f"cart_cat_share__{safe_cat}"] = category_counts[cat] / total
    return features


def build_cart_context_table(order_items: pd.DataFrame, items: pd.DataFrame, max_categories: int = 12) -> pd.DataFrame:
    item_meta = items[["item_id", "item_category", "item_price"]].drop_duplicates("item_id")
    rows: list[dict[str, Any]] = []

    for order_id, group in order_items.groupby("order_id"):
        seq = group.sort_values("position")
        item_ids = seq["item_id"].astype(str).tolist()
        for idx in range(1, len(item_ids)):
            cart_prefix = item_ids[:idx]
            feat = build_cart_feature_vector(cart_prefix, item_meta, max_categories=max_categories)
            feat["order_id"] = order_id
            feat["target_item_id"] = item_ids[idx]
            feat["position"] = idx + 1
            rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=["order_id", "target_item_id", "position", "cart_size", "cart_value", "session_position"])

    out = pd.DataFrame(rows)
    out = out.fillna(0.0)
    return out


def cart_array_from_ids(cart_item_ids: list[str], items: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feat = build_cart_feature_vector(cart_item_ids, items)
    numeric_cols = [k for k, v in feat.items() if isinstance(v, (float, int, np.floating))]
    vec = np.array([float(feat[c]) for c in numeric_cols], dtype=float)
    return vec, numeric_cols
