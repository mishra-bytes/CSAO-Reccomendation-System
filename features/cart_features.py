from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


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
        }

    lookup = item_lookup.set_index("item_id", drop=False)
    cart_items = lookup.reindex(cart_item_ids).fillna({"item_price": 0.0, "item_category": "unknown"})

    cart_size = float(len(cart_item_ids))
    cart_value = float(cart_items["item_price"].fillna(0.0).sum())
    last_added_item = str(cart_item_ids[-1])
    session_position = float(len(cart_item_ids))

    category_counts = Counter(cart_items["item_category"].astype(str).tolist())
    total = max(sum(category_counts.values()), 1)
    top_categories = [cat for cat, _ in category_counts.most_common(max_categories)]

    features: dict[str, Any] = {
        "cart_size": cart_size,
        "cart_value": cart_value,
        "session_position": session_position,
        "last_added_item": last_added_item,
    }
    for cat in top_categories:
        features[f"cart_cat_share__{cat}"] = category_counts[cat] / total
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

