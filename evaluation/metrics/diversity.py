from __future__ import annotations

from itertools import combinations

import pandas as pd


def intra_list_diversity_at_k(
    ranked_items: pd.DataFrame,
    item_catalog: pd.DataFrame,
    k: int = 10,
) -> float:
    item_to_cat = item_catalog.set_index("item_id")["item_category"].astype(str).to_dict()
    vals = []
    for _, group in ranked_items.groupby("query_id"):
        top = group.sort_values("score", ascending=False).head(k)["item_id"].astype(str).tolist()
        if len(top) < 2:
            continue
        mismatch = 0
        total = 0
        for a, b in combinations(top, 2):
            total += 1
            mismatch += 1 if item_to_cat.get(a, "unknown") != item_to_cat.get(b, "unknown") else 0
        vals.append(mismatch / max(total, 1))
    return float(sum(vals) / max(len(vals), 1))

