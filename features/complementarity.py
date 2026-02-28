from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


def compute_item_complementarity(order_items: pd.DataFrame, min_support: int = 2) -> pd.DataFrame:
    order_to_items = (
        order_items.sort_values(["order_id", "position"])
        .groupby("order_id")["item_id"]
        .apply(lambda s: list(dict.fromkeys(s.astype(str).tolist())))
    )

    item_counts: defaultdict[str, int] = defaultdict(int)
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for basket in order_to_items:
        unique_items = list(set(basket))
        for item in unique_items:
            item_counts[item] += 1
        for a, b in combinations(sorted(unique_items), 2):
            pair_counts[(a, b)] += 1

    n_orders = max(len(order_to_items), 1)
    rows = []
    eps = 1e-9
    for (a, b), co in pair_counts.items():
        if co < min_support:
            continue
        p_ab = co / n_orders
        p_a = item_counts[a] / n_orders
        p_b = item_counts[b] / n_orders
        lift = p_ab / max(p_a * p_b, eps)
        pmi = float(np.log((p_ab + eps) / (p_a * p_b + eps)))

        rows.append({"item_id": a, "candidate_item_id": b, "cooccurrence": co, "lift": lift, "pmi": pmi})
        rows.append({"item_id": b, "candidate_item_id": a, "cooccurrence": co, "lift": lift, "pmi": pmi})

    if not rows:
        return pd.DataFrame(columns=["item_id", "candidate_item_id", "cooccurrence", "lift", "pmi"])
    return pd.DataFrame(rows).sort_values(["item_id", "lift", "pmi"], ascending=[True, False, False]).reset_index(drop=True)


def compute_category_affinity(
    order_items: pd.DataFrame,
    items: pd.DataFrame,
    min_support: int = 2,
) -> pd.DataFrame:
    item_to_cat = items.set_index("item_id")["item_category"].astype(str).to_dict()
    order_to_cats = order_items.groupby("order_id")["item_id"].apply(
        lambda s: list({item_to_cat.get(str(i), "unknown") for i in s.astype(str).tolist()})
    )

    cat_counts: defaultdict[str, int] = defaultdict(int)
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    n_orders = max(len(order_to_cats), 1)
    eps = 1e-9

    for cats in order_to_cats:
        for c in cats:
            cat_counts[c] += 1
        for a, b in combinations(sorted(cats), 2):
            pair_counts[(a, b)] += 1

    rows = []
    for (a, b), co in pair_counts.items():
        if co < min_support:
            continue
        p_ab = co / n_orders
        p_a = cat_counts[a] / n_orders
        p_b = cat_counts[b] / n_orders
        affinity = p_ab / max(p_a * p_b, eps)
        rows.append({"from_category": a, "to_category": b, "affinity": affinity})
        rows.append({"from_category": b, "to_category": a, "affinity": affinity})
    if not rows:
        return pd.DataFrame(columns=["from_category", "to_category", "affinity"])
    return pd.DataFrame(rows).sort_values(["from_category", "affinity"], ascending=[True, False]).reset_index(drop=True)


def build_complementarity_lookup(comp_df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in comp_df.itertuples(index=False):
        lookup[(str(row.item_id), str(row.candidate_item_id))] = (float(row.lift), float(row.pmi))
    return lookup

