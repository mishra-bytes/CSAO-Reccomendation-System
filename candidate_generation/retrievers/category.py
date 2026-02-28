from __future__ import annotations

from collections import defaultdict

import pandas as pd


class CategoryComplementRetriever:
    def __init__(self, category_affinity: pd.DataFrame, items: pd.DataFrame, orders: pd.DataFrame, order_items: pd.DataFrame):
        item_map_series = items.drop_duplicates("item_id").set_index("item_id")["item_category"]
        self.item_to_category = item_map_series.astype(str).to_dict()
        self.category_map: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in category_affinity.itertuples(index=False):
            self.category_map[str(row.from_category)].append((str(row.to_category), float(row.affinity)))
        for cat in list(self.category_map.keys()):
            self.category_map[cat] = sorted(self.category_map[cat], key=lambda x: x[1], reverse=True)

        merged = order_items.merge(orders[["order_id", "restaurant_id"]], on="order_id", how="left")
        if "item_category" not in merged.columns:
            merged = merged.merge(items[["item_id", "item_category"]], on="item_id", how="left")
        else:
            merged["item_category"] = merged["item_category"].fillna(merged["item_id"].map(self.item_to_category))
        merged["item_category"] = merged["item_category"].fillna("unknown").astype(str)
        pop = (
            merged.groupby(["restaurant_id", "item_category", "item_id"])["order_id"]
            .nunique()
            .rename("cnt")
            .reset_index()
            .sort_values(["restaurant_id", "item_category", "cnt"], ascending=[True, True, False])
        )

        self.restaurant_category_items: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (rest, cat), grp in pop.groupby(["restaurant_id", "item_category"]):
            self.restaurant_category_items[(str(rest), str(cat))] = grp["item_id"].astype(str).tolist()

    def retrieve(self, cart_items: list[str], restaurant_id: str, exclude: set[str], k: int = 60) -> list[tuple[str, float]]:
        candidate_scores: dict[str, float] = defaultdict(float)
        for item_id in cart_items:
            cat = self.item_to_category.get(str(item_id), "unknown")
            for to_cat, affinity in self.category_map.get(cat, [])[:3]:
                for idx, cand in enumerate(self.restaurant_category_items.get((str(restaurant_id), to_cat), [])[:40]):
                    if cand in exclude:
                        continue
                    candidate_scores[cand] += affinity / (idx + 1)
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
