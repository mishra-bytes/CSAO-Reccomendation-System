from __future__ import annotations

from collections import defaultdict

import pandas as pd


class PopularityRetriever:
    def __init__(self, orders: pd.DataFrame, order_items: pd.DataFrame):
        self.global_pop = (
            order_items.groupby("item_id")["order_id"].nunique().sort_values(ascending=False).index.astype(str).tolist()
        )
        merged = order_items.merge(orders[["order_id", "restaurant_id"]], on="order_id", how="left")
        grouped = (
            merged.groupby(["restaurant_id", "item_id"])["order_id"]
            .nunique()
            .rename("count")
            .reset_index()
            .sort_values(["restaurant_id", "count"], ascending=[True, False])
        )

        self.by_restaurant: dict[str, list[str]] = defaultdict(list)
        for rest_id, grp in grouped.groupby("restaurant_id"):
            self.by_restaurant[str(rest_id)] = grp["item_id"].astype(str).tolist()

    def retrieve(self, restaurant_id: str, exclude: set[str], k: int = 80) -> list[tuple[str, float]]:
        ranked = []
        pool = self.by_restaurant.get(str(restaurant_id), self.global_pop)
        for idx, item in enumerate(pool):
            if item in exclude:
                continue
            ranked.append((item, 1.0 / (idx + 1)))
            if len(ranked) >= k:
                break
        return ranked

