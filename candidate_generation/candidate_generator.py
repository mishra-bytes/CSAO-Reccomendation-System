from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from features.meal_semantics import category_compatibility_multiplier

from candidate_generation.retrievers.category import CategoryComplementRetriever
from candidate_generation.retrievers.cooccurrence import CooccurrenceRetriever
from candidate_generation.retrievers.meal_gap import MealGapRetriever
from candidate_generation.retrievers.popularity import PopularityRetriever
from candidate_generation.retrievers.session_covisit import SessionCovisitRetriever
from candidate_generation.rules.fallback import fill_candidates


class CandidateGenerator:
    def __init__(
        self,
        complementarity: pd.DataFrame,
        category_affinity: pd.DataFrame,
        items: pd.DataFrame,
        orders: pd.DataFrame,
        order_items: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        self.cfg = config.get("candidate_generation", {})
        self.total_k = int(self.cfg.get("total_candidates", 200))
        self.co_k = int(self.cfg.get("cooccurrence_k", 100))
        self.pop_k = int(self.cfg.get("popularity_k", 60))
        self.cat_k = int(self.cfg.get("category_k", 50))
        self.session_k = int(self.cfg.get("session_covisit_k", 60))
        self.meal_gap_k = int(self.cfg.get("meal_gap_k", 40))

        self.co_retriever = CooccurrenceRetriever(complementarity)
        self.pop_retriever = PopularityRetriever(orders, order_items)
        self.cat_retriever = CategoryComplementRetriever(category_affinity, items, orders, order_items)
        self.session_retriever = SessionCovisitRetriever(orders, order_items)
        self.meal_gap_retriever = MealGapRetriever(items, orders, order_items)
        item_meta = items.drop_duplicates("item_id").set_index("item_id")
        self._item_name = item_meta.get("item_name", pd.Series(dtype=str)).astype(str).to_dict()
        self._item_category = item_meta.get("item_category", pd.Series(dtype=str)).astype(str).to_dict()

        # Retriever weights — tuned for CSAO where cart-aware signals dominate
        self._weights = {
            "cooccurrence": float(self.cfg.get("w_cooccurrence", 0.40)),
            "session": float(self.cfg.get("w_session", 0.20)),
            "meal_gap": float(self.cfg.get("w_meal_gap", 0.15)),
            "category": float(self.cfg.get("w_category", 0.15)),
            "popularity": float(self.cfg.get("w_popularity", 0.10)),
        }

    def generate(self, cart_items: list[str], restaurant_id: str, top_k: int | None = None) -> list[tuple[str, float]]:
        target_k = int(top_k or self.total_k)
        exclude = set(str(i) for i in cart_items)

        co = self.co_retriever.retrieve(cart_items, k=self.co_k)
        pop = self.pop_retriever.retrieve(restaurant_id=restaurant_id, exclude=exclude, k=self.pop_k)
        cat = self.cat_retriever.retrieve(cart_items=cart_items, restaurant_id=restaurant_id, exclude=exclude, k=self.cat_k)
        session = self.session_retriever.retrieve(cart_items=cart_items, exclude=exclude, k=self.session_k)
        meal_gap = self.meal_gap_retriever.retrieve(
            cart_items=cart_items, restaurant_id=restaurant_id, exclude=exclude, k=self.meal_gap_k,
        )

        w = self._weights
        aggregate: dict[str, float] = defaultdict(float)
        cart_names = [self._item_name.get(str(i), "") for i in cart_items]
        for item, score in co:
            if item not in exclude:
                aggregate[item] += category_compatibility_multiplier(cart_names, self._item_category.get(str(item), "unknown")) * w["cooccurrence"] * score
        for item, score in session:
            if item not in exclude:
                aggregate[item] += category_compatibility_multiplier(cart_names, self._item_category.get(str(item), "unknown")) * w["session"] * score
        for item, score in meal_gap:
            if item not in exclude:
                aggregate[item] += category_compatibility_multiplier(cart_names, self._item_category.get(str(item), "unknown")) * w["meal_gap"] * score
        for item, score in cat:
            if item not in exclude:
                aggregate[item] += category_compatibility_multiplier(cart_names, self._item_category.get(str(item), "unknown")) * w["category"] * score
        for item, score in pop:
            if item not in exclude:
                aggregate[item] += category_compatibility_multiplier(cart_names, self._item_category.get(str(item), "unknown")) * w["popularity"] * score

        ranked = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < target_k:
            ranked = fill_candidates(
                ranked_candidates=ranked,
                fallback_candidates=pop,
                exclude=exclude,
                target_k=target_k,
            )
        return ranked[:target_k]

