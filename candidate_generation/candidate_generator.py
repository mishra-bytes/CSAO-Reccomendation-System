from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from candidate_generation.retrievers.category import CategoryComplementRetriever
from candidate_generation.retrievers.cooccurrence import CooccurrenceRetriever
from candidate_generation.retrievers.popularity import PopularityRetriever
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
        self.co_k = int(self.cfg.get("cooccurrence_k", 120))
        self.pop_k = int(self.cfg.get("popularity_k", 80))
        self.cat_k = int(self.cfg.get("category_k", 60))

        self.co_retriever = CooccurrenceRetriever(complementarity)
        self.pop_retriever = PopularityRetriever(orders, order_items)
        self.cat_retriever = CategoryComplementRetriever(category_affinity, items, orders, order_items)

    def generate(self, cart_items: list[str], restaurant_id: str, top_k: int | None = None) -> list[tuple[str, float]]:
        target_k = int(top_k or self.total_k)
        exclude = set(str(i) for i in cart_items)

        co = self.co_retriever.retrieve(cart_items, k=self.co_k)
        pop = self.pop_retriever.retrieve(restaurant_id=restaurant_id, exclude=exclude, k=self.pop_k)
        cat = self.cat_retriever.retrieve(cart_items=cart_items, restaurant_id=restaurant_id, exclude=exclude, k=self.cat_k)

        aggregate: dict[str, float] = defaultdict(float)
        for item, score in co:
            if item not in exclude:
                aggregate[item] += 0.55 * score
        for item, score in cat:
            if item not in exclude:
                aggregate[item] += 0.35 * score
        for item, score in pop:
            if item not in exclude:
                aggregate[item] += 0.10 * score

        ranked = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < target_k:
            ranked = fill_candidates(
                ranked_candidates=ranked,
                fallback_candidates=pop,
                exclude=exclude,
                target_k=target_k,
            )
        return ranked[:target_k]

