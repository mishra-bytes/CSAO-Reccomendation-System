from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

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

        # Build restaurant → menu-item-set index for the restaurant-menu gate.
        # Only items actually served at a restaurant pass the candidate filter,
        # preventing cross-cuisine contamination from co-occurrence / session signals.
        merged_menu = order_items.merge(
            orders[["order_id", "restaurant_id"]], on="order_id", how="left",
        )
        self._restaurant_menu: dict[str, set[str]] = (
            merged_menu.groupby("restaurant_id")["item_id"]
            .apply(lambda s: set(s.astype(str).tolist()))
            .to_dict()
        )

        # Build item → category lookup for course-type filtering
        self._item_category: dict[str, str] = (
            items.set_index("item_id")["item_category"].astype(str).to_dict()
            if "item_category" in items.columns else {}
        )

        # Build item → is_veg lookup for dietary preference filtering
        self._item_is_veg: dict[str, bool] = (
            items.set_index("item_id")["is_veg"].astype(bool).to_dict()
            if "is_veg" in items.columns else {}
        )

        # Retriever weights — tuned for CSAO where cart-aware signals dominate
        # Defaults match configs/base.yaml; override via config dict.
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

        # Restaurant-menu gate: only items actually on this restaurant's menu
        # can be recommended.  Prevents cross-cuisine contamination.
        menu_items = self._restaurant_menu.get(str(restaurant_id), set())

        # ── Run all five retrievers in parallel (I/O-bound lookups) ────
        def _co():
            return self.co_retriever.retrieve(cart_items, k=self.co_k)
        def _pop():
            return self.pop_retriever.retrieve(restaurant_id=restaurant_id, exclude=exclude, k=self.pop_k)
        def _cat():
            return self.cat_retriever.retrieve(cart_items=cart_items, restaurant_id=restaurant_id, exclude=exclude, k=self.cat_k)
        def _session():
            return self.session_retriever.retrieve(cart_items=cart_items, exclude=exclude, k=self.session_k)
        def _meal_gap():
            return self.meal_gap_retriever.retrieve(cart_items=cart_items, restaurant_id=restaurant_id, exclude=exclude, k=self.meal_gap_k)

        with ThreadPoolExecutor(max_workers=5) as pool:
            f_co = pool.submit(_co)
            f_pop = pool.submit(_pop)
            f_cat = pool.submit(_cat)
            f_session = pool.submit(_session)
            f_meal_gap = pool.submit(_meal_gap)
        co = f_co.result()
        pop = f_pop.result()
        cat = f_cat.result()
        session = f_session.result()
        meal_gap = f_meal_gap.result()

        # Determine cart course types for course-type aware filtering:
        # penalise recommending another main_course when cart already has one.
        cart_cats = set()
        for ci in cart_items:
            cat_val = self._item_category.get(str(ci), "unknown")
            if cat_val != "unknown":
                cart_cats.add(cat_val)
        has_main = "main_course" in cart_cats

        # Veg/non-veg preference: if ALL cart items are veg, filter out non-veg
        # candidates (models the stricty-vegetarian user pattern common in India)
        cart_all_veg = all(
            self._item_is_veg.get(str(ci), True) for ci in cart_items
        ) if cart_items else False

        w = self._weights
        aggregate: dict[str, float] = defaultdict(float)
        for item, score in co:
            if item not in exclude:
                aggregate[item] += w["cooccurrence"] * score
        for item, score in session:
            if item not in exclude:
                aggregate[item] += w["session"] * score
        for item, score in meal_gap:
            if item not in exclude:
                aggregate[item] += w["meal_gap"] * score
        for item, score in cat:
            if item not in exclude:
                aggregate[item] += w["category"] * score
        for item, score in pop:
            if item not in exclude:
                aggregate[item] += w["popularity"] * score

        # ── Restaurant-menu gate + course-type penalty + veg filter ──────
        filtered: dict[str, float] = {}
        for item, score in aggregate.items():
            # Gate: candidate must exist on the restaurant's menu
            if menu_items and item not in menu_items:
                continue
            # Veg-preference gate: if cart is all-veg, strongly penalise non-veg
            if cart_all_veg and not self._item_is_veg.get(str(item), True):
                score *= 0.1  # 90% penalty — ranker can override if signal is very strong
            # Penalty: if cart already has a main_course, deprioritise other mains
            if has_main and self._item_category.get(str(item), "") == "main_course":
                score *= 0.3  # soft penalty — ranker can still surface if strong signal
            filtered[item] = score

        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < target_k:
            ranked = fill_candidates(
                ranked_candidates=ranked,
                fallback_candidates=pop,
                exclude=exclude,
                target_k=target_k,
            )
        return ranked[:target_k]

