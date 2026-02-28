"""Meal-gap retriever.

Detects which food categories are missing from the cart relative to
common meal archetypes, then retrieves popular items from those
missing categories at the same restaurant.

This is CSAO-native logic: it explicitly models "what's missing from
the meal" rather than relying on generic co-occurrence.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from features.cart_features import MEAL_ARCHETYPES


class MealGapRetriever:
    """Retrieves candidates that fill gaps in the customer's meal.

    Index structure:
        (restaurant_id, category) → [item_id, ...] sorted by popularity
    """

    def __init__(
        self,
        items: pd.DataFrame,
        orders: pd.DataFrame,
        order_items: pd.DataFrame,
    ) -> None:
        self.item_to_category: dict[str, str] = (
            items.drop_duplicates("item_id")
            .set_index("item_id")["item_category"]
            .astype(str)
            .to_dict()
        )

        # Build (restaurant, category) → [items by popularity]
        merged = order_items.merge(orders[["order_id", "restaurant_id"]], on="order_id", how="left")
        if "item_category" not in merged.columns:
            merged = merged.merge(
                items[["item_id", "item_category"]].drop_duplicates("item_id"),
                on="item_id",
                how="left",
            )
        merged["item_category"] = merged["item_category"].fillna("unknown").astype(str)

        pop = (
            merged.groupby(["restaurant_id", "item_category", "item_id"])["order_id"]
            .nunique()
            .rename("cnt")
            .reset_index()
            .sort_values(["restaurant_id", "item_category", "cnt"], ascending=[True, True, False])
        )

        self.restaurant_cat_items: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (rest, cat), grp in pop.groupby(["restaurant_id", "item_category"]):
            self.restaurant_cat_items[(str(rest), str(cat))] = grp["item_id"].astype(str).tolist()

    def retrieve(
        self,
        cart_items: list[str],
        restaurant_id: str,
        exclude: set[str],
        k: int = 40,
    ) -> list[tuple[str, float]]:
        """Find items that fill missing meal categories."""
        # Determine current cart categories
        cart_categories = set()
        for item in cart_items:
            cat = self.item_to_category.get(str(item), "unknown")
            if cat != "unknown":
                cart_categories.add(cat)

        # Find the best-matching archetype and its missing categories
        best_arch = None
        best_overlap = -1
        for arch in MEAL_ARCHETYPES:
            overlap = len(cart_categories & arch)
            if overlap > best_overlap:
                best_overlap = overlap
                best_arch = arch

        if best_arch is None:
            return []

        missing_cats = best_arch - cart_categories
        if not missing_cats:
            # Cart already complete — fall back to complementary categories
            # from other archetypes
            all_cats = set()
            for arch in MEAL_ARCHETYPES:
                if cart_categories & arch:
                    all_cats |= arch
            missing_cats = all_cats - cart_categories

        if not missing_cats:
            return []

        # Retrieve popular items from missing categories
        candidates: list[tuple[str, float]] = []
        for cat in missing_cats:
            pool = self.restaurant_cat_items.get((str(restaurant_id), cat), [])
            for rank_idx, item_id in enumerate(pool[:20]):
                if item_id in exclude:
                    continue
                # Score: inverse rank * category importance (main_course > addon)
                cat_weight = 1.5 if cat == "main_course" else (1.2 if cat in {"beverage", "dessert"} else 1.0)
                score = cat_weight / (rank_idx + 1)
                candidates.append((item_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
