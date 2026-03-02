"""Session co-visitation retriever.

Captures sequential add-to-cart patterns: "users who added items A,B
also added C next". This is a lightweight session-graph approach where
edges are (item_i, item_j) weighted by how often j immediately follows
i in order sequences.

This is a high-signal retriever for CSAO because it models actual
cart-building behaviour rather than just co-occurrence in baskets.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class SessionCovisitRetriever:
    """Retrieves candidates based on sequential add-to-cart transitions.

    Index structure:
        item_id → [(next_item_id, transition_score), ...]
        sorted by score descending.

    Transition score = count(i→j) / count(i) * decay_factor^(position_gap-1)
    """

    def __init__(
        self,
        orders: pd.DataFrame,
        order_items: pd.DataFrame,
        max_position_gap: int = 3,
        min_transitions: int = 2,
    ) -> None:
        self.index: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._build_index(orders, order_items, max_position_gap, min_transitions)

    def _build_index(
        self,
        orders: pd.DataFrame,
        order_items: pd.DataFrame,
        max_position_gap: int,
        min_transitions: int,
    ) -> None:
        """Build transition graph from order item sequences (vectorized)."""
        oi = order_items.sort_values(["order_id", "position"]).copy()
        oi["item_id"] = oi["item_id"].astype(str)

        # Source counts: how many times each item appears (vectorized)
        source_counts_s = oi["item_id"].value_counts()

        # Build transitions for each position gap using vectorized shift
        frames = []
        for gap in range(1, max_position_gap + 1):
            shifted = oi.copy()
            shifted["dst"] = shifted.groupby("order_id")["item_id"].shift(-gap)
            shifted = shifted.dropna(subset=["dst"])
            shifted["decay"] = 1.0 / gap
            frames.append(shifted[["item_id", "dst", "decay"]])

        if not frames:
            return

        all_trans = pd.concat(frames, ignore_index=True)
        # Aggregate transition counts with decay weights
        agg = all_trans.groupby(["item_id", "dst"])["decay"].sum().reset_index()
        agg.columns = ["src", "dst", "count"]

        # Filter by minimum support
        agg = agg[agg["count"] >= min_transitions]

        # Normalise by source frequency
        agg["src_freq"] = agg["src"].map(source_counts_s).fillna(1).clip(lower=1)
        agg["score"] = agg["count"] / agg["src_freq"]

        # Build index sorted by score descending
        for _, row in agg.sort_values("score", ascending=False).iterrows():
            self.index[row["src"]].append((row["dst"], row["score"]))

    def retrieve(
        self,
        cart_items: list[str],
        exclude: set[str],
        k: int = 80,
    ) -> list[tuple[str, float]]:
        """Retrieve candidates based on sequential transitions from cart items.

        Later cart items get higher weight (recency bias — the last item
        added is the strongest signal for what comes next).
        """
        scores: dict[str, float] = defaultdict(float)
        n_cart = len(cart_items)
        for pos, item in enumerate(cart_items):
            # Recency weight: last item gets weight 1.0, first gets lower
            recency = (pos + 1) / max(n_cart, 1)
            for cand, sc in self.index.get(str(item), [])[:k]:
                if cand not in exclude:
                    scores[cand] += sc * recency

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
