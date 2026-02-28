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
        """Build transition graph from order item sequences."""
        oi = order_items.sort_values(["order_id", "position"]).copy()
        oi["item_id"] = oi["item_id"].astype(str)

        transition_counts: defaultdict[tuple[str, str], float] = defaultdict(float)
        source_counts: defaultdict[str, int] = defaultdict(int)

        for _, group in oi.groupby("order_id"):
            items = group["item_id"].tolist()
            n = len(items)
            for i in range(n):
                source_counts[items[i]] += 1
                for gap in range(1, min(max_position_gap + 1, n - i)):
                    j = i + gap
                    if j < n:
                        decay = 1.0 / gap  # closer transitions get higher weight
                        transition_counts[(items[i], items[j])] += decay

        # Normalise by source frequency and filter by minimum support
        scored: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        for (src, dst), count in transition_counts.items():
            if count < min_transitions:
                continue
            src_freq = max(source_counts[src], 1)
            score = count / src_freq
            scored[src].append((dst, score))

        # Sort each source's candidates by score
        for src in scored:
            self.index[src] = sorted(scored[src], key=lambda x: x[1], reverse=True)

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
