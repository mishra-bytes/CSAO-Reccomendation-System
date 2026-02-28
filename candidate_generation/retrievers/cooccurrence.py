from __future__ import annotations

from collections import defaultdict

import pandas as pd


class CooccurrenceRetriever:
    def __init__(self, complementarity_df: pd.DataFrame):
        self.index: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in complementarity_df.itertuples(index=False):
            score = float(row.lift) + 0.2 * float(row.pmi)
            self.index[str(row.item_id)].append((str(row.candidate_item_id), score))
        for item_id in list(self.index.keys()):
            self.index[item_id] = sorted(self.index[item_id], key=lambda x: x[1], reverse=True)

    def retrieve(self, cart_items: list[str], k: int = 120) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for item in cart_items:
            for cand, sc in self.index.get(str(item), [])[:k]:
                scores[cand] += sc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

