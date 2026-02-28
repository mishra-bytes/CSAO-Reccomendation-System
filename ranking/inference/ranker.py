from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from features.cart_features import build_cart_feature_vector


class CSAORanker:
    def __init__(
        self,
        model_path: str,
        feature_columns_path: str,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        items: pd.DataFrame,
        complementarity_lookup: dict[tuple[str, str], tuple[float, float]],
    ) -> None:
        self.model = joblib.load(model_path)
        self.feature_columns = json.loads(Path(feature_columns_path).read_text(encoding="utf-8"))
        self.user_index = user_features.drop_duplicates("user_id").set_index("user_id", drop=False)
        self.item_index = item_features.drop_duplicates("item_id").set_index("item_id", drop=False)
        self.items = items[["item_id", "item_category", "item_price"]].drop_duplicates("item_id")
        self.comp_lookup = complementarity_lookup

    @staticmethod
    def _as_series(row: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, pd.Series):
            if value.empty:
                return 0.0
            value = value.iloc[0]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            value = value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return 0.0
            value = value[0]
        if pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _complement_agg(self, cart_items: list[str], candidate_item: str) -> tuple[float, float, float, float]:
        lifts: list[float] = []
        pmis: list[float] = []
        for i in cart_items:
            lift, pmi = self.comp_lookup.get((str(i), str(candidate_item)), (0.0, 0.0))
            lifts.append(float(lift))
            pmis.append(float(pmi))
        if not lifts:
            return 0.0, 0.0, 0.0, 0.0
        return max(lifts), float(np.mean(lifts)), max(pmis), float(np.mean(pmis))

    def _feature_row(
        self,
        user_id: str,
        restaurant_id: str,
        cart_items: list[str],
        candidate_item: str,
        candidate_score: float,
    ) -> dict[str, float]:
        cart = build_cart_feature_vector(cart_items, self.items)
        row = {k: float(v) for k, v in cart.items() if isinstance(v, (float, int, np.floating))}
        max_lift, mean_lift, max_pmi, mean_pmi = self._complement_agg(cart_items, candidate_item)
        row.update(
            {
                "candidate_score": float(candidate_score),
                "comp_max_lift": max_lift,
                "comp_mean_lift": mean_lift,
                "comp_max_pmi": max_pmi,
                "comp_mean_pmi": mean_pmi,
                "ctx_restaurant_hash": float(abs(hash(str(restaurant_id))) % 1000),
                "ctx_user_hash": float(abs(hash(str(user_id))) % 1000),
            }
        )

        if user_id in self.user_index.index:
            user_row = self._as_series(self.user_index.loc[user_id])
            for col, value in user_row.items():
                if col == "user_id":
                    continue
                row[f"user__{col}"] = self._to_float(value)
        if candidate_item in self.item_index.index:
            item_row = self._as_series(self.item_index.loc[candidate_item])
            for col, value in item_row.items():
                if col in {"item_id", "item_category"}:
                    continue
                row[f"item__{col}"] = self._to_float(value)

        for col in self.feature_columns:
            row.setdefault(col, 0.0)
        return {c: float(row[c]) for c in self.feature_columns}

    def rank(
        self,
        user_id: str,
        restaurant_id: str,
        cart_items: list[str],
        candidates: list[tuple[str, float]],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        rows = []
        ids = []
        for item_id, cand_score in candidates:
            rows.append(self._feature_row(user_id, restaurant_id, cart_items, str(item_id), float(cand_score)))
            ids.append(str(item_id))

        X = pd.DataFrame(rows).reindex(columns=self.feature_columns).fillna(0.0)
        scores = self.model.predict(X)
        out = [
            {"item_id": ids[i], "rank_score": float(scores[i]), "candidate_score": float(candidates[i][1])}
            for i in range(len(ids))
        ]
        out = sorted(out, key=lambda x: x["rank_score"], reverse=True)
        return out[:top_n]
