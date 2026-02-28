from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from features.cart_features import build_cart_feature_vector


@dataclass
class TrainingData:
    X: pd.DataFrame
    y: np.ndarray
    group: list[int]
    query_ids: list[str]
    candidate_items: list[str]
    query_meta: pd.DataFrame


def _as_series(row: pd.Series | pd.DataFrame | None) -> pd.Series | None:
    if row is None:
        return None
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        return row.iloc[0]
    return row


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


def _agg_complementarity(
    cart_items: list[str],
    candidate_item: str,
    comp_lookup: dict[tuple[str, str], tuple[float, float]],
) -> tuple[float, float, float, float]:
    lifts: list[float] = []
    pmis: list[float] = []
    for cart_item in cart_items:
        lift, pmi = comp_lookup.get((str(cart_item), str(candidate_item)), (0.0, 0.0))
        lifts.append(float(lift))
        pmis.append(float(pmi))
    if not lifts:
        return 0.0, 0.0, 0.0, 0.0
    return max(lifts), float(np.mean(lifts)), max(pmis), float(np.mean(pmis))


def _build_feature_row(
    user_id: str,
    restaurant_id: str,
    cart_items: list[str],
    candidate_item: str,
    candidate_score: float,
    user_feature_row: pd.Series | None,
    item_feature_row: pd.Series | None,
    item_lookup: pd.DataFrame,
    comp_lookup: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, Any]:
    feat = build_cart_feature_vector(cart_items, item_lookup)
    feat_num = {k: float(v) for k, v in feat.items() if isinstance(v, (float, int, np.floating))}

    max_lift, mean_lift, max_pmi, mean_pmi = _agg_complementarity(cart_items, candidate_item, comp_lookup)
    feat_num.update(
        {
            "candidate_score": float(candidate_score),
            "comp_max_lift": max_lift,
            "comp_mean_lift": mean_lift,
            "comp_max_pmi": max_pmi,
            "comp_mean_pmi": mean_pmi,
        }
    )

    user_feature_row = _as_series(user_feature_row)
    item_feature_row = _as_series(item_feature_row)

    if user_feature_row is not None:
        for col, value in user_feature_row.items():
            if col == "user_id":
                continue
            feat_num[f"user__{col}"] = _to_float(value)
    if item_feature_row is not None:
        for col, value in item_feature_row.items():
            if col in {"item_id", "item_category"}:
                continue
            feat_num[f"item__{col}"] = _to_float(value)

    # TODO(prod): include device, distance, ETA, and current-time context.
    feat_num["ctx_restaurant_hash"] = float(abs(hash(str(restaurant_id))) % 1000)
    feat_num["ctx_user_hash"] = float(abs(hash(str(user_id))) % 1000)
    return feat_num


def build_training_dataset(
    unified: dict[str, pd.DataFrame],
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    comp_lookup: dict[tuple[str, str], tuple[float, float]],
    config: dict[str, Any],
) -> TrainingData:
    orders = unified["orders"]
    order_items = unified["order_items"]
    item_lookup = unified["items"][["item_id", "item_category", "item_price"]].drop_duplicates("item_id")

    n_neg = int(config.get("ranking", {}).get("negative_samples_per_positive", 4))
    random_state = int(config.get("ranking", {}).get("random_state", 42))
    rng = np.random.default_rng(random_state)

    user_index = user_features.drop_duplicates("user_id").set_index("user_id", drop=False)
    item_index = item_features.drop_duplicates("item_id").set_index("item_id", drop=False)

    rest_item_pool = (
        order_items.merge(orders[["order_id", "restaurant_id"]], on="order_id", how="left")
        .groupby("restaurant_id")["item_id"]
        .apply(lambda s: list(set(s.astype(str).tolist())))
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    y: list[int] = []
    query_ids: list[str] = []
    candidate_items: list[str] = []
    query_meta_rows: list[dict[str, Any]] = []

    oi = order_items.merge(orders[["order_id", "user_id", "restaurant_id"]], on="order_id", how="left")
    for order_id, group in oi.groupby("order_id"):
        seq = group.sort_values("position")
        item_seq = seq["item_id"].astype(str).tolist()
        user_id = str(seq["user_id"].iloc[0])
        restaurant_id = str(seq["restaurant_id"].iloc[0])

        if len(item_seq) < 2:
            continue

        pool = rest_item_pool.get(restaurant_id, [])
        for pos in range(1, len(item_seq)):
            cart = item_seq[:pos]
            positive_item = item_seq[pos]

            qid = f"{order_id}__{pos}"
            user_row = user_index.loc[user_id] if user_id in user_index.index else None
            pos_item_row = item_index.loc[positive_item] if positive_item in item_index.index else None
            pos_features = _build_feature_row(
                user_id=user_id,
                restaurant_id=restaurant_id,
                cart_items=cart,
                candidate_item=positive_item,
                candidate_score=1.0,
                user_feature_row=user_row,
                item_feature_row=pos_item_row,
                item_lookup=item_lookup,
                comp_lookup=comp_lookup,
            )
            rows.append(pos_features)
            y.append(1)
            query_ids.append(qid)
            candidate_items.append(positive_item)
            query_meta_rows.append({"query_id": qid, "user_id": user_id, "restaurant_id": restaurant_id, "positive_item": positive_item})

            neg_pool = [i for i in pool if i not in set(cart) and i != positive_item]
            if len(neg_pool) == 0:
                continue
            sampled = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False)
            for neg_item in sampled:
                neg_row = item_index.loc[neg_item] if neg_item in item_index.index else None
                neg_features = _build_feature_row(
                    user_id=user_id,
                    restaurant_id=restaurant_id,
                    cart_items=cart,
                    candidate_item=str(neg_item),
                    candidate_score=0.2,
                    user_feature_row=user_row,
                    item_feature_row=neg_row,
                    item_lookup=item_lookup,
                    comp_lookup=comp_lookup,
                )
                rows.append(neg_features)
                y.append(0)
                query_ids.append(qid)
                candidate_items.append(str(neg_item))
                query_meta_rows.append({"query_id": qid, "user_id": user_id, "restaurant_id": restaurant_id, "positive_item": positive_item})

    if not rows:
        raise ValueError("No training rows generated. Check unified tables and feature inputs.")

    X = pd.DataFrame(rows).fillna(0.0)
    X = X.reindex(sorted(X.columns), axis=1)
    y_arr = np.array(y, dtype=int)

    query_series = pd.Series(query_ids, name="query_id")
    group = query_series.value_counts(sort=False).tolist()
    query_meta = pd.DataFrame(query_meta_rows).drop_duplicates("query_id").reset_index(drop=True)
    return TrainingData(
        X=X,
        y=y_arr,
        group=group,
        query_ids=query_ids,
        candidate_items=candidate_items,
        query_meta=query_meta,
    )
