from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.feature_names import normalize_feature_columns, normalize_feature_name
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
    candidate_score: float | None,
    user_feature_row: pd.Series | None,
    item_feature_row: pd.Series | None,
    item_lookup: pd.DataFrame,
    comp_lookup: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, Any]:
    feat = build_cart_feature_vector(cart_items, item_lookup)
    feat_num = {k: float(v) for k, v in feat.items() if isinstance(v, (float, int, np.floating))}

    max_lift, mean_lift, max_pmi, mean_pmi = _agg_complementarity(cart_items, candidate_item, comp_lookup)

    user_feature_row = _as_series(user_feature_row)
    item_feature_row = _as_series(item_feature_row)
    item_popularity = 0.0 if item_feature_row is None else _to_float(item_feature_row.get("item_popularity", 0.0))
    if candidate_score is None:
        # Avoid label leakage: the score should come from item/cart signals, not from target label.
        candidate_score = (
            0.65 * max_lift
            + 0.25 * mean_lift
            + 0.06 * max_pmi
            + 0.02 * mean_pmi
            + 0.01 * float(np.log1p(max(item_popularity, 0.0)))
        )

    feat_num.update(
        {
            "candidate_score": float(candidate_score),
            "comp_max_lift": max_lift,
            "comp_mean_lift": mean_lift,
            "comp_max_pmi": max_pmi,
            "comp_mean_pmi": mean_pmi,
        }
    )

    # --- CSAO-specific candidate-level features ---
    from features.cart_features import _cart_completeness_score, _missing_categories  # noqa: local import

    # Build category set for current cart
    lookup_idx = item_lookup.set_index("item_id", drop=False)
    cart_categories: set[str] = set()
    for ci in cart_items:
        if ci in lookup_idx.index:
            row_ci = lookup_idx.loc[ci]
            cat_val = str(row_ci["item_category"]) if isinstance(row_ci, pd.Series) else str(row_ci.iloc[0]["item_category"])
            if cat_val != "unknown":
                cart_categories.add(cat_val)

    cand_cat = "unknown"
    if candidate_item in lookup_idx.index:
        cand_row = lookup_idx.loc[candidate_item]
        cand_cat = str(cand_row["item_category"]) if isinstance(cand_row, pd.Series) else str(cand_row.iloc[0]["item_category"])

    # Does adding this candidate improve meal completeness?
    completeness_before = _cart_completeness_score(cart_categories)
    completeness_after = _cart_completeness_score(cart_categories | {cand_cat})
    completeness_delta = completeness_after - completeness_before

    # Does the candidate fill a gap in the best-matching archetype?
    _, missing_ratio_before = _missing_categories(cart_categories)
    _, missing_ratio_after = _missing_categories(cart_categories | {cand_cat})
    fills_gap = float(missing_ratio_after < missing_ratio_before)

    # Complement confidence: fraction of cart items with non-zero lift to candidate
    n_with_lift = sum(1 for ci in cart_items if comp_lookup.get((ci, candidate_item), (0.0, 0.0))[0] > 0)
    complement_confidence = n_with_lift / max(len(cart_items), 1)

    # Is the candidate in a new category not already in cart?
    candidate_new_category = float(cand_cat not in cart_categories and cand_cat != "unknown")

    feat_num.update(
        {
            "completeness_delta": completeness_delta,
            "fills_meal_gap": fills_gap,
            "complement_confidence": complement_confidence,
            "candidate_new_category": candidate_new_category,
        }
    )

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
    return {normalize_feature_name(k): v for k, v in feat_num.items()}


def _build_negative_sampling_weights(
    item_pool: list[str],
    item_popularity: dict[str, float],
    item_categories: dict[str, str],
    positive_item: str,
    cart_items: list[str],
    comp_lookup: dict[tuple[str, str], tuple[float, float]],
) -> np.ndarray:
    """Build sampling weights favouring harder negatives.

    Strategy (multi-signal):
    1. Popularity-weighted: popular items are harder negatives (the model
       must learn to distinguish popular-but-irrelevant from relevant).
    2. Same-category boost: items in the same category as the positive
       are near-miss hard negatives.
    3. Co-occurrence proximity: items with non-zero lift to any cart item
       are partially relevant and thus harder negatives.

    Weights are combined as:  pop * (1 + same_cat_boost + cooc_boost)
    """
    n = len(item_pool)
    weights = np.ones(n, dtype=np.float64)

    pos_cat = item_categories.get(positive_item, "unknown")
    cart_set = set(cart_items)

    for i, item in enumerate(item_pool):
        # Popularity component: log-scaled to avoid extreme skew
        pop = item_popularity.get(item, 0.0)
        weights[i] = np.log1p(max(pop, 0.0)) + 1.0

        # Same-category hard negative boost (2x)
        cat = item_categories.get(item, "unknown")
        if cat == pos_cat and cat != "unknown":
            weights[i] *= 2.0

        # Co-occurrence proximity boost — items co-occurring with cart
        for ci in cart_set:
            lift, _ = comp_lookup.get((ci, item), (0.0, 0.0))
            if lift > 0.0:
                weights[i] *= (1.0 + 0.5 * min(lift, 5.0))
                break  # one boost is enough

    # Normalise to probability distribution
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights[:] = 1.0 / n
    return weights


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

    # Build item popularity & category maps for smart negative sampling
    item_pop_series = (
        order_items.groupby("item_id")["order_id"].nunique().rename("popularity")
    )
    item_popularity: dict[str, float] = item_pop_series.to_dict()
    item_categories: dict[str, str] = (
        unified["items"].drop_duplicates("item_id").set_index("item_id")["item_category"].astype(str).to_dict()
    )

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
                candidate_score=None,
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

            # Advanced negative sampling: popularity + same-category + co-occurrence weighted
            sampling_weights = _build_negative_sampling_weights(
                item_pool=neg_pool,
                item_popularity=item_popularity,
                item_categories=item_categories,
                positive_item=positive_item,
                cart_items=cart,
                comp_lookup=comp_lookup,
            )
            n_sample = min(n_neg, len(neg_pool))
            sampled = rng.choice(neg_pool, size=n_sample, replace=False, p=sampling_weights)

            for neg_item in sampled:
                neg_row = item_index.loc[neg_item] if neg_item in item_index.index else None
                neg_features = _build_feature_row(
                    user_id=user_id,
                    restaurant_id=restaurant_id,
                    cart_items=cart,
                    candidate_item=str(neg_item),
                    candidate_score=None,
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
    X.columns = normalize_feature_columns(list(X.columns))
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
