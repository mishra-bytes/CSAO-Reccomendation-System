from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.feature_names import normalize_feature_columns, normalize_feature_name
from features.cart_features import build_cart_feature_vector, _cart_completeness_score, _missing_categories
from collections import Counter


def _fast_cart_features(
    cart_item_ids: list[str],
    item_cat_dict: dict[str, str],
    item_price_dict: dict[str, float],
) -> dict[str, float]:
    """Ultra-fast cart feature computation using pure dicts (no DataFrame ops)."""
    if not cart_item_ids:
        return {
            "cart_size": 0.0, "cart_value": 0.0, "session_position": 0.0,
            "cart_completeness": 0.0, "cart_missing_cats": 0.0, "cart_missing_cat_ratio": 0.0,
            "cart_unique_categories": 0.0, "cart_avg_price": 0.0, "cart_price_std": 0.0,
            "cart_has_main": 0.0, "cart_has_beverage": 0.0, "cart_has_dessert": 0.0,
            "cart_has_starter": 0.0,
        }

    cart_size = float(len(cart_item_ids))
    prices = [item_price_dict.get(ci, 0.0) for ci in cart_item_ids]
    cart_value = sum(prices)
    cart_avg_price = cart_value / cart_size if cart_size > 0 else 0.0
    if cart_size > 1:
        mean_p = cart_value / cart_size
        cart_price_std = (sum((p - mean_p) ** 2 for p in prices) / cart_size) ** 0.5
    else:
        cart_price_std = 0.0

    categories = [item_cat_dict.get(ci, "unknown") for ci in cart_item_ids]
    cat_counts = Counter(categories)
    cat_set = set(categories) - {"unknown"}
    total = max(sum(cat_counts.values()), 1)

    completeness = _cart_completeness_score(cat_set)
    n_missing, missing_ratio = _missing_categories(cat_set)

    features: dict[str, float] = {
        "cart_size": cart_size,
        "cart_value": cart_value,
        "cart_avg_price": cart_avg_price,
        "cart_price_std": cart_price_std,
        "session_position": cart_size,
        "cart_completeness": completeness,
        "cart_missing_cats": float(n_missing),
        "cart_missing_cat_ratio": missing_ratio,
        "cart_unique_categories": float(len(cat_set)),
        "cart_has_main": float("main_course" in cat_set),
        "cart_has_beverage": float("beverage" in cat_set),
        "cart_has_dessert": float("dessert" in cat_set),
        "cart_has_starter": float("starter" in cat_set),
    }
    for cat, cnt in cat_counts.most_common(12):
        safe_cat = normalize_feature_name(cat)
        features[f"cart_cat_share__{safe_cat}"] = cnt / total
    return features


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
    item_cat_dict: dict[str, str] | None = None,
    item_price_dict: dict[str, float] | None = None,
) -> dict[str, Any]:
    feat = build_cart_feature_vector(cart_items, item_lookup)
    feat_num = {k: float(v) for k, v in feat.items() if isinstance(v, (float, int, np.floating))}

    max_lift, mean_lift, max_pmi, mean_pmi = _agg_complementarity(cart_items, candidate_item, comp_lookup)

    user_feature_row = _as_series(user_feature_row)
    item_feature_row = _as_series(item_feature_row)
    item_popularity = 0.0 if item_feature_row is None else _to_float(item_feature_row.get("item_popularity", 0.0))
    if candidate_score is None:
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

    # --- CSAO-specific candidate-level features (dict-based for speed) ---
    # Use dict lookups instead of DataFrame .loc for categories
    if item_cat_dict is not None:
        cart_categories: set[str] = set()
        for ci in cart_items:
            cat_val = item_cat_dict.get(ci, "unknown")
            if cat_val != "unknown":
                cart_categories.add(cat_val)
        cand_cat = item_cat_dict.get(candidate_item, "unknown")
    else:
        lookup_idx = item_lookup if item_lookup.index.name == "item_id" else item_lookup.set_index("item_id", drop=False)
        cart_categories = set()
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

    Vectorised strategy:
    1. Popularity-weighted: popular items are harder negatives.
    2. Same-category boost (2x): near-miss hard negatives.
    3. Co-occurrence proximity boost (capped).
    """
    n = len(item_pool)

    # Vectorised popularity weights
    pop_arr = np.array([item_popularity.get(item, 0.0) for item in item_pool], dtype=np.float64)
    weights = np.log1p(np.maximum(pop_arr, 0.0)) + 1.0

    # Same-category boost
    pos_cat = item_categories.get(positive_item, "unknown")
    if pos_cat != "unknown":
        cat_arr = np.array([1 if item_categories.get(item, "unknown") == pos_cat else 0 for item in item_pool])
        weights *= (1.0 + cat_arr)  # 2x for same category

    # Co-occurrence boost (sampling a single cart item for speed)
    if cart_items and comp_lookup:
        cart_sample = cart_items[:3]  # check at most 3 cart items
        for ci in cart_sample:
            boosts = np.array([
                min(comp_lookup.get((ci, item), (0.0, 0.0))[0], 5.0)
                for item in item_pool
            ])
            mask = boosts > 0.0
            if mask.any():
                weights[mask] *= (1.0 + 0.5 * boosts[mask])
                break

    # Normalise
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
    # Pre-index for O(1) lookup — avoids re-indexing on every feature row call
    item_lookup_indexed = item_lookup.set_index("item_id", drop=False)

    n_neg = int(config.get("ranking", {}).get("negative_samples_per_positive", 4))
    random_state = int(config.get("ranking", {}).get("random_state", 42))
    max_training_orders = int(config.get("ranking", {}).get("max_training_orders", 50_000))
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

    _raw_pool = (
        order_items.merge(orders[["order_id", "restaurant_id"]], on="order_id", how="left")
        .groupby("restaurant_id")["item_id"]
        .apply(lambda s: list(set(s.astype(str).tolist())))
        .to_dict()
    )
    # Pre-cap restaurant pools to avoid 32K-item neg-pool filtering
    _MAX_POOL = 500
    rest_item_pool: dict[str, list[str]] = {}
    for rid, pool in _raw_pool.items():
        if len(pool) > _MAX_POOL:
            # Keep most popular items in each pool
            scored = sorted(pool, key=lambda i: item_popularity.get(i, 0.0), reverse=True)
            rest_item_pool[rid] = scored[:_MAX_POOL]
        else:
            rest_item_pool[rid] = pool
    del _raw_pool
    print(f"[dataset] Restaurant pools: {len(rest_item_pool)} restaurants, max pool={max(len(v) for v in rest_item_pool.values()) if rest_item_pool else 0}")

    # Pre-build dict lookups for fast feature computation (avoids DataFrame .loc per row)
    item_cat_dict: dict[str, str] = item_categories  # already built above
    item_price_dict: dict[str, float] = (
        unified["items"].drop_duplicates("item_id").set_index("item_id")["item_price"].astype(float).to_dict()
    )

    rows: list[dict[str, Any]] = []
    y: list[int] = []
    query_ids: list[str] = []
    candidate_items: list[str] = []
    query_meta_rows: list[dict[str, Any]] = []

    oi = order_items.merge(orders[["order_id", "user_id", "restaurant_id"]], on="order_id", how="left")

    # Cap orders for tractable training time on large datasets
    unique_order_ids = oi["order_id"].unique()
    if len(unique_order_ids) > max_training_orders:
        sampled_order_ids = set(rng.choice(unique_order_ids, size=max_training_orders, replace=False))
        oi = oi[oi["order_id"].isin(sampled_order_ids)]
        print(f"[dataset] Sampled {max_training_orders}/{len(unique_order_ids)} orders for training")
    else:
        print(f"[dataset] Using all {len(unique_order_ids)} orders for training")

    order_groups = list(oi.groupby("order_id"))
    total_orders = len(order_groups)
    max_positions_per_order = 15  # Cap to prevent huge orders from dominating
    print(f"[dataset] Building features for {total_orders} orders (max {max_positions_per_order} positions/order) ...")

    for oi_idx, (order_id, group) in enumerate(order_groups):
        if oi_idx % 500 == 0:
            print(f"[dataset] Progress: {oi_idx}/{total_orders} orders ({oi_idx*100//max(total_orders,1)}%), {len(rows)} rows", flush=True)
        seq = group.sort_values("position")
        item_seq = seq["item_id"].astype(str).tolist()
        user_id = str(seq["user_id"].iloc[0])
        restaurant_id = str(seq["restaurant_id"].iloc[0])

        if len(item_seq) < 2:
            continue

        pool = rest_item_pool.get(restaurant_id, [])
        user_row = user_index.loc[user_id] if user_id in user_index.index else None
        # Pre-convert user features to dict once per order
        user_feat_dict: dict[str, float] = {}
        if user_row is not None:
            ur = _as_series(user_row)
            if ur is not None:
                for col, value in ur.items():
                    if col == "user_id":
                        continue
                    user_feat_dict[f"user__{col}"] = _to_float(value)

        for pos in range(1, min(len(item_seq), max_positions_per_order + 1)):
            cart = item_seq[:pos]
            positive_item = item_seq[pos]

            qid = f"{order_id}__{pos}"
            # Compute cart features ONCE per position (shared across positive + all negatives)
            cart_feat_num = _fast_cart_features(cart, item_cat_dict, item_price_dict)

            # Pre-compute cart categories once per position
            cart_categories: set[str] = set()
            for ci in cart:
                cat_val = item_cat_dict.get(ci, "unknown")
                if cat_val != "unknown":
                    cart_categories.add(cat_val)

            ctx_feats = {
                "ctx_restaurant_hash": float(abs(hash(str(restaurant_id))) % 1000),
                "ctx_user_hash": float(abs(hash(str(user_id))) % 1000),
            }

            def _build_candidate_features(candidate_item: str) -> dict[str, Any]:
                """Build features for a single candidate, reusing cached cart features."""
                feat_num = dict(cart_feat_num)  # shallow copy of cart features
                feat_num.update(user_feat_dict)
                feat_num.update(ctx_feats)

                # Complementarity features
                max_lift, mean_lift, max_pmi, mean_pmi = _agg_complementarity(cart, candidate_item, comp_lookup)

                item_row = item_index.loc[candidate_item] if candidate_item in item_index.index else None
                item_row_s = _as_series(item_row)
                item_pop = 0.0 if item_row_s is None else _to_float(item_row_s.get("item_popularity", 0.0))
                candidate_score = (
                    0.65 * max_lift + 0.25 * mean_lift + 0.06 * max_pmi + 0.02 * mean_pmi
                    + 0.01 * float(np.log1p(max(item_pop, 0.0)))
                )
                feat_num.update({
                    "candidate_score": float(candidate_score),
                    "comp_max_lift": max_lift,
                    "comp_mean_lift": mean_lift,
                    "comp_max_pmi": max_pmi,
                    "comp_mean_pmi": mean_pmi,
                })

                # CSAO candidate-level features
                cand_cat = item_cat_dict.get(candidate_item, "unknown")
                completeness_before = _cart_completeness_score(cart_categories)
                completeness_after = _cart_completeness_score(cart_categories | {cand_cat})
                _, missing_ratio_before = _missing_categories(cart_categories)
                _, missing_ratio_after = _missing_categories(cart_categories | {cand_cat})
                n_with_lift = sum(1 for ci in cart if comp_lookup.get((ci, candidate_item), (0.0, 0.0))[0] > 0)
                feat_num.update({
                    "completeness_delta": completeness_after - completeness_before,
                    "fills_meal_gap": float(missing_ratio_after < missing_ratio_before),
                    "complement_confidence": n_with_lift / max(len(cart), 1),
                    "candidate_new_category": float(cand_cat not in cart_categories and cand_cat != "unknown"),
                })

                # Item features
                if item_row_s is not None:
                    for col, value in item_row_s.items():
                        if col in {"item_id", "item_category"}:
                            continue
                        feat_num[f"item__{col}"] = _to_float(value)

                return {normalize_feature_name(k): v for k, v in feat_num.items()}

            # Positive example
            pos_features = _build_candidate_features(positive_item)
            rows.append(pos_features)
            y.append(1)
            query_ids.append(qid)
            candidate_items.append(positive_item)
            query_meta_rows.append({"query_id": qid, "user_id": user_id, "restaurant_id": restaurant_id, "positive_item": positive_item})

            # Fast set-based filtering instead of list comprehension
            cart_set = set(cart)
            cart_set.add(positive_item)
            neg_pool = list(set(pool) - cart_set)
            if len(neg_pool) == 0:
                continue

            # Cap neg pool for tractability
            max_neg_pool = 200
            if len(neg_pool) > max_neg_pool:
                neg_pool = list(rng.choice(neg_pool, size=max_neg_pool, replace=False))

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
                neg_features = _build_candidate_features(str(neg_item))
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
