from __future__ import annotations

import numpy as np
import pandas as pd

from common.feature_names import normalize_feature_name


def build_user_features(
    users: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    orders_local = orders.copy()
    orders_local["order_ts"] = pd.to_datetime(orders_local["order_ts"], errors="coerce")

    active_days = (
        orders_local.groupby("user_id")["order_ts"].agg(lambda s: max((s.max() - s.min()).days, 1))
        .rename("active_days")
        .reset_index()
    )
    order_freq = orders_local.groupby("user_id")["order_id"].nunique().rename("order_count").reset_index()
    order_freq = order_freq.merge(active_days, on="user_id", how="left")
    order_freq["order_frequency"] = order_freq["order_count"] / order_freq["active_days"].clip(lower=1)

    avg_order_value = orders_local.groupby("user_id")["total_value"].mean().rename("avg_order_value").reset_index()

    # --- RFM Recency: days since last order (lower = more recent) ---
    ref_date = orders_local["order_ts"].max()
    recency = (
        orders_local.groupby("user_id")["order_ts"]
        .max()
        .rename("last_order_ts")
        .reset_index()
    )
    recency["recency_days"] = (ref_date - recency["last_order_ts"]).dt.days.clip(lower=0)
    recency = recency[["user_id", "recency_days"]]

    # --- Total monetary value (RFM 'M') ---
    total_spend = orders_local.groupby("user_id")["total_value"].sum().rename("total_spend").reset_index()

    # --- User segment (budget / mid / premium) based on avg order value ---
    def _user_segment(aov: float) -> float:
        if aov < 250:
            return 0.0  # budget
        elif aov < 500:
            return 1.0  # mid
        else:
            return 2.0  # premium

    cuisine_pivot = (
        orders_local.assign(cnt=1)
        .pivot_table(index="user_id", columns="cuisine", values="cnt", aggfunc="sum", fill_value=0)
        .astype(float)
    )
    cuisine_share = cuisine_pivot.div(cuisine_pivot.sum(axis=1).replace(0.0, 1.0), axis=0)
    cuisine_share.columns = [f"user_cuisine_share__{normalize_feature_name(c)}" for c in cuisine_share.columns]
    cuisine_share = cuisine_share.reset_index()

    item_price = items[["item_id", "item_price"]].drop_duplicates("item_id")
    order_item_prices = order_items.merge(item_price, on="item_id", how="left")
    order_item_prices["item_price"] = order_item_prices["item_price"].fillna(order_item_prices["unit_price"])
    user_price = (
        order_item_prices.merge(orders_local[["order_id", "user_id"]], on="order_id", how="left")
        .groupby("user_id")["item_price"]
        .mean()
        .rename("avg_item_price_user")
        .reset_index()
    )
    global_median = float(item_price["item_price"].median()) if not item_price.empty else 1.0
    user_price["price_sensitivity"] = user_price["avg_item_price_user"] / max(global_median, 1e-6)

    feats = (
        users[["user_id"]]
        .merge(order_freq[["user_id", "order_count", "order_frequency"]], on="user_id", how="left")
        .merge(avg_order_value, on="user_id", how="left")
        .merge(recency, on="user_id", how="left")
        .merge(total_spend, on="user_id", how="left")
        .merge(cuisine_share, on="user_id", how="left")
        .merge(user_price[["user_id", "price_sensitivity"]], on="user_id", how="left")
    )
    feats["user_segment"] = feats["avg_order_value"].apply(_user_segment)
    feats = feats.fillna(0.0)
    numeric_cols = [c for c in feats.columns if c != "user_id"]
    feats[numeric_cols] = feats[numeric_cols].replace([np.inf, -np.inf], 0.0)
    return feats
