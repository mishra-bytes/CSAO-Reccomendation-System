from __future__ import annotations

import numpy as np
import pandas as pd


def build_item_features(items: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    items = items.drop_duplicates("item_id").copy()
    popularity = (
        order_items.groupby("item_id")["order_id"]
        .nunique()
        .rename("item_popularity")
        .reset_index()
    )

    feat = items.merge(popularity, on="item_id", how="left")
    feat["item_popularity"] = feat["item_popularity"].fillna(0).astype(float)

    # Veg/non-veg indicator (🟢/🔴 — fundamental for Indian food delivery)
    if "is_veg" in feat.columns:
        feat["is_veg"] = feat["is_veg"].fillna(True).astype(float)
    else:
        feat["is_veg"] = 1.0  # default to veg if column missing

    feat["price_band"] = pd.qcut(
        feat["item_price"].rank(method="first"),
        q=4,
        labels=["low", "mid", "high", "premium"],
    ).astype(str)

    cat_ohe = pd.get_dummies(feat["item_category"].fillna("unknown"), prefix="item_cat")
    price_band_ohe = pd.get_dummies(feat["price_band"].fillna("unknown"), prefix="item_price_band")

    out = pd.concat([feat[["item_id", "item_category", "item_price", "item_popularity", "is_veg"]], cat_ohe, price_band_ohe], axis=1)
    numeric_cols = [c for c in out.columns if c not in {"item_id", "item_category"}]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
