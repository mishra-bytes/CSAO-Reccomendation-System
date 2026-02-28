from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from features.cart_features import build_cart_context_table
from features.complementarity import build_complementarity_lookup, compute_category_affinity, compute_item_complementarity
from features.item_features import build_item_features
from features.store.cache import save_df
from features.user_features import build_user_features


@dataclass
class FeatureArtifacts:
    user_features: pd.DataFrame
    item_features: pd.DataFrame
    complementarity: pd.DataFrame
    category_affinity: pd.DataFrame
    cart_context: pd.DataFrame
    complementarity_lookup: dict[tuple[str, str], tuple[float, float]]


def build_feature_artifacts(unified: dict[str, pd.DataFrame], config: dict[str, Any]) -> FeatureArtifacts:
    max_cart_categories = int(config.get("feature_build", {}).get("max_cart_categories", 12))
    min_item_support = int(config.get("feature_build", {}).get("min_item_support", 2))

    user_features = build_user_features(
        users=unified["users"],
        orders=unified["orders"],
        order_items=unified["order_items"],
        items=unified["items"],
    )
    item_features = build_item_features(unified["items"], unified["order_items"])
    complementarity = compute_item_complementarity(unified["order_items"], min_support=min_item_support)
    category_affinity = compute_category_affinity(unified["order_items"], unified["items"], min_support=min_item_support)
    cart_context = build_cart_context_table(unified["order_items"], unified["items"], max_categories=max_cart_categories)
    comp_lookup = build_complementarity_lookup(complementarity)

    return FeatureArtifacts(
        user_features=user_features,
        item_features=item_features,
        complementarity=complementarity,
        category_affinity=category_affinity,
        cart_context=cart_context,
        complementarity_lookup=comp_lookup,
    )


def save_feature_artifacts(artifacts: FeatureArtifacts, processed_dir: str) -> None:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_df(artifacts.user_features, out_dir / "features_user.parquet")
    save_df(artifacts.item_features, out_dir / "features_item.parquet")
    save_df(artifacts.complementarity, out_dir / "features_complementarity.parquet")
    save_df(artifacts.category_affinity, out_dir / "features_category_affinity.parquet")
    save_df(artifacts.cart_context, out_dir / "features_cart_context.parquet")

