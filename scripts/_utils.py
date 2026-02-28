from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.config import load_config
from common.io import load_table


def load_project_config() -> dict[str, Any]:
    return load_config("configs/base.yaml", "configs/ranking.yaml", "configs/serving.yaml")


def load_unified_tables(processed_dir: str) -> dict[str, pd.DataFrame]:
    p = Path(processed_dir)
    names = ["users", "orders", "order_items", "items", "restaurants"]
    out = {name: load_table(p / f"{name}.parquet", required=True) for name in names}
    recipe_path = p / "recipe_embeddings.parquet"
    recipe_df = load_table(recipe_path, required=False)
    out["recipe_embeddings"] = recipe_df if recipe_df is not None else pd.DataFrame(columns=["item_id"])
    return out


def load_feature_tables(processed_dir: str) -> dict[str, pd.DataFrame]:
    p = Path(processed_dir)
    names = {
        "user_features": "features_user.parquet",
        "item_features": "features_item.parquet",
        "complementarity": "features_complementarity.parquet",
        "category_affinity": "features_category_affinity.parquet",
        "cart_context": "features_cart_context.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, file_name in names.items():
        path = p / file_name
        df = load_table(path, required=False)
        out[key] = df if df is not None else pd.DataFrame()
    return out
