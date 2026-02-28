from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.schemas import normalize_columns


def _load_csv_or_parquet(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def _generate_synthetic_orders(n_orders: int, source: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_ids = [f"i_{i:03d}" for i in range(1, 121)]
    categories = ["main_course", "beverage", "dessert", "starter", "addon"]
    cities = ["Delhi", "Noida", "Gurgaon"]
    cuisines = ["North Indian", "Chinese", "Italian", "Biryani"]

    rows: list[dict[str, Any]] = []
    for order_idx in range(n_orders):
        order_id = f"{source}_o_{order_idx:06d}"
        user_id = f"u_{rng.integers(1, 5000):05d}"
        restaurant_id = f"r_{rng.integers(1, 500):04d}"
        city = rng.choice(cities)
        cuisine = rng.choice(cuisines)
        ts = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 525600)))

        basket_size = int(rng.integers(1, 7))
        chosen_items = rng.choice(item_ids, size=basket_size, replace=False)
        for pos, item_id in enumerate(chosen_items, start=1):
            price = float(np.round(rng.uniform(50, 450), 2))
            quantity = int(rng.integers(1, 3))
            rows.append(
                {
                    "order_id": order_id,
                    "user_id": user_id,
                    "restaurant_id": restaurant_id,
                    "order_time": ts,
                    "item_id": item_id,
                    "item_name": f"Item {item_id}",
                    "item_type": rng.choice(categories),
                    "price": price,
                    "quantity": quantity,
                    "line_total": float(np.round(price * quantity, 2)),
                    "position": pos,
                    "city": city,
                    "cuisine": cuisine,
                    "restaurant_name": f"Restaurant {restaurant_id}",
                }
            )
    return pd.DataFrame(rows)


def _generate_synthetic_embeddings(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_ids = [f"i_{i:03d}" for i in range(1, 121)]
    embeddings = rng.normal(0, 1, size=(len(item_ids), 8))
    df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(8)])
    df.insert(0, "item_id", item_ids)
    return df


def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "customer_id": "user_id",
        "user": "user_id",
        "restaurant": "restaurant_id",
        "restaurantid": "restaurant_id",
        "order_time_stamp": "order_time",
        "timestamp": "order_time",
        "datetime": "order_time",
        "menu_item_id": "item_id",
        "menu_item": "item_name",
        "item": "item_name",
        "category": "item_type",
        "item_category": "item_type",
        "unit_price": "price",
        "amount": "price",
        "qty": "quantity",
        "restaurant_city": "city",
    }
    rename_map = {col: aliases[col] for col in df.columns if col in aliases}
    return df.rename(columns=rename_map)


def _load_or_synthetic(path: Path, source_name: str, allow_synthetic: bool, seed: int) -> pd.DataFrame:
    if path.exists():
        df = _load_csv_or_parquet(path)
    elif allow_synthetic:
        n_orders = 4000 if source_name == "primary" else 2500
        df = _generate_synthetic_orders(n_orders=n_orders, source=source_name, seed=seed)
    else:
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = normalize_columns(df)
    return _rename_with_aliases(df)


def load_raw_datasets(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_cfg = config.get("data", {})
    seed = int(config.get("project", {}).get("seed", 42))
    allow_synthetic = bool(data_cfg.get("allow_synthetic_fallback", True))

    primary_path = Path(data_cfg.get("primary_orders_path", "data/raw/restaurant_orders.csv"))
    mendeley_path = Path(data_cfg.get("mendeley_orders_path", "data/raw/mendeley_orders.csv"))
    recipe_path = Path(data_cfg.get("recipe_embeddings_path", "data/raw/recipe_embeddings.csv"))

    primary = _load_or_synthetic(primary_path, "primary", allow_synthetic, seed)
    mendeley = _load_or_synthetic(mendeley_path, "mendeley", allow_synthetic, seed + 1)

    if recipe_path.exists():
        recipe = normalize_columns(_load_csv_or_parquet(recipe_path))
    elif allow_synthetic:
        recipe = _generate_synthetic_embeddings(seed + 2)
    else:
        recipe = pd.DataFrame(columns=["item_id"])

    return {
        "primary_orders": primary,
        "mendeley_orders": mendeley,
        "recipe_embeddings": recipe,
    }

