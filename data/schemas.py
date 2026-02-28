from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class UnifiedSchemas:
    users: tuple[str, ...] = ("user_id", "first_order_ts", "last_order_ts", "order_count")
    orders: tuple[str, ...] = (
        "order_id",
        "user_id",
        "restaurant_id",
        "order_ts",
        "total_value",
        "source",
        "city",
        "cuisine",
    )
    order_items: tuple[str, ...] = (
        "order_id",
        "item_id",
        "quantity",
        "unit_price",
        "line_total",
        "added_ts",
        "position",
        "item_category",
    )
    items: tuple[str, ...] = ("item_id", "item_name", "item_category", "item_price")
    restaurants: tuple[str, ...] = ("restaurant_id", "restaurant_name", "city", "cuisine")


SCHEMAS = UnifiedSchemas()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def ensure_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def ensure_not_null(df: pd.DataFrame, cols: Iterable[str], table_name: str) -> None:
    null_cols = [c for c in cols if df[c].isna().any()]
    if null_cols:
        raise ValueError(f"{table_name} contains nulls in key columns: {null_cols}")

