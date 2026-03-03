from __future__ import annotations

from typing import Any

import pandas as pd

from common.io import save_table
from data.preprocessing import preprocess_order_rows
from data.schemas import SCHEMAS, ensure_columns, ensure_not_null


def _normalize_order_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = preprocess_order_rows(df)

    defaults = {
        "quantity": 1,
        "position": 1,
        "city": "unknown",
        "cuisine": "unknown",
        "restaurant_name": "unknown_restaurant",
    }
    for col, value in defaults.items():
        if col not in frame.columns:
            frame[col] = value

    required = ["order_id", "user_id", "restaurant_id", "order_time", "item_id", "item_name", "item_type", "price"]
    ensure_columns(frame, required, f"{source}_orders")

    frame["order_time"] = pd.to_datetime(frame["order_time"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(1).astype(int)
    frame["line_total"] = frame["price"] * frame["quantity"]
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(1).astype(int)

    # Ensure all ID columns are clean strings (no float artifacts like ".0")
    for id_col in ["order_id", "user_id", "restaurant_id", "item_id"]:
        if id_col in frame.columns:
            frame[id_col] = frame[id_col].astype(str).str.replace(r'\.0$', '', regex=True)

    frame["source"] = source
    return frame


def _build_orders(order_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = order_rows.groupby(["order_id", "user_id", "restaurant_id", "source", "city", "cuisine"], as_index=False).agg(
        order_ts=("order_time", "min"),
        total_value=("line_total", "sum"),
    )
    return grouped


def _build_order_items(order_rows: pd.DataFrame) -> pd.DataFrame:
    out = order_rows[
        ["order_id", "item_id", "quantity", "price", "line_total", "order_time", "position", "item_type"]
    ].copy()
    out = out.rename(
        columns={
            "price": "unit_price",
            "order_time": "added_ts",
            "item_type": "item_category",
        }
    )
    out = out.sort_values(["order_id", "position", "item_id"]).reset_index(drop=True)
    return out


def _build_items(order_rows: pd.DataFrame) -> pd.DataFrame:
    items = order_rows.groupby("item_id", as_index=False).agg(
        item_name=("item_name", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        item_category=("item_type", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        item_price=("price", "median"),
    )
    # Propagate is_veg if available in the raw data
    if "is_veg" in order_rows.columns:
        veg_map = order_rows.groupby("item_id")["is_veg"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else True
        ).reset_index()
        veg_map.columns = ["item_id", "is_veg"]
        items = items.merge(veg_map, on="item_id", how="left")
        items["is_veg"] = items["is_veg"].fillna(True)
    else:
        items["is_veg"] = True
    return items


def _build_restaurants(order_rows: pd.DataFrame) -> pd.DataFrame:
    restaurants = (
        order_rows.groupby(["restaurant_id", "restaurant_name", "city", "cuisine"], as_index=False)
        .size()
        .drop(columns=["size"])
    )
    return restaurants


def _build_users(orders: pd.DataFrame) -> pd.DataFrame:
    users = orders.groupby("user_id", as_index=False).agg(
        first_order_ts=("order_ts", "min"),
        last_order_ts=("order_ts", "max"),
        order_count=("order_id", "nunique"),
    )
    return users


def validate_unified_tables(tables: dict[str, pd.DataFrame]) -> None:
    ensure_columns(tables["users"], SCHEMAS.users, "users")
    ensure_columns(tables["orders"], SCHEMAS.orders, "orders")
    ensure_columns(tables["order_items"], SCHEMAS.order_items, "order_items")
    ensure_columns(tables["items"], SCHEMAS.items, "items")
    ensure_columns(tables["restaurants"], SCHEMAS.restaurants, "restaurants")

    ensure_not_null(tables["users"], ["user_id"], "users")
    ensure_not_null(tables["orders"], ["order_id", "user_id", "restaurant_id"], "orders")
    ensure_not_null(tables["order_items"], ["order_id", "item_id"], "order_items")
    ensure_not_null(tables["items"], ["item_id"], "items")
    ensure_not_null(tables["restaurants"], ["restaurant_id"], "restaurants")


def build_unified_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    primary = _normalize_order_rows(raw["primary_orders"], source="primary")
    mendeley = _normalize_order_rows(raw["mendeley_orders"], source="mendeley")

    order_rows = pd.concat([primary, mendeley], axis=0, ignore_index=True)
    orders = _build_orders(order_rows)
    order_items = _build_order_items(order_rows)
    items = _build_items(order_rows)
    restaurants = _build_restaurants(order_rows)
    users = _build_users(orders)

    tables = {
        "users": users,
        "orders": orders,
        "order_items": order_items,
        "items": items,
        "restaurants": restaurants,
        "recipe_embeddings": raw.get("recipe_embeddings", pd.DataFrame()),
    }
    validate_unified_tables(tables)
    return tables


def save_unified_tables(tables: dict[str, pd.DataFrame], processed_dir: str) -> None:
    from pathlib import Path

    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        if name == "recipe_embeddings":
            if df.empty:
                continue
            save_table(df, out_dir / "recipe_embeddings.parquet", index=False)
            continue
        save_table(df, out_dir / f"{name}.parquet", index=False)
