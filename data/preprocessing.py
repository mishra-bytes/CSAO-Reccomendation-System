from __future__ import annotations

import pandas as pd


def preprocess_order_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
    if "quantity" in out.columns:
        out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    if "order_time" in out.columns:
        out["order_time"] = pd.to_datetime(out["order_time"], errors="coerce")
    return out

