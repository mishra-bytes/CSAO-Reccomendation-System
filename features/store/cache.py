from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def load_df(path: str | Path) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)

