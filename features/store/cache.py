from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from common.io import load_table, save_table


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    save_table(df, path, index=False)


def load_df(path: str | Path) -> Optional[pd.DataFrame]:
    return load_table(path, required=False)
