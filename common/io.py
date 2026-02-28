from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PARQUET_SUFFIXES = {".parquet", ".pq"}


def _fallback_csv_path(path: Path) -> Path:
    if path.suffix:
        return path.with_suffix(".csv")
    return path.with_name(f"{path.name}.csv")


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out.suffix.lower() in PARQUET_SUFFIXES:
            df.to_parquet(out, index=index)
            return out
        df.to_csv(out, index=index)
        return out
    except (ImportError, ValueError, ModuleNotFoundError):
        fallback = _fallback_csv_path(out)
        df.to_csv(fallback, index=index)
        return fallback


def load_table(path: str | Path, required: bool = True) -> Optional[pd.DataFrame]:
    target = Path(path)
    candidates: list[Path] = [target]
    if target.suffix.lower() in PARQUET_SUFFIXES:
        candidates.append(_fallback_csv_path(target))
    elif target.suffix.lower() == ".csv":
        candidates.append(target.with_suffix(".parquet"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in PARQUET_SUFFIXES:
            try:
                return pd.read_parquet(candidate)
            except (ImportError, ValueError, ModuleNotFoundError):
                continue
        return pd.read_csv(candidate)

    if required:
        checked = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Could not load table. Checked: {checked}")
    return None

