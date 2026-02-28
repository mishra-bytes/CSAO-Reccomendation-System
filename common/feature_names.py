from __future__ import annotations

import re
from collections import Counter


_INVALID_CHARS = re.compile(r"[^0-9A-Za-z_]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def normalize_feature_name(name: str) -> str:
    value = _INVALID_CHARS.sub("_", str(name).strip())
    value = _MULTI_UNDERSCORE.sub("_", value).strip("_")
    return value or "feature"


def normalize_feature_columns(columns: list[str]) -> list[str]:
    normalized = [normalize_feature_name(c) for c in columns]
    counts: Counter[str] = Counter()
    output: list[str] = []
    for col in normalized:
        counts[col] += 1
        if counts[col] == 1:
            output.append(col)
        else:
            output.append(f"{col}_{counts[col]}")
    return output

