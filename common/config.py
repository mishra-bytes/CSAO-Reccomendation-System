from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*config_paths: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path_like in config_paths:
        path = Path(path_like)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = _deep_merge(config, data)
    return config

