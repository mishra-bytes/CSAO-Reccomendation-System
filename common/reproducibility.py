"""
Reproducibility & Model Versioning Utilities
===============================================
Provides:
  - Global random seed control
  - Training config snapshot saving
  - Model card generation

Usage:
    from common.reproducibility import set_global_seed, save_model_card
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across all relevant libraries for reproducibility."""
    np.random.seed(seed)

    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    # LightGBM uses its own seed via random_state param
    # Set env var for any library that reads PYTHONHASHSEED
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_training_snapshot(
    config: dict[str, Any],
    feature_columns: list[str],
    training_summary: dict[str, Any],
    output_dir: str = "models",
) -> Path:
    """Save a complete training configuration snapshot for reproducibility."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "training_summary": training_summary,
    }

    # Add package versions
    try:
        import lightgbm
        snapshot["lightgbm_version"] = lightgbm.__version__
    except Exception:
        pass
    try:
        import sklearn
        snapshot["sklearn_version"] = sklearn.__version__
    except Exception:
        pass

    path = out / "training_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path


def generate_model_card(
    model_name: str = "CSAO LightGBM LambdaRank Ranker",
    config: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    eval_metrics: dict[str, Any] | None = None,
    output_path: str = "models/model_card.md",
) -> Path:
    """Generate a model card documenting the trained model."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    config = config or {}
    training_summary = training_summary or {}
    eval_metrics = eval_metrics or {}

    lgbm_cfg = config.get("lightgbm", {})
    ranking_cfg = config.get("ranking", {})

    lines = [
        f"# Model Card: {model_name}",
        "",
        f"**Generated:** {now}",
        f"**Model type:** LightGBM LambdaRank (gradient-boosted decision trees)",
        f"**Task:** Cart Super Add-On recommendation ranking (Learning-to-Rank)",
        f"**Framework:** LightGBM {_get_version('lightgbm')}",
        "",
        "## Intended Use",
        "",
        "Ranks candidate add-on items for a user's current cart in a food delivery",
        "context. The model scores ~200 candidates and outputs a ranked list optimised",
        "for NDCG (Normalised Discounted Cumulative Gain).",
        "",
        "## Training Data",
        "",
        "- **Source:** Synthetic Indian food delivery orders + Mendeley Takeaway dataset",
        f"- **Training rows:** {training_summary.get('train_rows', 'N/A')}",
        f"- **Validation rows:** {training_summary.get('val_rows', 'N/A')}",
        f"- **Features:** {training_summary.get('num_features', 'N/A')}",
        f"- **Split method:** {training_summary.get('split_method', 'temporal')}",
        "",
        "## Hyperparameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in lgbm_cfg.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        f"**Random seed:** {ranking_cfg.get('random_state', 42)}",
        f"**Negative samples per positive:** {ranking_cfg.get('negative_samples_per_positive', 6)}",
        "",
        "## Evaluation Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Limitations",
        "",
        "- Trained on synthetic data; metrics may not transfer to production.",
        "- No real-time feature freshness (user features are batch-computed).",
        "- Cold-start users/items rely on fallback strategies, not the main model.",
        "",
        "## Ethical Considerations",
        "",
        "- No user demographic data is used in features.",
        "- Popularity bias is mitigated via MMR diversity reranking.",
        "- Price fairness is monitored (see diversity & fairness report).",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _get_version(package: str) -> str:
    try:
        import importlib
        mod = importlib.import_module(package)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"
