"""
SHAP Feature Importance Analysis
=================================
Computes SHAP values on the trained LightGBM ranker to explain
which features drive recommendations most.

Outputs:
  - Top-N feature importance bar chart (saved as PNG)
  - Per-feature SHAP summary (JSON)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_model_and_data(
    model_path: Path | None = None,
) -> tuple:
    """Load trained LightGBM model and validation data."""
    from scripts._utils import load_project_config
    from data.loaders import load_processed
    from ranking.training.dataset import build_training_dataset
    from ranking.training.train import _temporal_train_valid_split

    config = load_project_config()
    model_p = model_path or ROOT / "models" / "lgbm_ranker.joblib"
    model = joblib.load(model_p)

    unified = load_processed()
    user_features = pd.read_parquet(ROOT / "data" / "processed" / "features_user.parquet")
    item_features = pd.read_parquet(ROOT / "data" / "processed" / "features_item.parquet")
    comp = pd.read_parquet(ROOT / "data" / "processed" / "features_complementarity.parquet")

    comp_lookup: dict = {}
    for _, row in comp.iterrows():
        a, b = str(row.get("item_a", "")), str(row.get("item_b", ""))
        if a and b:
            comp_lookup[(a, b)] = row.to_dict()

    ranking_cfg = config.get("ranking", {})
    data = build_training_dataset(
        unified=unified,
        user_features=user_features,
        item_features=item_features,
        comp_lookup=comp_lookup,
        config={"ranking": ranking_cfg},
    )

    orders = unified.get("orders", pd.DataFrame())
    val_frac = float(config.get("train", {}).get("validation_fraction", 0.2))
    seed = int(ranking_cfg.get("random_state", 42))
    _, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)
    X_val = data.X.iloc[val_idx]

    return model, X_val, data.X.columns.tolist()


def run_shap_analysis(
    max_samples: int = 2000,
    top_n: int = 20,
    save_dir: Path | None = None,
) -> dict:
    """Run SHAP TreeExplainer on the LightGBM ranker.

    Returns dict with:
      - feature_importance: list of (feature, mean_abs_shap) sorted desc
    """
    import shap

    model, X_val, feature_names = load_model_and_data()

    # Subsample for speed
    if len(X_val) > max_samples:
        X_sample = X_val.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_val

    print(f"[SHAP] Computing SHAP values for {len(X_sample)} samples, "
          f"{len(feature_names)} features...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    result = {
        "n_samples": len(X_sample),
        "n_features": len(feature_names),
        "feature_importance": [
            {"feature": f, "mean_abs_shap": round(v, 6)}
            for f, v in importance[:top_n]
        ],
        "bottom_features": [
            {"feature": f, "mean_abs_shap": round(v, 6)}
            for f, v in importance[-5:]
        ],
    }

    # Save outputs
    out_dir = save_dir or ROOT / "experiments" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "shap_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SHAP] Results saved to {out_dir / 'shap_results.json'}")

    # Try to save summary plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X_sample, show=False, max_display=top_n)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Summary plot saved to {out_dir / 'shap_summary.png'}")
        result["plot_saved"] = True
    except Exception as e:
        print(f"[SHAP] Could not save plot: {e}")
        result["plot_saved"] = False

    return result


def main() -> None:
    result = run_shap_analysis()
    print("\n=== SHAP Feature Importance (top 20) ===")
    for i, entry in enumerate(result["feature_importance"], 1):
        print(f"  {i:2d}. {entry['feature']:40s}  {entry['mean_abs_shap']:.6f}")
    print(f"\nSamples analyzed: {result['n_samples']}")


if __name__ == "__main__":
    main()

