"""
SHAP Feature Importance Analysis for CSAO LightGBM Ranker
==========================================================
Computes global and per-feature SHAP values using TreeExplainer
on the trained LightGBM LambdaRank model.

Outputs:
  - artifacts/shap/shap_summary.csv            (mean |SHAP| per feature)
  - artifacts/shap/shap_summary_bar.png         (bar plot, top 20)
  - artifacts/shap/shap_beeswarm.png            (beeswarm, top 20)
  - artifacts/shap/shap_dependence_*.png         (top 3 features)
  - artifacts/shap/shap_report.md                (markdown explanation)

Usage:
    python -m experiments.feature_importance.shap_analysis [--n-samples 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
from features.complementarity import build_complementarity_lookup
from ranking.training.dataset import build_training_dataset
from ranking.training.train import _temporal_train_valid_split


def run_shap_analysis(n_samples: int = 500, output_dir: str = "artifacts/shap") -> dict:
    """Run full SHAP analysis on the trained LightGBM model.

    Parameters
    ----------
    n_samples : int
        Number of validation samples to explain (controls speed vs. detail).
    output_dir : str
        Directory to write artefacts.

    Returns
    -------
    dict  with keys: top_features (list), shap_summary_path, plots
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_project_config()
    ranking_cfg = config.get("ranking", {})
    model_path = ranking_cfg.get("model_path", "models/lgbm_ranker.joblib")
    cols_path = ranking_cfg.get("feature_columns_path", "models/feature_columns.json")

    # ── Load model ───────────────────────────────────────────────────────
    print("[shap] Loading model …")
    model = joblib.load(model_path)
    feature_columns = json.loads(Path(cols_path).read_text(encoding="utf-8"))

    # ── Build validation data ────────────────────────────────────────────
    print("[shap] Loading data & building features …")
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    data = build_training_dataset(
        unified=unified,
        user_features=features["user_features"],
        item_features=features["item_features"],
        comp_lookup=comp_lookup,
        config=config,
    )

    orders = unified.get("orders", pd.DataFrame())
    val_frac = float(config.get("train", {}).get("validation_fraction", 0.2))
    seed = int(ranking_cfg.get("random_state", 42))
    _, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)

    X_val = data.X.iloc[val_idx].copy()
    X_val.columns = feature_columns  # ensure alignment

    # Sub-sample for speed
    if n_samples < len(X_val):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_val), size=n_samples, replace=False)
        X_sample = X_val.iloc[idx]
    else:
        X_sample = X_val
    print(f"[shap] Explaining {len(X_sample)} samples ({len(feature_columns)} features) …")

    # ── TreeExplainer ────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Global feature importance ────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = range(1, len(importance_df) + 1)
    csv_path = out / "shap_summary.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"[shap] Summary CSV saved → {csv_path}")

    # ── Summary bar plot (top 20) ────────────────────────────────────────
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    top20 = importance_df.head(20)
    ax_bar.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#4C72B0")
    ax_bar.set_xlabel("Mean |SHAP value|")
    ax_bar.set_title("Top 20 Features — Global SHAP Importance")
    fig_bar.tight_layout()
    bar_path = out / "shap_summary_bar.png"
    fig_bar.savefig(bar_path, dpi=150)
    plt.close(fig_bar)
    print(f"[shap] Bar plot saved → {bar_path}")

    # ── Beeswarm plot (top 20) ───────────────────────────────────────────
    bee_path = None
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample, max_display=20, show=False,
            plot_type="dot",
        )
        bee_path = out / "shap_beeswarm.png"
        plt.tight_layout()
        plt.savefig(bee_path, dpi=150)
        plt.close()
        print(f"[shap] Beeswarm plot saved → {bee_path}")
    except Exception as e:
        print(f"[shap] Beeswarm plot skipped: {e}")

    # ── Dependence plots (top 3 features) ────────────────────────────────
    dep_paths = []
    for i, feat in enumerate(importance_df["feature"].head(3)):
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(
                feat, shap_values, X_sample, show=False,
            )
            dep_path = out / f"shap_dependence_{i+1}_{feat}.png"
            plt.tight_layout()
            plt.savefig(dep_path, dpi=150)
            plt.close()
            dep_paths.append(str(dep_path))
            print(f"[shap] Dependence plot saved → {dep_path}")
        except Exception as e:
            print(f"[shap] Dependence plot for '{feat}' skipped: {e}")

    # ── Markdown report ──────────────────────────────────────────────────
    report_lines = [
        "# SHAP Feature Importance Report",
        "",
        f"**Samples analysed:** {len(X_sample)}",
        f"**Total features:** {len(feature_columns)}",
        "",
        "## Top 20 Features by Mean |SHAP|",
        "",
        "| Rank | Feature | Mean |SHAP| |",
        "|------|---------|-------------|",
    ]
    for _, row in importance_df.head(20).iterrows():
        report_lines.append(
            f"| {int(row['rank'])} | `{row['feature']}` | {row['mean_abs_shap']:.6f} |"
        )

    # Group-level aggregation
    from experiments.ablations.tuning_and_ablation import FEATURE_GROUPS
    report_lines += ["", "## Feature Group Importance", "", "| Group | Sum Mean |SHAP| |", "|-------|-------------|"]
    group_imp = {}
    for grp, cols in FEATURE_GROUPS.items():
        mask = importance_df["feature"].isin(cols)
        grp_total = float(importance_df.loc[mask, "mean_abs_shap"].sum())
        group_imp[grp] = grp_total
    for grp, val in sorted(group_imp.items(), key=lambda x: -x[1]):
        report_lines.append(f"| {grp} | {val:.6f} |")

    report_lines += [
        "",
        "## Plots",
        "",
        f"- Bar chart: `{bar_path}`",
        f"- Beeswarm: `{bee_path}`",
    ]
    for dp in dep_paths:
        report_lines.append(f"- Dependence: `{dp}`")

    report_lines += [
        "",
        "## Interpretation",
        "",
        "The SHAP analysis reveals which features the LightGBM LambdaRank model",
        "relies on most heavily when scoring add-on candidates. Features with high",
        "mean |SHAP| values have the greatest influence on the model's ranking",
        "decisions. The beeswarm plot shows the distribution and direction of each",
        "feature's impact (red = high feature value).",
        "",
        "Group-level aggregation shows which conceptual feature groups contribute",
        "most to the model's predictions, informing future feature engineering.",
    ]

    report_path = out / "shap_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[shap] Report saved → {report_path}")

    return {
        "top_features": importance_df.head(20)["feature"].tolist(),
        "csv_path": str(csv_path),
        "bar_path": str(bar_path),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis for CSAO ranker")
    parser.add_argument("--n-samples", type=int, default=500, help="Samples to explain")
    parser.add_argument("--output-dir", type=str, default="artifacts/shap", help="Output dir")
    args = parser.parse_args()
    run_shap_analysis(n_samples=args.n_samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

