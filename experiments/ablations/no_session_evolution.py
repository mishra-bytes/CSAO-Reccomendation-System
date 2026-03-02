"""
Ablation Study: No Session Evolution Features
===============================================
Retrains the LightGBM ranker without session-position and cart-progression
features to quantify the value of modelling within-order evolution.

Outputs:
  - artifacts/ablations/session_evolution_ablation.csv
  - artifacts/ablations/session_evolution_ablation_report.md

Usage:
    python -m experiments.ablations.no_session_evolution
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts._utils import load_project_config, load_unified_tables, load_feature_tables
from features.complementarity import build_complementarity_lookup
from ranking.training.dataset import build_training_dataset
from ranking.training.train import _temporal_train_valid_split, _group_from_query_ids
from evaluation.metrics.ranking_metrics import ndcg_at_k, precision_at_k, recall_at_k

SESSION_EVOLUTION_COLS = [
    "session_position", "cart_size", "cart_completeness",
    "cart_missing_cats", "cart_missing_cat_ratio",
    "completeness_delta", "fills_meal_gap",
]


def run_session_evolution_ablation(output_dir: str = "artifacts/ablations") -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_project_config()
    ranking_cfg = config.get("ranking", {})
    ranking_params = config.get("lightgbm", {})

    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    print("[ablation] Building training dataset …")
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
    train_idx, val_idx = _temporal_train_valid_split(data, orders, val_frac, seed)

    q_train = _group_from_query_ids([data.query_ids[i] for i in train_idx])
    q_val = _group_from_query_ids([data.query_ids[i] for i in val_idx])
    all_cols = list(data.X.columns)

    def _train_eval(drop_cols: list[str], label: str) -> dict:
        from lightgbm import LGBMRanker
        keep = [c for c in all_cols if c not in drop_cols]
        X_tr = data.X.iloc[train_idx][keep]
        X_va = data.X.iloc[val_idx][keep]

        model = LGBMRanker(**ranking_params)
        model.fit(
            X_tr, data.y[train_idx], group=q_train,
            eval_set=[(X_va, data.y[val_idx])], eval_group=[q_val], eval_at=[10],
        )
        scores = model.predict(X_va)
        preds = pd.DataFrame({
            "query_id": [data.query_ids[i] for i in val_idx],
            "item_id": [data.candidate_items[i] for i in val_idx],
            "label": data.y[val_idx].astype(int),
            "score": scores.astype(float),
        })
        return {
            "variant": label,
            "ndcg_10": ndcg_at_k(preds, k=10),
            "precision_10": precision_at_k(preds, k=10),
            "recall_10": recall_at_k(preds, k=10),
            "n_features": len(keep),
        }

    print("[ablation] Training FULL model …")
    full = _train_eval([], "full_model")

    actual_drop = [c for c in SESSION_EVOLUTION_COLS if c in all_cols]
    print(f"[ablation] Training WITHOUT {actual_drop} …")
    ablated = _train_eval(actual_drop, "no_session_evolution")

    rows = [full, ablated]
    for r in rows:
        r["ndcg_delta"] = r["ndcg_10"] - full["ndcg_10"]

    df = pd.DataFrame(rows)
    csv_path = out / "session_evolution_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ablation] Results saved → {csv_path}")
    print(df.to_string(index=False))

    delta = ablated["ndcg_10"] - full["ndcg_10"]
    md = [
        "# Ablation: Session Evolution Features",
        "",
        "## Setup",
        f"- Dropped features: `{actual_drop}`",
        f"- Training samples: {len(train_idx)}, Validation: {len(val_idx)}",
        "",
        "## Results",
        "",
        "| Variant | NDCG@10 | Precision@10 | Recall@10 | NDCG Δ |",
        "|---------|---------|-------------|-----------|--------|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r['variant']} | {r['ndcg_10']:.4f} | {r['precision_10']:.4f} "
            f"| {r['recall_10']:.4f} | {r['ndcg_delta']:+.4f} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        f"Removing session evolution features causes an NDCG@10 change of **{delta:+.4f}**.",
        "These features model how the cart grows within an order, helping the ranker",
        "understand which items are needed at different stages of meal assembly.",
    ]

    report_path = out / "session_evolution_ablation_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/ablations")
    args = parser.parse_args()
    run_session_evolution_ablation(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

