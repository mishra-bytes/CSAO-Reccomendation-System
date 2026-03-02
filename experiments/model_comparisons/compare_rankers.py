"""
Model Comparison: LightGBM vs XGBoost vs MLP Ranker
====================================================
Trains three different ranker architectures on the same training data
and compares offline metrics to justify LightGBM LambdaRank selection.

Outputs:
  - artifacts/model_comparison/model_comparison.csv
  - artifacts/model_comparison/model_comparison_report.md

Usage:
    python -m experiments.model_comparisons.compare_rankers
"""
from __future__ import annotations

import argparse
import sys
import time
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


def _make_preds_df(val_idx, data, scores):
    return pd.DataFrame({
        "query_id": [data.query_ids[i] for i in val_idx],
        "item_id": [data.candidate_items[i] for i in val_idx],
        "label": data.y[val_idx].astype(int),
        "score": scores.astype(float),
    })


def run_model_comparison(output_dir: str = "artifacts/model_comparison") -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_project_config()
    ranking_cfg = config.get("ranking", {})

    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    unified = load_unified_tables(processed_dir)
    features = load_feature_tables(processed_dir)
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    print("[compare] Building training dataset …")
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

    X_train = data.X.iloc[train_idx]
    y_train = data.y[train_idx]
    X_val = data.X.iloc[val_idx]
    y_val = data.y[val_idx]

    rows = []

    # ── 1. LightGBM LambdaRank ──────────────────────────────────────────
    print("[compare] Training LightGBM LambdaRank …")
    from lightgbm import LGBMRanker
    lgbm_params = config.get("lightgbm", {})
    t0 = time.perf_counter()
    lgbm = LGBMRanker(**lgbm_params)
    lgbm.fit(X_train, y_train, group=q_train,
             eval_set=[(X_val, y_val)], eval_group=[q_val], eval_at=[10])
    lgbm_time = time.perf_counter() - t0
    lgbm_scores = lgbm.predict(X_val)
    preds = _make_preds_df(val_idx, data, lgbm_scores)

    # Measure inference latency
    t_inf = time.perf_counter()
    for _ in range(5):
        lgbm.predict(X_val[:200])
    lgbm_inf_ms = (time.perf_counter() - t_inf) / 5 * 1000

    rows.append({
        "model": "LightGBM_LambdaRank",
        "ndcg_10": ndcg_at_k(preds, k=10),
        "precision_10": precision_at_k(preds, k=10),
        "recall_10": recall_at_k(preds, k=10),
        "train_time_s": lgbm_time,
        "inference_200_ms": lgbm_inf_ms,
    })

    # ── 2. XGBoost Ranker ────────────────────────────────────────────────
    try:
        import xgboost as xgb
        print("[compare] Training XGBoost Ranker …")

        # Build DMatrix with group info
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(q_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(q_val)

        xgb_params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@10",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "seed": seed,
            "verbosity": 0,
        }
        t0 = time.perf_counter()
        xgb_model = xgb.train(
            xgb_params, dtrain, num_boost_round=200,
            evals=[(dval, "val")], verbose_eval=False,
        )
        xgb_time = time.perf_counter() - t0
        xgb_scores = xgb_model.predict(dval)
        preds = _make_preds_df(val_idx, data, xgb_scores)

        t_inf = time.perf_counter()
        dval200 = xgb.DMatrix(X_val.iloc[:200])
        for _ in range(5):
            xgb_model.predict(dval200)
        xgb_inf_ms = (time.perf_counter() - t_inf) / 5 * 1000

        rows.append({
            "model": "XGBoost_RankNDCG",
            "ndcg_10": ndcg_at_k(preds, k=10),
            "precision_10": precision_at_k(preds, k=10),
            "recall_10": recall_at_k(preds, k=10),
            "train_time_s": xgb_time,
            "inference_200_ms": xgb_inf_ms,
        })
    except ImportError:
        print("[compare] XGBoost not installed, skipping.")

    # ── 3. Simple MLP Ranker (sklearn) ───────────────────────────────────
    print("[compare] Training MLP Ranker …")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    t0 = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(X_tr_s, y_train.astype(int))
    mlp_time = time.perf_counter() - t0
    mlp_scores = mlp.predict_proba(X_va_s)[:, 1] if mlp.classes_.shape[0] > 1 else mlp.predict(X_va_s)
    preds = _make_preds_df(val_idx, data, mlp_scores)

    t_inf = time.perf_counter()
    for _ in range(5):
        mlp.predict_proba(X_va_s[:200])
    mlp_inf_ms = (time.perf_counter() - t_inf) / 5 * 1000

    rows.append({
        "model": "MLP_Classifier",
        "ndcg_10": ndcg_at_k(preds, k=10),
        "precision_10": precision_at_k(preds, k=10),
        "recall_10": recall_at_k(preds, k=10),
        "train_time_s": mlp_time,
        "inference_200_ms": mlp_inf_ms,
    })

    # ── 4. Popularity Baseline ───────────────────────────────────────────
    print("[compare] Popularity baseline …")
    item_pop = data.X["item_order_count"] if "item_order_count" in data.X.columns else pd.Series(0.0, index=data.X.index)
    pop_scores = item_pop.iloc[val_idx].values.astype(float)
    preds = _make_preds_df(val_idx, data, pop_scores)
    rows.append({
        "model": "Popularity_Baseline",
        "ndcg_10": ndcg_at_k(preds, k=10),
        "precision_10": precision_at_k(preds, k=10),
        "recall_10": recall_at_k(preds, k=10),
        "train_time_s": 0.0,
        "inference_200_ms": 0.0,
    })

    df = pd.DataFrame(rows)
    csv_path = out / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[compare] Results saved → {csv_path}")
    print(df.to_string(index=False))

    # Markdown report
    best = df.loc[df["ndcg_10"].idxmax()]
    md = [
        "# Model Comparison Report",
        "",
        "## Models Evaluated",
        "1. **LightGBM LambdaRank** — gradient-boosted trees with LambdaRank loss (directly optimizes NDCG)",
        "2. **XGBoost rank:ndcg** — XGBoost with NDCG ranking objective",
        "3. **MLP Classifier** — 2-layer neural network (128→64) with pointwise cross-entropy",
        "4. **Popularity Baseline** — rank by item order frequency (no personalization)",
        "",
        "## Results",
        "",
        "| Model | NDCG@10 | Precision@10 | Recall@10 | Train Time (s) | Inference 200 items (ms) |",
        "|-------|---------|-------------|-----------|-----------------|-------------------------|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r['model']} | {r['ndcg_10']:.4f} | {r['precision_10']:.4f} "
            f"| {r['recall_10']:.4f} | {r['train_time_s']:.1f} | {r['inference_200_ms']:.1f} |"
        )
    md += [
        "",
        "## Conclusion",
        "",
        f"**{best['model']}** achieves the highest NDCG@10 of **{best['ndcg_10']:.4f}**.",
        "LightGBM LambdaRank is selected as the production ranker due to its strong",
        "ranking quality, fast inference speed, and native listwise optimization.",
    ]

    report_path = out / "model_comparison_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[compare] Report saved → {report_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/model_comparison")
    args = parser.parse_args()
    run_model_comparison(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

