"""
Comprehensive Evaluation Runner
=================================
Runs ALL evaluation components end-to-end and produces a single unified report.

Usage:
    python scripts/run_full_evaluation.py [--skip-hp-tuning] [--skip-neural]

Output:
    - Console: formatted summary
    - data/processed/full_evaluation_report.json: machine-readable results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.io import load_table
from features.complementarity import build_complementarity_lookup
from scripts._utils import load_feature_tables, load_project_config, load_unified_tables


def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hp-tuning", action="store_true", help="Skip Optuna HP tuning (slow)")
    parser.add_argument("--skip-neural", action="store_true", help="Skip neural reranker training")
    parser.add_argument("--hp-trials", type=int, default=15, help="Number of Optuna trials")
    args = parser.parse_args()

    config = load_project_config()
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    ranking_cfg = config.get("ranking", {})

    # Load all data
    _sep("Loading Data")
    unified = load_unified_tables(str(processed_dir))
    features = load_feature_tables(str(processed_dir))
    comp_lookup = build_complementarity_lookup(features["complementarity"])

    pred_path = Path(ranking_cfg.get("validation_predictions_path", processed_dir / "validation_predictions.parquet"))
    meta_path = Path(ranking_cfg.get("query_meta_path", processed_dir / "query_meta.parquet"))
    predictions = load_table(pred_path, required=True)
    query_meta = load_table(meta_path, required=True)
    k = int(ranking_cfg.get("eval_k", 10))

    print(f"  Predictions: {len(predictions)} rows, {predictions['query_id'].nunique()} queries")
    print(f"  Items: {len(unified['items'])}, Users: {len(unified['users'])}")

    report = {}

    # ── 1. Standard Offline Eval ──────────────────────────────────────────
    _sep("1. Standard Offline Evaluation")
    from evaluation.run_eval import evaluate_offline

    std = evaluate_offline(
        predictions=predictions,
        item_catalog=unified["items"],
        query_meta=query_meta,
        user_features=features["user_features"],
        k=k,
    )
    print("  Overall:")
    for m, v in std["overall"].items():
        print(f"    {m}: {v:.4f}")
    report["standard_eval"] = {m: round(float(v), 4) for m, v in std["overall"].items()}

    # ── 2. Baseline Comparison ────────────────────────────────────────────
    _sep("2. Baseline Comparison")
    try:
        from experiments.baselines import run_baseline_comparison

        bl_results = run_baseline_comparison(
            unified=unified,
            user_features=features["user_features"],
            item_features=features["item_features"],
            comp_lookup=comp_lookup,
            config=config,
        )
        print(bl_results.to_string())
        report["baseline_comparison"] = bl_results.to_dict(orient="index")
    except Exception as e:
        print(f"  Skipped: {e}")
        report["baseline_comparison"] = {"error": str(e)}

    # ── 3. Feature Ablation ───────────────────────────────────────────────
    _sep("3. Feature Group Ablation")
    try:
        from experiments.ablations.tuning_and_ablation import run_feature_ablation

        ablation = run_feature_ablation(
            unified=unified,
            user_features=features["user_features"],
            item_features=features["item_features"],
            comp_lookup=comp_lookup,
            config=config,
        )
        print(ablation.to_string(index=False))
        report["feature_ablation"] = ablation.to_dict(orient="records")
    except Exception as e:
        print(f"  Skipped: {e}")
        report["feature_ablation"] = {"error": str(e)}

    # ── 4. Cold-Start Segment Analysis ────────────────────────────────────
    _sep("4. Cold-Start Segment Analysis")
    try:
        from serving.pipeline.cold_start import evaluate_cold_start_segments

        cs = evaluate_cold_start_segments(
            predictions=predictions,
            query_meta=query_meta,
            user_features=features["user_features"],
            min_orders_for_warm=3,
            k=k,
        )
        print(cs.to_string(index=False))
        report["cold_start_segments"] = cs.to_dict(orient="records")
    except Exception as e:
        print(f"  Skipped: {e}")
        report["cold_start_segments"] = {"error": str(e)}

    # ── 5. Business Impact Model ──────────────────────────────────────────
    _sep("5. Business Impact Projection")
    try:
        from evaluation.business_impact_model import (
            compute_business_impact,
            format_executive_summary,
        )

        biz = compute_business_impact(
            predictions=predictions,
            item_catalog=unified["items"],
            user_features=features["user_features"],
            k=k,
        )
        print(format_executive_summary(biz))
        print("\n  Segment Impact:")
        print(biz.segment_impact.to_string(index=False))
        print("\n  A/B Test Plan:")
        for k2, v2 in biz.ab_test_plan.items():
            print(f"    {k2}: {v2}")

        report["business_impact"] = {
            "ndcg_10": biz.ndcg_at_10,
            "precision_10": biz.precision_at_10,
            "attach_rate": biz.attach_rate,
            "aov_uplift_percent": biz.aov_uplift_percent,
            "daily_revenue_inr": biz.daily_incremental_revenue,
            "monthly_revenue_inr": biz.monthly_incremental_revenue,
            "annual_revenue_inr": biz.annual_incremental_revenue,
            "ab_test_plan": biz.ab_test_plan,
        }
    except Exception as e:
        print(f"  Skipped: {e}")
        report["business_impact"] = {"error": str(e)}

    # ── 6. Optuna HP Tuning ───────────────────────────────────────────────
    if not args.skip_hp_tuning:
        _sep("6. Hyperparameter Tuning (Optuna)")
        try:
            from experiments.ablations.tuning_and_ablation import run_optuna_tuning

            hp = run_optuna_tuning(
                unified=unified,
                user_features=features["user_features"],
                item_features=features["item_features"],
                comp_lookup=comp_lookup,
                config=config,
                n_trials=args.hp_trials,
                timeout_seconds=300,
            )
            print(f"  Best NDCG@10: {hp['best_ndcg']:.4f}")
            print(f"  Best Params: {json.dumps(hp['best_params'], indent=2)}")
            print(f"  Trials completed: {hp['n_trials_completed']}")
            report["hp_tuning"] = hp
        except Exception as e:
            print(f"  Skipped: {e}")
            report["hp_tuning"] = {"error": str(e)}
    else:
        print("\n  [Skipped HP tuning]")
        report["hp_tuning"] = {"skipped": True}

    # ── 7. Neural Reranker ────────────────────────────────────────────────
    if not args.skip_neural:
        _sep("7. Neural Cross-Attention Reranker")
        try:
            from ranking.inference.neural_reranker import (
                NeuralReranker,
                build_training_triplets,
                train_reranker,
            )

            triplets = build_training_triplets(predictions, top_k=k)
            if len(triplets) < 10:
                print("  Not enough triplets for neural training, skipping.")
                report["neural_reranker"] = {"skipped": True, "reason": "insufficient triplets"}
            else:
                print(f"  Training triplets: {len(triplets)}")
                reranker = train_reranker(triplets, epochs=10, lr=1e-3)
                if reranker is not None:
                    print(f"  Neural reranker trained successfully")
                    report["neural_reranker"] = {"trained": True, "n_triplets": len(triplets)}
                else:
                    report["neural_reranker"] = {"skipped": True, "reason": "PyTorch not available"}
        except Exception as e:
            print(f"  Skipped: {e}")
            report["neural_reranker"] = {"error": str(e)}
    else:
        print("\n  [Skipped neural reranker]")
        report["neural_reranker"] = {"skipped": True}

    # ── 8. Scalability Analysis ───────────────────────────────────────────
    _sep("8. Scalability Analysis")
    try:
        from serving.scalability import (
            compute_capacity_plan,
            deployment_spec,
            monitoring_plan,
            simulate_cache_performance,
        )

        # Simulate cache with item IDs from predictions
        item_lists = []
        for _, grp in predictions.groupby("query_id"):
            items = grp.sort_values("score", ascending=False).head(k)["item_id"].astype(str).tolist()
            item_lists.append(items)

        cache = simulate_cache_performance(item_lists[:500])
        print(f"  Cache hit rate: {cache.hit_rate:.1%}")
        print(f"  Avg latency saving: {cache.estimated_latency_saving_ms:.1f}ms per lookup")

        cap = compute_capacity_plan(peak_qps=500)
        print(f"  Pods needed: {cap['pods_needed_min']} (min) / {cap['pods_recommended']} (recommended)")
        print(f"  Est. monthly cost: ${cap['estimated_cost_monthly_usd']}")

        deploy = deployment_spec()
        mon = monitoring_plan()

        report["scalability"] = {
            "cache_hit_rate": cache.hit_rate,
            "cache_latency_saving_ms": cache.estimated_latency_saving_ms,
            "capacity_plan": cap,
            "deployment_services": list(deploy["services"].keys()),
            "monitoring_dashboards": list(mon["dashboards"].keys()),
        }
    except Exception as e:
        print(f"  Skipped: {e}")
        report["scalability"] = {"error": str(e)}

    # ── Save Report ───────────────────────────────────────────────────────
    _sep("REPORT SAVED")
    report_path = processed_dir / "full_evaluation_report.json"

    # JSON-safe conversion
    def _safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return str(obj)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_safe)
    print(f"  Report: {report_path}")

    # Final summary
    _sep("FINAL SCORE SUMMARY")
    print(f"  NDCG@10:              {report.get('standard_eval', {}).get('ndcg_at_k', 'N/A')}")
    print(f"  Precision@10:         {report.get('standard_eval', {}).get('precision_at_k', 'N/A')}")

    bl = report.get("baseline_comparison", {})
    if isinstance(bl, dict) and "error" not in bl:
        for model_name, metrics in bl.items():
            if isinstance(metrics, dict):
                ndcg_val = metrics.get("ndcg_at_k", "N/A")
                print(f"  {model_name} NDCG@10: {ndcg_val}")

    biz = report.get("business_impact", {})
    if "aov_uplift_percent" in biz:
        print(f"  AOV Uplift:           +{biz['aov_uplift_percent']:.2f}%")
        print(f"  Annual Revenue (INR): {biz['annual_revenue_inr']:,.0f}")

    hp = report.get("hp_tuning", {})
    if "best_ndcg" in hp:
        print(f"  Best HP NDCG@10:      {hp['best_ndcg']:.4f}")


if __name__ == "__main__":
    main()
